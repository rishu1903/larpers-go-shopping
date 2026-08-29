from __future__ import annotations

import math
import re
from collections import Counter

from src.profile import (
    attribute_affinity,
)

from src.state import (
    SessionState,
)


TOKEN_RE = re.compile(
    r"[a-z0-9]+",
    re.IGNORECASE,
)


ATTRIBUTE_VALUES: dict[
    str,
    tuple[str, ...],
] = {
    "material": (
        "cotton",
        "polyester",
        "nylon",
        "leather",
        "wool",
        "spandex",
        "silk",
        "rayon",
        "fabric",
    ),

    "color": (
        "black",
        "white",
        "blue",
        "red",
        "pink",
        "green",
        "brown",
        "gray",
        "grey",
        "purple",
        "yellow",
        "orange",
    ),

    "size": (
        "small",
        "medium",
        "large",
        "xl",
        "xxl",
        "wide",
        "narrow",
        "petite",
        "plus",
        "oversized",
    ),

    "style": (
        "slim",
        "regular",
        "relaxed",
        "loose",
        "fitted",
        "sleeveless",
        "sleeve",
        "neck",
        "vneck",
        "crew",
        "cropped",
        "classic",
    ),

    "use_case": (
        "hiking",
        "running",
        "gym",
        "winter",
        "outdoor",
        "work",
        "walking",
        "sports",
        "travel",
        "casual",
    ),

    "feature": (
        "waterproof",
        "pocket",
        "pockets",
        "zipper",
        "button",
        "buttons",
        "drawstring",
        "hood",
        "hooded",
        "closure",
        "stretch",
        "lightweight",
        "breathable",
        "insulated",
        "nonslip",
        "slip",
        "sole",
        "adjustable",
    ),
}


QUESTION_PRIORITY = {
    "material": 1.20,
    "color": 1.10,
    "size": 0.95,
    "style": 1.00,
    "use_case": 1.00,
    "feature": 0.90,
}


# --------------------------------------------------
# PROFILE SAFETY GATE
# --------------------------------------------------
#
# Historical profile information can influence
# question choice only when another attribute is
# already nearly as informative as the best one.
#
# Example:
#
#     color information = 1.00
#     fit information   = 0.95
#
# and profile says fit matters:
#
#     fit may win.
#
# But:
#
#     material = 1.00
#     style    = 0.30
#
# profile preference for style must NOT overpower
# the much stronger current-session evidence.

PROFILE_NEAR_TIE_RATIO = 0.90


def _tokens(
    text: object,
) -> set[str]:
    """
    Normalize text into a token set.
    """

    return set(
        TOKEN_RE.findall(
            str(
                text
            ).lower()
        )
    )


def _known_attributes(
    state: SessionState,
) -> set[str]:
    """
    Infer which attribute families are already
    represented in current-session evidence.

    Historical profile information does NOT count
    as a known current preference.
    """

    text = " ".join(
        item.text

        for item
        in state.evidence
    )

    tokens = _tokens(
        text
    )

    known: set[
        str
    ] = set()

    for (
        attribute,
        values,
    ) in ATTRIBUTE_VALUES.items():

        if any(
            value in tokens

            for value
            in values
        ):

            known.add(
                attribute
            )

    return known


def _attribute_information(
    candidates: list[dict],
    attribute: str,
) -> float:
    """
    Estimate how informative an attribute would
    be for separating the current candidates.

    Higher values mean:

        - good coverage across candidates
        - multiple possible values
        - reasonably diverse value distribution

    Normalized entropy approximates information
    gain.
    """

    values = ATTRIBUTE_VALUES[
        attribute
    ]

    if not candidates:
        return 0.0

    counts: Counter[
        str
    ] = Counter()

    covered = 0

    sample = candidates[
        :30
    ]

    for candidate in sample:

        tokens = _tokens(
            candidate.get(
                "searchable_text",
                "",
            )
        )

        found = [
            value

            for value
            in values

            if value in tokens
        ]

        if not found:
            continue

        covered += 1

        counts.update(
            found
        )

    if (
        covered < 2
        or
        len(counts) < 2
    ):

        return 0.0

    total = sum(
        counts.values()
    )

    entropy = 0.0

    for count in counts.values():

        probability = (
            count
            / total
        )

        entropy -= (
            probability
            * math.log(
                probability
            )
        )

    max_entropy = math.log(
        len(
            counts
        )
    )

    normalized_entropy = (
        entropy
        / max_entropy

        if max_entropy > 0

        else 0.0
    )

    coverage = (
        covered
        / len(
            sample
        )
    )

    return (
        coverage
        * normalized_entropy
        * QUESTION_PRIORITY[
            attribute
        ]
    )


def _profile_aware_choice(
    state: SessionState,
    scored: list[
        tuple[
            float,
            str,
        ]
    ],
) -> str:
    """
    Safely use the aggregate profile as a
    near-tie breaker.

    Candidate uncertainty remains primary.

    Profile affinity can only choose between
    attributes whose information score is at
    least 90% of the strongest candidate-driven
    question.

    This prevents historical behaviour from
    overriding the customer's current session.
    """

    scored.sort(
        key=lambda item: (
            -item[0],
            item[1],
        )
    )

    (
        best_information,
        best_attribute,
    ) = scored[0]

    affinities = (
        attribute_affinity(
            state.user_profile
        )
    )

    if not affinities:
        return best_attribute

    threshold = (
        best_information
        * PROFILE_NEAR_TIE_RATIO
    )

    near_best = [
        (
            information,
            attribute,
        )

        for (
            information,
            attribute,
        ) in scored

        if information >= threshold
    ]

    profile_supported = [
        (
            affinities.get(
                attribute,
                0.0,
            ),
            information,
            attribute,
        )

        for (
            information,
            attribute,
        ) in near_best

        if (
            affinities.get(
                attribute,
                0.0,
            )
            > 0.0
        )
    ]

    if not profile_supported:
        return best_attribute

    profile_supported.sort(
        key=lambda item: (
            -item[0],
            -item[1],
            item[2],
        )
    )

    return profile_supported[
        0
    ][2]


def choose_candidate_attribute(
    state: SessionState,
    candidates: list[dict],
    turn: int,
) -> str:
    """
    Choose the next clarification dimension.

    Turns 1-3
    ---------

    Keep broad discovery with `other`.

    Turn 4+
    -------

    1. Estimate candidate information gain.
    2. Remove attributes already known/asked/
       explicitly declined.
    3. Find the strongest candidate-driven
       clarification.
    4. Allow the anonymized aggregate profile
       to break only a near tie.
    5. Fall back to `other` when no structured
       attribute is useful.

    No target label or hidden simulator state
    is available to this policy.
    """

    if turn <= 3:
        return "other"

    known = _known_attributes(
        state
    )

    scored: list[
        tuple[
            float,
            str,
        ]
    ] = []

    for attribute in ATTRIBUTE_VALUES:

        if attribute in known:
            continue

        if (
            attribute
            in state.asked_attributes
        ):
            continue

        if (
            attribute
            in state.no_preference
        ):
            continue

        information = (
            _attribute_information(
                candidates,
                attribute,
            )
        )

        if information > 0.10:

            scored.append(
                (
                    information,
                    attribute,
                )
            )

    if not scored:
        return "other"

    return _profile_aware_choice(
        state=state,
        scored=scored,
    )