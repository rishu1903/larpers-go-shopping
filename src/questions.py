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


# --------------------------------------------------
# EARLY-TURN DOMINANT ATTRIBUTE TARGETING
# --------------------------------------------------
#
# Turns 1-3 default to broad discovery ("other")
# because a small/ambiguous candidate pool usually
# carries no reliable signal yet. But a real,
# production-scale retrieval pool (up to 30
# candidates) can already show an overwhelmingly
# dominant attribute even before any evidence has
# been given -- e.g. every candidate sharing one
# material while spanning many colors. Asking about
# that dominant attribute immediately, instead of
# spending a turn on the generic catch-all, gets
# useful evidence one turn sooner.
#
# Deliberately conservative: requires BOTH a much
# higher information-score bar than the turn 4+
# threshold AND a minimum pool size, so this never
# fires on small/synthetic candidate sets (the
# entropy-based score can already be near its
# theoretical maximum on a clean 2-item split).

EARLY_TURN_MIN_POOL = 10

EARLY_TURN_INFO_THRESHOLD = 0.50


def _scored_attributes(
    state: SessionState,
    candidates: list[dict],
    threshold: float,
) -> list[
    tuple[
        float,
        str,
    ]
]:
    """
    Score every attribute not already known/asked/
    declined, keeping only those above `threshold`.
    """

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

        if information > threshold:

            scored.append(
                (
                    information,
                    attribute,
                )
            )

    return scored


def choose_candidate_attribute(
    state: SessionState,
    candidates: list[dict],
    turn: int,
) -> str:
    """
    Choose the next clarification dimension.

    Turns 1-3
    ---------

    Default to broad discovery with `other`, UNLESS
    the candidate pool is large enough to be a real
    retrieval result AND one attribute is
    overwhelmingly dominant (see
    EARLY_TURN_MIN_POOL / EARLY_TURN_INFO_THRESHOLD
    above) -- in that case ask about it directly
    instead of wasting a turn on the catch-all.

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

        if (
            len(candidates)
            >= EARLY_TURN_MIN_POOL
        ):

            early_scored = (
                _scored_attributes(
                    state,
                    candidates,
                    EARLY_TURN_INFO_THRESHOLD,
                )
            )

            if early_scored:

                return _profile_aware_choice(
                    state=state,
                    scored=early_scored,
                )

        return "other"

    scored = _scored_attributes(
        state,
        candidates,
        0.10,
    )

    if not scored:
        return "other"

    return _profile_aware_choice(
        state=state,
        scored=scored,
    )