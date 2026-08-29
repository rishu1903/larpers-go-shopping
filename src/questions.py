from __future__ import annotations

import math
import re
from collections import Counter

from src.state import SessionState


TOKEN_RE = re.compile(
    r"[a-z0-9]+",
    re.IGNORECASE,
)


# Candidate signals that can justify asking
# a structured clarification question.
#
# These are deliberately lightweight and local.
# We are not using an LLM to choose the question.
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


# Small priors.
#
# Materials and colors tend to be particularly
# useful discriminators in this catalogue.
QUESTION_PRIORITY = {
    "material": 1.20,
    "color": 1.10,
    "size": 0.95,
    "style": 1.00,
    "use_case": 1.00,
    "feature": 0.90,
}


def _tokens(
    text: object,
) -> set[str]:
    """
    Normalize text into a token set.
    """

    return set(
        TOKEN_RE.findall(
            str(text).lower()
        )
    )


def _known_attributes(
    state: SessionState,
) -> set[str]:
    """
    Infer which attribute families have already
    been expressed by the customer.

    We should not ask:

        "What material do you want?"

    if they have already told us:

        "cotton"
    """

    text = " ".join(
        item.text
        for item
        in state.evidence
    )

    tokens = _tokens(
        text
    )

    known: set[str] = set()

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

    Higher score means:

        - the attribute occurs in many candidates
        - candidates contain different values
        - the distribution is reasonably diverse

    We approximate information gain using
    normalized entropy.
    """

    values = ATTRIBUTE_VALUES[
        attribute
    ]

    if not candidates:
        return 0.0

    counts: Counter[str] = (
        Counter()
    )

    covered = 0

    # Top 30 is enough to estimate ambiguity
    # without analysing hundreds of candidates.
    sample = candidates[:30]

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

        # Multi-material / multi-feature products
        # may contribute multiple observed values.
        counts.update(
            found
        )

    # No meaningful disagreement.
    if (
        covered < 2
        or len(counts) < 2
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
        len(counts)
    )

    normalized_entropy = (
        entropy
        / max_entropy
        if max_entropy > 0
        else 0.0
    )

    coverage = (
        covered
        / len(sample)
    )

    return (
        coverage
        * normalized_entropy
        * QUESTION_PRIORITY[
            attribute
        ]
    )


def choose_candidate_attribute(
    state: SessionState,
    candidates: list[dict],
    turn: int,
) -> str:
    """
    Choose the next clarification attribute.

    Policy
    ------

    Turns 1–3:
        Broad discovery using `other`.

        This rapidly collects the shopper's
        important constraints.

    Turn 4+:
        If the session is still unresolved,
        inspect the live candidate set and ask
        about the attribute with the highest
        estimated information gain.

    We never ask an attribute that:

        - the shopper already specified
        - was already asked
        - the shopper explicitly declined

    If nothing looks informative enough,
    fall back to `other`.
    """

    # Early dialogue remains broad because it
    # efficiently establishes the shopper's
    # initial constraint set.
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

        # Avoid asking weak / arbitrary questions.
        if information > 0.10:

            scored.append(
                (
                    information,
                    attribute,
                )
            )

    if not scored:
        return "other"

    scored.sort(
        key=lambda item: (
            -item[0],
            item[1],
        )
    )

    return scored[0][1]