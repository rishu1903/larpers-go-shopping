from __future__ import annotations

import math
import re

from src.state import SessionState


TOKEN_RE = re.compile(
    r"[a-z0-9]+",
    re.IGNORECASE,
)


# V12 production configuration.
#
# Selected through the label-free 130-case
# semantic exploration fusion ablation.
#
# Weight 1.0 was the smallest tested value that:
#
# - improved cumulative shadow Hit@10;
# - increased second-turn rescues;
# - increased semantic-only rescues; and
# - preserved the protected turn-1 ranking path.
#
# Larger weights produced no additional coverage
# while degrading secondary ranking metrics.
SEMANTIC_EXPLORATION_WEIGHT = 1.0


SEMANTIC_RRF_K = 60.0

MIN_CATEGORY_COVERAGE = 0.50


def configure_semantic_exploration_weight(
    weight: float,
) -> None:
    """
    Configure the semantic-only exploration bonus.

    This function exists primarily for controlled
    offline ablations.

    Production uses
    SEMANTIC_EXPLORATION_WEIGHT = 1.0.
    """

    if (
        not math.isfinite(weight)
        or weight < 0.0
    ):
        raise ValueError(
            "semantic exploration weight must "
            "be a finite non-negative number"
        )

    global SEMANTIC_EXPLORATION_WEIGHT

    SEMANTIC_EXPLORATION_WEIGHT = float(
        weight
    )


def _tokens(
    text: object,
) -> set[str]:
    return set(
        TOKEN_RE.findall(
            str(text).lower()
        )
    )


def category_coverage(
    candidate: dict,
    state: SessionState,
) -> float:
    """
    Measure whether a semantic-only candidate
    remains compatible with the shopper's
    active product category.

    Dense retrieval is useful for recovering
    paraphrased requirements, but it must not
    introduce unrelated product-domain drift.
    """

    wanted = _tokens(
        state.category_text
    )

    if not wanted:
        return 0.0

    available = _tokens(
        (
            f"{candidate.get('title', '')} "
            f"{candidate.get('categories', '')}"
        )
    )

    return (
        len(
            wanted
            & available
        )
        / len(wanted)
    )


def normalized_rrf(
    rank: int | None,
    k: float = SEMANTIC_RRF_K,
) -> float:
    """
    Convert semantic retrieval rank into a
    bounded reciprocal-rank signal.

    Rank 1 receives 1.0.

    Using retrieval rank rather than raw cosine
    similarity avoids assuming that LSA similarity
    scores are calibrated across different queries.
    """

    if (
        rank is None
        or rank <= 0
    ):
        return 0.0

    return (
        (k + 1.0)
        / (k + float(rank))
    )


def semantic_exploration_bonus(
    candidate: dict,
    state: SessionState,
    semantic_rank: int | None,
) -> float:
    """
    Return the V12 exploration bonus.

    The bonus is intentionally conservative.

    It applies only when:

    1. the candidate was found exclusively by
       semantic retrieval;

    2. the candidate remains sufficiently
       compatible with the active category; and

    3. semantic retrieval provided a valid rank.

    Hybrid candidates already benefit from
    lexical retrieval and therefore receive no
    additional semantic promotion.

    Normal non-exploration ranking never calls
    this function.
    """

    if (
        SEMANTIC_EXPLORATION_WEIGHT
        <= 0.0
    ):
        return 0.0

    if (
        candidate.get(
            "source"
        )
        != "semantic"
    ):
        return 0.0

    if (
        category_coverage(
            candidate,
            state,
        )
        < MIN_CATEGORY_COVERAGE
    ):
        return 0.0

    return (
        SEMANTIC_EXPLORATION_WEIGHT
        * normalized_rrf(
            semantic_rank
        )
    )