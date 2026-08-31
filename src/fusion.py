from __future__ import annotations

import math
import re

from src.state import SessionState


TOKEN_RE = re.compile(
    r"[a-z0-9]+",
    re.IGNORECASE,
)


# V14 production configuration.
#
# The V12 ablation (fixed 0-1 bonus, weight=1.0)
# plateaued at weight >= 1.0 -- larger weights
# changed nothing, which was the first sign the
# bonus was simply too small to compete with
# relevance scores that can run into double digits,
# not that the weight itself was well-tuned.
#
# V14 scales the bonus by the query's own relevance
# range (see semantic_exploration_bonus). Re-running
# the same label-free 130-case ablation against this
# new formula shape shows the OLD weight of 1.0 is
# now too strong -- it overshoots and actively
# regresses cumulative shadow Hit@10 (0.90 -> 0.80).
# Weight 0.25 is the smallest tested value that
# reaches the sweep's best cumulative Hit@10 and
# rescue count.
#
# Measured outcome, stated plainly: at its best
# setting, the rescaled formula ties the old fixed
# formula's own best setting on every primary metric
# (cumulative Hit@10, rescues, misses) -- it does not
# unlock further improvement on this benchmark. What
# it does provide is a properly calibrated weight
# range (a fixed constant no longer has to be tuned
# against an unknown, query-dependent relevance
# scale), which is the reason it is still adopted
# here rather than reverting to the fixed formula.
SEMANTIC_EXPLORATION_WEIGHT = 0.25


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
    relevance_scale: float = 1.0,
) -> float:
    """
    Return the exploration bonus for a semantic-only
    candidate.

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

    `relevance_scale` (V14) multiplies the bounded
    0-1 rank signal by the current query's
    deterministic relevance range, so the bonus is
    meaningful relative to *this* candidate pool
    instead of being a fixed constant that a strong
    lexical relevance score (which can run into the
    5-15+ range) always dwarfs. Defaults to 1.0,
    which reproduces the exact V12 formula.
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
        * relevance_scale
    )