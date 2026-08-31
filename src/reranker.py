from __future__ import annotations

import re
from collections.abc import Iterable

from src.fusion import (
    normalized_rrf,
    semantic_exploration_bonus,
)

from src.intent import (
    ShoppingIntent,
)

from src.state import SessionState


TOKEN_RE = re.compile(
    r"[a-z0-9]+",
    re.IGNORECASE,
)


# --------------------------------------------------
# BUYING/OVERRIDE RELEVANCE-LABEL STRIPPING
# --------------------------------------------------
#
# The evaluator's scripted constraint text is
# frequently attribute-label-prefixed, e.g.
# "color: red" or "size: large" (see
# evaluator/local_evaluator.py::intent_card). Real
# product metadata usually states the bare value
# without that literal label word adjacent to it in
# the same word order, so the label token can silently
# drag an otherwise-exact evidence match down to
# partial token-overlap credit. Stripping a small,
# generic set of attribute-name prefixes recovers the
# exact-match bonus for these label-prefixed chunks.
#
# This is scoped to buying-intent contexts only (see
# configure_buying_relevance_labels) so it never
# touches Browsing/exploration-mode ranking.

_ATTRIBUTE_LABEL_PREFIX_RE = re.compile(
    r"^(?:category|material|color|size|style|"
    r"use_case|feature|budget)\s*:\s*",
    re.IGNORECASE,
)


def _strip_attribute_label(
    chunk: str,
) -> str:
    """
    Remove a leading "attribute:" label from an
    evidence chunk, leaving the bare value.
    """

    return _ATTRIBUTE_LABEL_PREFIX_RE.sub(
        "",
        chunk,
    )


# Ablation modes:
#
#   "off"           -- never strip (pre-experiment
#                       behaviour).
#   "buying_only"   -- strip only for buying-intent
#                       turns that never saw an
#                       override.
#   "override_only" -- strip only for turns after an
#                       intent override.
#   "both"          -- strip for every buying-intent
#                       turn, override or not
#                       (production setting).
_BUYING_RELEVANCE_LABEL_MODE = "both"


def configure_buying_relevance_labels(
    mode: str,
) -> None:
    """
    Configure which population receives the
    buying/override relevance-label stripping.

    Exists primarily for controlled offline
    ablations -- see the module docstring above.
    """

    if mode not in (
        "off",
        "buying_only",
        "override_only",
        "both",
    ):

        raise ValueError(
            "mode must be one of: "
            "off, buying_only, override_only, both"
        )

    global _BUYING_RELEVANCE_LABEL_MODE

    _BUYING_RELEVANCE_LABEL_MODE = mode


def _should_strip_attribute_labels(
    state: SessionState,
) -> bool:

    if _BUYING_RELEVANCE_LABEL_MODE == "off":
        return False

    is_buying = (
        state.intent
        == ShoppingIntent.BUYING
    )

    if _BUYING_RELEVANCE_LABEL_MODE == "both":
        return is_buying

    if _BUYING_RELEVANCE_LABEL_MODE == "buying_only":
        return (
            is_buying
            and not state.override_seen
        )

    # "override_only"
    return state.override_seen


# --------------------------------------------------
# BLENDED ZERO-EVIDENCE FALLBACK
# --------------------------------------------------
#
# V15.1 replaced popularity-first with a hard switch
# to BM25-first when state.evidence is empty. This
# generalizes that binary switch into a tunable
# rank-based blend of the two signals, reusing the
# bounded reciprocal-rank fusion already defined in
# src/fusion.py for the semantic exploration bonus.
#
# Defaults reproduce the exact V15.1 ordering bit-for
# -bit (BM25 rank fully weighted, popularity weight
# zero), so this is a safe, no-op starting point until
# an ablation demonstrates a better setting.

FALLBACK_BM25_WEIGHT = 1.0

FALLBACK_POPULARITY_WEIGHT = 0.0


def configure_fallback_blend_weights(
    bm25_weight: float,
    popularity_weight: float,
) -> None:
    """
    Configure the zero-evidence fallback blend.

    Exists primarily for controlled offline
    ablations -- see the module docstring above.
    """

    for weight in (
        bm25_weight,
        popularity_weight,
    ):

        if (
            not isinstance(
                weight,
                (int, float),
            )
            or weight != weight  # NaN check
            or weight < 0.0
        ):

            raise ValueError(
                "fallback blend weights must be "
                "finite non-negative numbers"
            )

    global FALLBACK_BM25_WEIGHT
    global FALLBACK_POPULARITY_WEIGHT

    FALLBACK_BM25_WEIGHT = float(
        bm25_weight
    )

    FALLBACK_POPULARITY_WEIGHT = float(
        popularity_weight
    )


def _fallback_score(
    bm25_rank: int,
    popularity_rank: int,
) -> float:
    """
    Blend BM25 retrieval order and popularity rank
    into one tie-break score for the zero-evidence
    fallback, using bounded reciprocal-rank fusion
    for both signals so neither raw BM25 scores nor
    raw rating counts need to be independently
    normalized.
    """

    return (
        FALLBACK_BM25_WEIGHT
        * normalized_rrf(
            bm25_rank
        )
    ) + (
        FALLBACK_POPULARITY_WEIGHT
        * normalized_rrf(
            popularity_rank
        )
    )


def _normalize(
    text: object,
) -> str:
    return " ".join(
        TOKEN_RE.findall(
            str(text).lower()
        )
    )


def _tokens(
    text: object,
) -> set[str]:
    return set(
        TOKEN_RE.findall(
            str(text).lower()
        )
    )


def _evidence_chunks(
    state: SessionState,
) -> list[str]:
    """
    Split accumulated evidence into
    independently matchable constraints.
    """

    chunks: list[str] = []

    for item in state.evidence:
        chunks.extend(
            part.strip()

            for part
            in item.text.split(";")

            if part.strip()
        )

    return chunks


def _candidate_relevance(
    candidate: dict,
    state: SessionState,
    *,
    strip_attribute_labels: bool = False,
) -> float:
    """
    Measure how completely one candidate
    matches the shopper's active intent.

    This relevance score is shared by both
    exploitation and exploration modes.
    """

    category_tokens = _tokens(
        state.category_text
    )

    category_normalized = _normalize(
        state.category_text
    )

    searchable = _normalize(
        candidate.get(
            "searchable_text",
            "",
        )
    )

    product_tokens = _tokens(
        searchable
    )

    category_text = _normalize(
        candidate.get(
            "categories",
            "",
        )
    )

    title_text = _normalize(
        candidate.get(
            "title",
            "",
        )
    )

    category_or_title_tokens = _tokens(
        f"{category_text} {title_text}"
    )

    score = 0.0

    # ----------------------------------
    # CATEGORY COMPATIBILITY
    # ----------------------------------

    if category_tokens:
        category_coverage = (
            len(
                category_tokens
                & category_or_title_tokens
            )
            / len(category_tokens)
        )

        score += (
            2.0
            * category_coverage
        )

        if (
            category_normalized
            and (
                category_normalized
                in category_text

                or

                category_normalized
                in title_text
            )
        ):
            score += 1.0

    # ----------------------------------
    # CUSTOMER CONSTRAINT COVERAGE
    # ----------------------------------

    for chunk in _evidence_chunks(
        state
    ):
        if strip_attribute_labels:

            chunk = _strip_attribute_label(
                chunk
            )

        chunk_normalized = _normalize(
            chunk
        )

        chunk_tokens = _tokens(
            chunk
        )

        if not chunk_tokens:
            continue

        coverage = (
            len(
                chunk_tokens
                & product_tokens
            )
            / len(chunk_tokens)
        )

        score += (
            2.0
            * coverage
        )

        # Exact product-metadata phrase matches
        # remain our strongest deterministic
        # signal.
        if (
            chunk_normalized
            and
            chunk_normalized
            in searchable
        ):
            score += 3.0

            score += (
                0.20
                * min(
                    len(chunk_tokens),
                    12,
                )
            )

        elif coverage >= 0.80:
            score += 1.0

    return score


def rerank_candidates(
    candidates: Iterable[dict],
    state: SessionState,
) -> list[dict]:
    """
    EXPLOITATION MODE.

    Ranking priority (once evidence exists):

    1. Intent relevance
    2. Popularity
    3. Original BM25 order

    V12 deliberately does NOT modify this path.

    This remains our high-precision ranking
    strategy while clarification is still
    providing useful information.

    V15 exception, scoped narrowly: when NO
    evidence has been accumulated yet
    (`state.evidence` is empty -- turn 1 of a
    browsing session, before any constraint has
    been disclosed), every same-category candidate
    ties on relevance. In that specific case,
    original BM25 order is preferred ahead of raw
    popularity, since BM25's full-text rank
    already reflects real textual relevance across
    every indexed field, while popularity alone
    can bury a genuinely relevant long-tail
    product. This is a no-op the moment any
    evidence exists, which is true for every
    buying-session turn and every browsing-session
    turn from turn 2 onward.
    """

    has_evidence = bool(
        state.evidence
    )

    strip_attribute_labels = (
        _should_strip_attribute_labels(
            state
        )
    )

    scored: list[
        tuple[
            float,
            int,
            int,
            dict,
        ]
    ] = []

    for (
        bm25_index,
        candidate,
    ) in enumerate(
        candidates
    ):
        relevance = _candidate_relevance(
            candidate,
            state,
            strip_attribute_labels=(
                strip_attribute_labels
            ),
        )

        rating_number = int(
            candidate.get(
                "rating_number",
                0,
            )
            or 0
        )

        scored.append(
            (
                relevance,
                rating_number,
                bm25_index,
                candidate,
            )
        )

    if has_evidence:

        scored.sort(
            key=lambda item: (
                -item[0],
                -item[1],
                item[2],
            )
        )

    else:

        popularity_rank_by_bm25_index = {
            bm25_index: rank

            for (
                rank,
                (
                    _,
                    _,
                    bm25_index,
                    _,
                ),
            )
            in enumerate(
                sorted(
                    scored,
                    key=lambda item: (
                        -item[1],
                        item[2],
                    ),
                ),
                start=1,
            )
        }

        scored.sort(
            key=lambda item: (
                -item[0],
                -_fallback_score(
                    bm25_rank=(
                        item[2]
                        + 1
                    ),
                    popularity_rank=(
                        popularity_rank_by_bm25_index[
                            item[2]
                        ]
                    ),
                ),
                item[2],
            )
        )

    return [
        candidate

        for (
            _,
            _,
            _,
            candidate,
        )
        in scored
    ]


def rerank_for_exploration(
    candidates: Iterable[dict],
    state: SessionState,
) -> list[dict]:
    """
    EXPLORATION MODE.

    Exploration activates only after the shopper
    explicitly indicates that there are no more
    preferences to provide.

    V12 retains the existing deterministic
    relevance calculation but allows a bounded
    semantic-route signal to help SEMANTIC-ONLY
    candidates that are:

        - highly ranked by dense retrieval; and
        - still compatible with the active category.

    Normal lexical and hybrid candidates receive
    no semantic bonus.

    If the V12 semantic weight is zero, this
    produces the same ordering policy as V11.

    Existing long-tail behaviour remains:

    1. relevance / fusion score
    2. lower-review products
    3. deeper retrieval products
    """

    candidate_list = list(
        candidates
    )

    # ----------------------------------
    # RECONSTRUCT SEMANTIC ROUTE RANK
    # ----------------------------------
    #
    # Retrieval already attaches semantic_score
    # to every semantic/hybrid candidate.
    #
    # Sorting those scores recreates the dense
    # route rank without changing retrieval.py
    # or the public Agent contract.
    #
    # Hybrid candidates are included here even
    # though they do not receive a bonus. This
    # means a semantic-only candidate retains its
    # true rank within the original dense route.

    semantic_ranked = sorted(
        (
            candidate

            for candidate
            in candidate_list

            if float(
                candidate.get(
                    "semantic_score",
                    0.0,
                )
                or 0.0
            )
            > 0.0
        ),
        key=lambda candidate: (
            -float(
                candidate.get(
                    "semantic_score",
                    0.0,
                )
                or 0.0
            ),

            str(
                candidate.get(
                    "parent_asin",
                    "",
                )
            ),
        ),
    )

    semantic_rank_by_asin = {
        str(
            candidate.get(
                "parent_asin",
                "",
            )
        ): rank

        for rank, candidate
        in enumerate(
            semantic_ranked,
            start=1,
        )
    }

    # ----------------------------------
    # SCORE EXPLORATION CANDIDATES
    # ----------------------------------

    relevances = [
        _candidate_relevance(
            candidate,
            state,
        )

        for candidate
        in candidate_list
    ]

    # V14: scale the semantic bonus relative to how
    # large relevance scores actually run for this
    # query, instead of a fixed 0-1 constant that a
    # strong deterministic match always dwarfs. Falls
    # back to 1.0 (the V12 bounded formula) when the
    # pool carries no lexical signal at all.
    relevance_scale = (
        max(relevances)
        if relevances
        else 0.0
    )

    if relevance_scale <= 0.0:
        relevance_scale = 1.0

    scored: list[
        tuple[
            float,
            float,
            int,
            int,
            float,
            dict,
        ]
    ] = []

    for (
        retrieval_index,
        candidate,
    ) in enumerate(
        candidate_list
    ):
        relevance = relevances[
            retrieval_index
        ]

        semantic_rank = (
            semantic_rank_by_asin.get(
                str(
                    candidate.get(
                        "parent_asin",
                        "",
                    )
                )
            )
        )

        semantic_bonus = (
            semantic_exploration_bonus(
                candidate=candidate,
                state=state,
                semantic_rank=(
                    semantic_rank
                ),
                relevance_scale=(
                    relevance_scale
                ),
            )
        )

        exploration_score = (
            relevance
            + semantic_bonus
        )

        rating_number = int(
            candidate.get(
                "rating_number",
                0,
            )
            or 0
        )

        average_rating = float(
            candidate.get(
                "average_rating"
            )
            or 0.0
        )

        scored.append(
            (
                exploration_score,
                relevance,
                rating_number,
                retrieval_index,
                average_rating,
                candidate,
            )
        )

    scored.sort(
        key=lambda item: (
            # V12 bounded semantic fusion.
            -item[0],

            # If fusion scores tie, prefer the
            # stronger deterministic relevance.
            -item[1],

            # Preserve V4 long-tail exploration.
            item[2],

            # Prefer deeper retrieval products.
            -item[3],

            # V14 final tie-break: among candidates
            # tied on everything above, prefer the
            # better-rated product. Missing ratings
            # sort last, never above a rated product.
            -item[4],
        )
    )

    return [
        candidate

        for (
            _,
            _,
            _,
            _,
            _,
            candidate,
        )
        in scored
    ]