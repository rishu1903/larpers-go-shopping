from __future__ import annotations

import re
from collections.abc import Iterable

from src.fusion import (
    semantic_exploration_bonus,
)

from src.state import SessionState


TOKEN_RE = re.compile(
    r"[a-z0-9]+",
    re.IGNORECASE,
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

    Ranking priority:

    1. Intent relevance
    2. Popularity
    3. Original BM25 order

    V12 deliberately does NOT modify this path.

    This remains our high-precision ranking
    strategy while clarification is still
    providing useful information.
    """

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

    scored.sort(
        key=lambda item: (
            -item[0],
            -item[1],
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