from __future__ import annotations

import re
from collections.abc import Iterable

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

    This mode activates only after the shopper
    explicitly indicates that there are no more
    preferences to provide.

    Relevance remains the primary objective.

    Within an equal-relevance tier we deliberately
    increase long-tail exposure by preferring:

    1. Lower-review products
    2. Products deeper in the original BM25 pool

    This counters popularity collapse and prevents
    the agent from endlessly repeating the same
    mainstream items when the remaining intent is
    genuinely ambiguous.
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
            # Relevance ALWAYS stays first.
            -item[0],

            # Within equal relevance,
            # explore less-popular products.
            item[1],

            # Within another tie, surface
            # candidates BM25 previously
            # placed deeper in the pool.
            -item[2],
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