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


def rerank_candidates(
    candidates: Iterable[dict],
    state: SessionState,
) -> list[dict]:
    """
    Rerank BM25 candidates using:

    1. Active category compatibility
    2. Customer constraint coverage
    3. Exact constraint phrase matches
    4. Product popularity as a relevance tie-breaker
    5. Original BM25 order as the final tie-breaker

    BM25 is responsible for candidate generation.
    It is deliberately not added to the second-stage
    relevance score again.
    """

    category_tokens = _tokens(
        state.category_text
    )

    category_normalized = _normalize(
        state.category_text
    )

    chunks = _evidence_chunks(
        state
    )

    chunk_features = [
        (
            _normalize(
                chunk
            ),
            _tokens(
                chunk
            ),
        )
        for chunk in chunks
    ]

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

        category_or_title_tokens = (
            _tokens(
                f"{category_text} "
                f"{title_text}"
            )
        )

        # V2 started with:
        #
        # score = 0.20 / (bm25_index + 1)
        #
        # V3 removes this.
        #
        # BM25 already selected the Top 100.
        # Reusing its position here was
        # effectively double-counting the
        # lexical retriever.
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
                / len(
                    category_tokens
                )
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

        for (
            chunk_normalized,
            chunk_tokens,
        ) in chunk_features:

            if not chunk_tokens:
                continue

            coverage = (
                len(
                    chunk_tokens
                    & product_tokens
                )
                / len(
                    chunk_tokens
                )
            )

            score += (
                2.0
                * coverage
            )

            # Exact structured-product phrases
            # remain our strongest deterministic
            # relevance signal.
            if (
                chunk_normalized
                and chunk_normalized
                in searchable
            ):

                score += 3.0

                score += (
                    0.20
                    * min(
                        len(
                            chunk_tokens
                        ),
                        12,
                    )
                )

            elif coverage >= 0.80:

                score += 1.0

        rating_number = int(
            candidate.get(
                "rating_number",
                0,
            )
            or 0
        )

        scored.append(
            (
                score,
                rating_number,
                bm25_index,
                candidate,
            )
        )

    # Ranking hierarchy:
    #
    # 1. Highest intent relevance
    # 2. Highest rating count
    # 3. Original BM25 position
    #
    # This is intentionally lexicographic.
    # Popularity cannot compensate for a
    # lower relevance score.
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