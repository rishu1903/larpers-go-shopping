from __future__ import annotations

import re
import sqlite3

from src.semantic import (
    SemanticRetriever,
)

from src.state import (
    SessionState,
)


TOKEN_RE = re.compile(
    r"[a-z0-9]+",
    re.IGNORECASE,
)


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "i",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "please",
    "some",
    "that",
    "the",
    "this",
    "to",
    "want",
    "with",
    "would",
    "you",
    "looking",
}


def terms(
    text: str,
) -> list[str]:
    """
    Produce unique lexical query terms
    while preserving original order.
    """

    return list(
        dict.fromkeys(
            token.lower()
            for token
            in TOKEN_RE.findall(
                text
            )
            if (
                len(token) > 1
                and token.lower()
                not in STOPWORDS
            )
        )
    )


def _ors(
    tokens: list[str],
    max_terms: int = 40,
) -> str:
    """
    Build an FTS5 OR expression.
    """

    return " OR ".join(
        f'"{token}"'
        for token
        in tokens[
            :max_terms
        ]
    )


def build_expression(
    state: SessionState,
) -> str:
    """
    Build a field-aware BM25 query.

    V4 searched all customer terms across
    every FTS column.

    V5 separates:

        CATEGORY TERMS
            ↓
        title + categories

    from:

        PREFERENCE / CONSTRAINT TERMS
            ↓
        title + features + details
        + description + store

    This prevents category words from being
    rewarded merely because they appear in a
    long description and gives structured
    evidence more appropriate search fields.
    """

    category_terms = terms(
        state.category_text
    )

    evidence_terms = terms(
        " ".join(
            item.text
            for item
            in state.evidence
        )
    )

    clauses: list[str] = []

    if category_terms:

        clauses.append(
            (
                "{title categories} : "
                f"({_ors(category_terms, 16)})"
            )
        )

    if evidence_terms:

        clauses.append(
            (
                "{title features details "
                "description store} : "
                f"({_ors(evidence_terms, 32)})"
            )
        )

    return " OR ".join(
        f"({clause})"
        for clause
        in clauses
    )


def _candidate_from_row(
    row: tuple,
    rating_numbers: dict[
        str,
        int,
    ],
) -> dict:
    """
    Convert an FTS row into the candidate
    structure expected by our reranker.
    """

    parent_asin = str(
        row[1]
    )

    return {
        "rowid":
            int(
                row[0]
            ),

        "parent_asin":
            parent_asin,

        "title":
            row[2]
            or "",

        "categories":
            row[3]
            or "",

        "searchable_text":
            " ".join(
                str(
                    value
                    or ""
                )
                for value
                in row[2:]
            ),

        "rating_number":
            rating_numbers.get(
                parent_asin,
                0,
            ),

        # Reserved now because later versions
        # may include this score directly in
        # fusion/reranking.
        "semantic_score":
            0.0,

        "source":
            "lexical",
    }


def retrieve_candidates(
    connection: sqlite3.Connection,
    state: SessionState,
    rating_numbers: dict[
        str,
        int,
    ],
    asin_to_rowid: dict[
        str,
        int,
    ],
    semantic: SemanticRetriever | None,
    exploration: bool = False,
) -> list[dict]:
    """
    Retrieve the candidate pool.

    NORMAL MODE
    -----------

        field-aware BM25
        Top 100

    EXPLORATION MODE
    ----------------

        field-aware BM25
        Top 500

            +

        dense semantic search
        Top 250

            ↓

        candidate union


    Why semantic search is conditional:

    Our public ablation showed that injecting
    semantic candidates on every turn reduced
    MRR despite preserving HR@10.

    Therefore the dense route activates only
    when structured clarification is exhausted.
    """

    expression = build_expression(
        state
    )

    lexical_limit = (
        500
        if exploration
        else 100
    )

    lexical: list[dict] = []

    # --------------------------------------------------
    # ROUTE 1:
    # FIELD-AWARE BM25
    # --------------------------------------------------

    if expression:

        rows = (
            connection.execute(
                (
                    "SELECT "
                    "rowid, "
                    "parent_asin, "
                    "title, "
                    "categories, "
                    "features, "
                    "details, "
                    "store, "
                    "description "
                    "FROM products "
                    "WHERE products MATCH ? "
                    "ORDER BY bm25("
                    "products, "
                    "0.0, "
                    "6.0, "
                    "4.0, "
                    "2.5, "
                    "2.5, "
                    "1.5, "
                    "1.0"
                    ") "
                    "LIMIT ?"
                ),
                (
                    expression,
                    lexical_limit,
                ),
            )
            .fetchall()
        )

        lexical = [
            _candidate_from_row(
                row,
                rating_numbers,
            )
            for row
            in rows
        ]

    # --------------------------------------------------
    # HIGH-PRECISION MODE
    # --------------------------------------------------
    #
    # Do not inject dense candidates while
    # clarification is still productive.
    #
    # This deliberately preserves our proven
    # lexical precision.

    if (
        semantic is None
        or not exploration
    ):
        return lexical

    # --------------------------------------------------
    # ROUTE 2:
    # DENSE LATENT-SEMANTIC RETRIEVAL
    # --------------------------------------------------

    semantic_limit = 250

    semantic_hits = (
        semantic.search(
            state.active_text(),
            top_n=semantic_limit,
        )
    )

    if not semantic_hits:
        return lexical

    # Existing lexical candidates by ASIN.
    by_asin = {
        candidate[
            "parent_asin"
        ]:
        candidate

        for candidate
        in lexical
    }

    score_by_asin = dict(
        semantic_hits
    )

    # Find products discovered only by
    # the semantic retrieval route.
    missing_asins = [
        asin
        for asin, _
        in semantic_hits

        if (
            asin
            not in by_asin

            and

            asin
            in asin_to_rowid
        )
    ]

    missing_rowids = [
        asin_to_rowid[
            asin
        ]
        for asin
        in missing_asins
    ]

    # Fetch metadata for semantic-only
    # products from the existing FTS table.
    #
    # We map ASIN -> rowid so this still works
    # correctly even if the runtime catalogue
    # ordering differs from the semantic asset
    # ordering.
    if missing_rowids:

        placeholders = ",".join(
            "?"
            for _
            in missing_rowids
        )

        rows = (
            connection.execute(
                (
                    "SELECT "
                    "rowid, "
                    "parent_asin, "
                    "title, "
                    "categories, "
                    "features, "
                    "details, "
                    "store, "
                    "description "
                    "FROM products "
                    f"WHERE rowid IN "
                    f"({placeholders})"
                ),
                missing_rowids,
            )
            .fetchall()
        )

        for row in rows:

            candidate = (
                _candidate_from_row(
                    row,
                    rating_numbers,
                )
            )

            candidate[
                "source"
            ] = "semantic"

            by_asin[
                candidate[
                    "parent_asin"
                ]
            ] = candidate

    # Mark products that appeared in both routes
    # and retain their semantic similarity score.
    for (
        asin,
        candidate,
    ) in by_asin.items():

        if asin in score_by_asin:

            candidate[
                "semantic_score"
            ] = (
                score_by_asin[
                    asin
                ]
            )

            if (
                candidate[
                    "source"
                ]
                == "lexical"
            ):
                candidate[
                    "source"
                ] = "hybrid"

    lexical_asins = {
        candidate[
            "parent_asin"
        ]
        for candidate
        in lexical
    }

    # Preserve lexical ordering first.
    #
    # Semantic retrieval EXPANDS recall;
    # it does not arbitrarily replace our
    # proven lexical route.
    semantic_only = [
        by_asin[
            asin
        ]

        for asin, _
        in semantic_hits

        if (
            asin
            not in lexical_asins

            and

            asin
            in by_asin
        )
    ]

    return (
        lexical
        + semantic_only
    )