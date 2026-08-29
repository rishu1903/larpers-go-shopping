from __future__ import annotations

import re
import sqlite3

from src.intent import (
    ShoppingIntent,
)

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
    Produce unique lexical terms while
    preserving their original order.
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

                and

                token.lower()
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

    Category evidence is searched primarily
    against:

        title
        categories

    Product constraints are searched against:

        title
        features
        details
        description
        store
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

    clauses: list[
        str
    ] = []

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


def should_use_semantic(
    state: SessionState,
    lexical_count: int,
    exploration: bool,
) -> bool:
    """
    Decide whether the dense retrieval route
    should be activated.

    BUYING
    -------

    Prefer the high-precision lexical path.

    BROWSING
    --------

    Activate semantic recall when lexical
    retrieval produces too small a candidate
    pool.

    EXPLORATION
    -----------

    Always enable semantic recall because
    clarification has already been exhausted.

    This gives us explicit route-aware retrieval
    without blindly injecting dense candidates
    into every query.
    """

    if exploration:
        return True

    if (
        state.intent
        == ShoppingIntent.BROWSING

        and

        lexical_count < 50
    ):

        return True

    return False


def _candidate_from_row(
    row: tuple,
    rating_numbers: dict[
        str,
        int,
    ],
) -> dict:
    """
    Convert one FTS row to the structure
    expected by the reranking layer.
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
    Execute the route-aware candidate
    retrieval policy.

    BUYING
    ======

        BM25 Top 100

    BROWSING
    ========

        BM25 Top 100

        If lexical recall is sparse:

            +
        Semantic Top 100

    EXPLORATION
    ===========

        BM25 Top 500
            +
        Semantic Top 250
    """

    expression = (
        build_expression(
            state
        )
    )

    # ----------------------------------
    # ROUTE 1:
    # FIELD-AWARE BM25
    # ----------------------------------

    lexical_limit = (
        500
        if exploration
        else 100
    )

    lexical: list[
        dict
    ] = []

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

    # ----------------------------------
    # INTENT-ROUTED SEMANTIC DECISION
    # ----------------------------------

    use_semantic = (
        should_use_semantic(
            state=state,
            lexical_count=len(
                lexical
            ),
            exploration=exploration,
        )
    )

    if (
        semantic is None

        or

        not use_semantic
    ):

        return lexical

    # ----------------------------------
    # ROUTE 2:
    # DENSE SEMANTIC RETRIEVAL
    # ----------------------------------

    semantic_limit = (
        250
        if exploration
        else 100
    )

    semantic_hits = (
        semantic.search(
            state.active_text(),
            top_n=semantic_limit,
        )
    )

    if not semantic_hits:

        return lexical

    # ----------------------------------
    # FUSE THE TWO CANDIDATE ROUTES
    # ----------------------------------

    by_asin = {
        candidate[
            "parent_asin"
        ]:
        candidate

        for candidate
        in lexical
    }

    semantic_score_by_asin = dict(
        semantic_hits
    )

    missing_asins = [
        asin

        for (
            asin,
            _,
        ) in semantic_hits

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

    # ----------------------------------
    # LOAD SEMANTIC-ONLY PRODUCT METADATA
    # ----------------------------------

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

    # ----------------------------------
    # MARK HYBRID CANDIDATES
    # ----------------------------------

    for (
        asin,
        candidate,
    ) in by_asin.items():

        if (
            asin
            in semantic_score_by_asin
        ):

            candidate[
                "semantic_score"
            ] = (
                semantic_score_by_asin[
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

    # Preserve the proven lexical ordering
    # first, while allowing semantic retrieval
    # to expand the candidate universe.
    semantic_only = [
        by_asin[
            asin
        ]

        for (
            asin,
            _,
        ) in semantic_hits

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