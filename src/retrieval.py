from __future__ import annotations

import re
import sqlite3

from src.context import (
    retrieval_active_text,
    retrieval_evidence_text,
)

from src.intent import (
    ShoppingIntent,
)

from src.orchestration import (
    RetrievalPlan,
    retrieval_plan,
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

    V14B changes only the evidence source:

        V13:
            raw legacy evidence

        V14B shadow:
            structured distilled evidence

    Category routing remains unchanged.
    """

    category_terms = terms(
        state.category_text
    )

    evidence_terms = terms(
        retrieval_evidence_text(
            state
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

    if exploration:
        return True

    return (
        state.intent
        == ShoppingIntent.BROWSING

        and

        lexical_count < 50
    )


def _candidate_from_row(
    row: tuple,
    rating_numbers: dict[
        str,
        int,
    ],
) -> dict:

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

        "features":
            row[4]
            or "",

        "details":
            row[5]
            or "",

        "store":
            row[6]
            or "",

        "description":
            row[7]
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
    plan_override: RetrievalPlan | None = None,
) -> list[dict]:

    expression = (
        build_expression(
            state
        )
    )

    plan = (
        plan_override

        if (
            plan_override
            is not None
        )

        else

        retrieval_plan(
            state=state,
            exploration=exploration,
        )
    )

    # ----------------------------------
    # ROUTE 1: FIELD-AWARE BM25
    # ----------------------------------

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
                    plan.lexical_limit,
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
    # ROUTE 2 DECISION
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
    # ROUTE 2: SEMANTIC RETRIEVAL
    # ----------------------------------

    semantic_hits = (
        semantic.search(
            retrieval_active_text(
                state
            ),
            top_n=(
                plan.semantic_limit
            ),
        )
    )

    if not semantic_hits:
        return lexical

    # ----------------------------------
    # FUSE CANDIDATE ROUTES
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

    for (
        asin,
        candidate,
    ) in by_asin.items():

        if (
            asin
            not in semantic_score_by_asin
        ):
            continue

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
        +
        semantic_only
    )