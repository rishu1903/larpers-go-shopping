from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

from src.dialogue import choose_clarification
from src.query import build_search_query
from src.reranker import rerank_candidates
from src.state import SessionState


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


def _text(value: object) -> str:
    if value is None:
        return ""

    if isinstance(value, dict):
        return " ".join(
            f"{key} {item}"
            for key, item in value.items()
        )

    if isinstance(value, list):
        return " ".join(
            str(item)
            for item in value
        )

    return str(value)


def _terms(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if (
            len(token) > 1
            and token.lower() not in STOPWORDS
        )
    ]


class Agent:
    """
    Stateful shopping-search agent with
    BM25 recall and deterministic reranking.
    """

    def __init__(
        self,
        catalog_path: str | Path = "data/catalog.jsonl",
    ) -> None:

        self.catalog_path = Path(
            catalog_path
        )

        self.connection = sqlite3.connect(
            ":memory:"
        )

        self._sessions: dict[
            str,
            SessionState,
        ] = {}

        # Keep popularity metadata outside the FTS index.
        #
        # rating_number is NOT used for retrieval.
        # It is only used as a secondary tie-breaker
        # after intent compatibility has been scored.
        self._rating_numbers: dict[
            str,
            int,
        ] = {}

        self._build_index()

    def _build_index(self) -> None:

        cursor = self.connection.cursor()

        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, "
            "title, "
            "categories, "
            "features, "
            "details, "
            "store, "
            "description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )

        batch: list[
            tuple[
                str,
                str,
                str,
                str,
                str,
                str,
                str,
            ]
        ] = []

        with self.catalog_path.open(
            encoding="utf-8"
        ) as handle:

            for line in handle:

                product = json.loads(
                    line
                )

                parent_asin = str(
                    product["parent_asin"]
                )

                self._rating_numbers[
                    parent_asin
                ] = int(
                    product.get(
                        "rating_number"
                    )
                    or 0
                )

                batch.append(
                    (
                        parent_asin,
                        _text(
                            product.get(
                                "title"
                            )
                        ),
                        _text(
                            product.get(
                                "categories"
                            )
                        ),
                        _text(
                            product.get(
                                "features"
                            )
                        ),
                        _text(
                            product.get(
                                "details"
                            )
                        ),
                        _text(
                            product.get(
                                "store"
                            )
                        ),
                        _text(
                            product.get(
                                "description"
                            )
                        ),
                    )
                )

                if len(batch) >= 1000:

                    cursor.executemany(
                        "INSERT INTO products "
                        "VALUES (?, ?, ?, ?, ?, ?, ?)",
                        batch,
                    )

                    batch.clear()

        if batch:

            cursor.executemany(
                "INSERT INTO products "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                batch,
            )

        self.connection.commit()

    def reset(
        self,
        session_id: str,
        user_profile: dict,
    ) -> None:

        self._sessions[
            session_id
        ] = SessionState(
            user_profile=user_profile
        )

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:

        if session_id not in self._sessions:
            raise RuntimeError(
                "reset must be called before respond"
            )

        state = self._sessions[
            session_id
        ]

        state.update(
            user_message=user_message,
            turn=turn,
        )

        search_query = build_search_query(
            state
        )

        unique_terms = list(
            dict.fromkeys(
                _terms(
                    search_query
                )
            )
        )[:40]

        expression = " OR ".join(
            f'"{term}"'
            for term in unique_terms
        )

        if not expression:

            recommendations: list[
                dict
            ] = []

        else:

            # BM25 is now explicitly the
            # candidate-generation stage.
            #
            # We retrieve 100 candidates,
            # then let the second-stage
            # reranker choose the best 10.
            rows = self.connection.execute(
                "SELECT "
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
                "LIMIT ?",
                (
                    expression,
                    100,
                ),
            ).fetchall()

            candidates = [
                {
                    "parent_asin":
                        str(row[0]),

                    "title":
                        row[1] or "",

                    "categories":
                        row[2] or "",

                    "searchable_text":
                        " ".join(
                            str(
                                value
                                or ""
                            )
                            for value
                            in row[1:]
                        ),

                    "rating_number":
                        self._rating_numbers.get(
                            str(row[0]),
                            0,
                        ),
                }
                for row in rows
            ]

            reranked = rerank_candidates(
                candidates,
                state,
            )

            recommendations = [
                {
                    "parent_asin":
                        item[
                            "parent_asin"
                        ]
                }
                for item
                in reranked[:top_k]
            ]

        ask_attribute, message = (
            choose_clarification(
                state,
                turn,
            )
        )

        return {
            "message":
                message,

            "ask_attribute":
                ask_attribute,

            "recommendations":
                recommendations,

            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
            },
        }