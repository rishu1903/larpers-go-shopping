from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from src.dialogue import (
    choose_clarification,
)

from src.reranker import (
    rerank_candidates,
    rerank_for_exploration,
)

from src.retrieval import (
    retrieve_candidates,
)

from src.semantic import (
    SemanticRetriever,
)

from src.state import (
    SessionState,
)


def _text(
    value: object,
) -> str:
    """
    Flatten catalogue metadata for FTS5.
    """

    if value is None:
        return ""

    if isinstance(
        value,
        dict,
    ):
        return " ".join(
            f"{key} {item}"
            for key, item
            in value.items()
        )

    if isinstance(
        value,
        list,
    ):
        return " ".join(
            str(item)
            for item
            in value
        )

    return str(value)


class Agent:
    """
    Adaptive conversational shopping agent.

    Architecture:

        Session State
            ↓
        Field-aware BM25
            ↓
        Relevance Reranking
            ↓
        Clarification
            ↓
        Adaptive Exploration
            ↓
        Lexical + Dense Semantic Recall
    """

    def __init__(
        self,
        catalog_path: str | Path = (
            "data/catalog.jsonl"
        ),
    ) -> None:

        self.catalog_path = Path(
            catalog_path
        )

        self.connection = (
            sqlite3.connect(
                ":memory:"
            )
        )

        self._sessions: dict[
            str,
            SessionState,
        ] = {}

        self._rating_numbers: dict[
            str,
            int,
        ] = {}

        # The semantic asset maps embeddings
        # to ASINs, not runtime SQLite rowids.
        #
        # This dictionary lets us efficiently
        # convert semantic hits back into the
        # current FTS table.
        self._asin_to_rowid: dict[
            str,
            int,
        ] = {}

        self._build_index()

        # Load the prebuilt 96-dimensional
        # semantic representation.
        self.semantic = (
            SemanticRetriever()
        )

    def _build_index(
        self,
    ) -> None:

        cursor = (
            self.connection
            .cursor()
        )

        cursor.execute(
            (
                "CREATE VIRTUAL TABLE "
                "products USING fts5("
                "parent_asin UNINDEXED, "
                "title, "
                "categories, "
                "features, "
                "details, "
                "store, "
                "description, "
                "tokenize='unicode61 "
                "remove_diacritics 2'"
                ")"
            )
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

            for (
                rowid,
                line,
            ) in enumerate(
                handle,
                start=1,
            ):

                product = json.loads(
                    line
                )

                parent_asin = str(
                    product[
                        "parent_asin"
                    ]
                )

                self._asin_to_rowid[
                    parent_asin
                ] = rowid

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

                if (
                    len(batch)
                    >= 1000
                ):

                    cursor.executemany(
                        (
                            "INSERT INTO "
                            "products VALUES "
                            "(?, ?, ?, ?, ?, ?, ?)"
                        ),
                        batch,
                    )

                    batch.clear()

        if batch:

            cursor.executemany(
                (
                    "INSERT INTO "
                    "products VALUES "
                    "(?, ?, ?, ?, ?, ?, ?)"
                ),
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

        if (
            session_id
            not in self._sessions
        ):
            raise RuntimeError(
                "reset must be called "
                "before respond"
            )

        state = self._sessions[
            session_id
        ]

        # --------------------------------------------------
        # 1. UPDATE DIALOGUE STATE
        # --------------------------------------------------

        state.update(
            user_message=user_message,
            turn=turn,
        )

        # --------------------------------------------------
        # 2. HYBRID CANDIDATE RETRIEVAL
        # --------------------------------------------------

        candidates = (
            retrieve_candidates(
                connection=(
                    self.connection
                ),
                state=state,
                rating_numbers=(
                    self._rating_numbers
                ),
                asin_to_rowid=(
                    self._asin_to_rowid
                ),
                semantic=(
                    self.semantic
                ),
                exploration=(
                    state
                    .clarification_exhausted
                ),
            )
        )

        recommendations: list[
            dict
        ]

        if not candidates:

            recommendations = []

        else:

            # --------------------------------------------------
            # 3. SELECT RANKING POLICY
            # --------------------------------------------------

            if (
                state
                .clarification_exhausted
            ):

                ranked = (
                    rerank_for_exploration(
                        candidates,
                        state,
                    )
                )

                # Do not repeat products once
                # the search enters exploration.
                ranked = [
                    candidate

                    for candidate
                    in ranked

                    if (
                        candidate[
                            "parent_asin"
                        ]

                        not in

                        state
                        .recommended_asins
                    )
                ]

            else:

                ranked = (
                    rerank_candidates(
                        candidates,
                        state,
                    )
                )

            # --------------------------------------------------
            # 4. RETURN TOP K
            # --------------------------------------------------

            recommendations = [
                {
                    "parent_asin":
                        item[
                            "parent_asin"
                        ]
                }

                for item
                in ranked[
                    :top_k
                ]
            ]

        # --------------------------------------------------
        # 5. REMEMBER PRODUCTS ALREADY SHOWN
        # --------------------------------------------------

        state.record_recommendations(
            [
                item[
                    "parent_asin"
                ]

                for item
                in recommendations
            ]
        )

        # --------------------------------------------------
        # 6. CHOOSE NEXT CONVERSATIONAL ACTION
        # --------------------------------------------------

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