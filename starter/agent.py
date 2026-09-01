from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from src.dialogue import (
    choose_clarification,
)

from src.intent import (
    ShoppingIntent,
)

from src.hard_constraints import (
    apply_budget_constraint,
    coerce_price,
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
        Intent Router
            ↓
        Field-aware BM25
            +
        Conditional Semantic Retrieval
            ↓
        Hard Constraint Filtering
            ↓
        Relevance Reranking
            ↓
        Candidate-aware Clarification
            ↓
        Adaptive Exploration
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

        self._prices: dict[
            str,
            float | None,
        ] = {}

        self._asin_to_rowid: dict[
            str,
            int,
        ] = {}

        self._build_index()

        self.semantic = (
            SemanticRetriever()
        )

    def _build_index(
        self,
    ) -> None:

        cursor = (
            self.connection.cursor()
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

                # Price is deliberately NOT placed
                # in the FTS text index.
                #
                # It is structured numeric metadata
                # and will be used for actual hard
                # constraint enforcement.
                self._prices[
                    parent_asin
                ] = coerce_price(
                    product.get(
                        "price"
                    )
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
                            "INSERT INTO products "
                            "VALUES "
                            "(?, ?, ?, ?, ?, ?, ?)"
                        ),
                        batch,
                    )

                    batch.clear()

        if batch:

            cursor.executemany(
                (
                    "INSERT INTO products "
                    "VALUES "
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

        # ----------------------------------
        # 1. UPDATE CONVERSATIONAL STATE
        # ----------------------------------

        state.update(
            user_message=user_message,
            turn=turn,
        )

        # ----------------------------------
        # 2. RETRIEVE CANDIDATES
        # ----------------------------------

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

                catalog_size=(
                    len(
                        self._asin_to_rowid
                    )
                ),
            )
        )

        # ----------------------------------
        # 3. ATTACH STRUCTURED PRICE
        # ----------------------------------

        for candidate in candidates:

            candidate[
                "price"
            ] = self._prices.get(
                candidate[
                    "parent_asin"
                ]
            )

        # ----------------------------------
        # 4. APPLY HARD CONSTRAINTS
        # ----------------------------------
        #
        # This happens BEFORE reranking.
        #
        # A product that violates a hard budget
        # is ineligible regardless of lexical,
        # semantic or popularity score.

        candidates = (
            apply_budget_constraint(
                candidates,
                state.budget_constraint,
            )
        )

        recommendations: list[
            dict
        ] = []

        question_candidates: list[
            dict
        ] = []

        tied_pair: list[dict] | None = None

        width = top_k
        if candidates:

            # ----------------------------------
            # 5. SELECT RANKING POLICY
            # ----------------------------------

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

            # Candidate-aware dialogue policy
            # analyses the currently feasible
            # product set, not products that
            # violate hard requirements.
            question_candidates = (
                ranked[:30]
            )

            top_relevance = ranked[0].get(
                "relevance", 0.0
            )
            if (
                turn >= 4
                and len(ranked) >= 2
                and top_relevance > 0.0
                and top_relevance
                == ranked[1].get(
                    "relevance", 0.0
                )
            ):
                tied_pair = [ranked[0], ranked[1]]

            # ----------------------------------
            # 6. RETURN TOP K
            # ----------------------------------

            recommendations = [
                {
                    "parent_asin":
                        candidate[
                            "parent_asin"
                        ]
                }

                for candidate
                in ranked[:top_k]
            ]

        # ----------------------------------
        # 7. REMEMBER SHOWN PRODUCTS
        # ----------------------------------

        # Browsing sessions on turn 1 have no evidence beyond
        # the category, making the ranking a pure popularity
        # tiebreak. Withhold recommendations and let the
        # question cycle run first so turn 2 has real signal.
        if state.intent == ShoppingIntent.BROWSING and turn == 1:
            recommendations = []

        state.record_recommendations(
            [
                item[
                    "parent_asin"
                ]

                for item
                in recommendations
            ]
        )

        # ----------------------------------
        # 8. CHOOSE NEXT QUESTION
        # ----------------------------------

        (
            ask_attribute,
            message,
        ) = choose_clarification(
            state=state,
            turn=turn,
            candidates=(
                question_candidates
            ),
            tied_pair=(
                tied_pair
                if candidates
                else None
            ),
        )

        state.record_question(
            ask_attribute
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