from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from src.dialogue import (
    choose_clarification,
)

from src.hard_constraints import (
    apply_budget_constraint,
    coerce_price,
)

from src.orchestration import (
    RetrievalPlan,
    retrieval_plan,
    select_protected_recovery,
    should_use_protected_recovery,
    v12_retrieval_plan,
)

from src.relevance_gate import (
    filter_observed_exclusions,
)

from src.shopping_intent import (
    CompiledShoppingIntent,
    compile_shopping_intent,
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

            for (
                key,
                item,
            ) in value.items()
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

    return str(
        value
    )


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
            ↓
        Failure-Aware Protected Recovery
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

        # Preserve structured details cheaply for the rare exclusion path.
        # Keeping compact JSON avoids duplicating the full catalogue in memory,
        # while allowing ProductProfile to retain native key provenance.
        self._details_json: dict[str, str] = {}

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

                product = (
                    json.loads(
                        line
                    )
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

                # Price is deliberately excluded
                # from the full-text index.
                #
                # It remains structured numeric
                # metadata for real filtering.
                self._prices[
                    parent_asin
                ] = coerce_price(
                    product.get(
                        "price"
                    )
                )

                details = product.get("details")
                if not isinstance(details, dict):
                    details = {}
                self._details_json[parent_asin] = json.dumps(
                    details,
                    separators=(",", ":"),
                    ensure_ascii=False,
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
            user_profile=(
                user_profile
            )
        )

    def _rank_for_plan(
        self,
        state: SessionState,
        *,
        exploration: bool,
        plan_override: (
            RetrievalPlan
            | None
        ) = None,
        compiled_intent: (
            CompiledShoppingIntent
            | None
        ) = None,
    ) -> list[dict]:
        """
        Execute one complete candidate plan:

            retrieval
                ↓
            price attachment
                ↓
            hard filtering
                ↓
            reranking
                ↓
            seen-item filtering

        V13 uses this helper twice only after an
        explicit recommendation failure:

        1. frozen V12 continuation;
        2. expanded failure-recovery plan.
        """

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
                    exploration
                ),

                plan_override=(
                    plan_override
                ),
            )
        )

        for candidate in candidates:

            candidate[
                "price"
            ] = self._prices.get(
                candidate[
                    "parent_asin"
                ]
            )

        candidates = (
            apply_budget_constraint(
                candidates,
                state.budget_constraint,
            )
        )

        if not candidates:
            return []

        if compiled_intent is not None and compiled_intent.exclusions:
            for candidate in candidates:
                details_json = self._details_json.get(
                    candidate["parent_asin"],
                    "{}",
                )
                candidate["details"] = json.loads(details_json)

        if exploration:

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

            return filter_observed_exclusions(
                ranked,
                compiled_intent,
            )

        ranked = (
            rerank_candidates(
                candidates,
                state,
            )
        )

        return filter_observed_exclusions(
            ranked,
            compiled_intent,
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

        state = (
            self._sessions[
                session_id
            ]
        )

        # ----------------------------------
        # 1. UPDATE CONVERSATIONAL STATE
        # ----------------------------------

        state.update(
            user_message=user_message,
            turn=turn,
        )

        compiled_intent = compile_shopping_intent(
            state=state,
            user_message=user_message,
            turn=turn,
        )

        exploration = (
            state
            .clarification_exhausted
        )

        # ----------------------------------
        # 2. RANK CANDIDATES
        # ----------------------------------
        #
        # Normal path:
        #
        #     exactly the existing V12/V13A
        #     candidate plan.
        #
        # Failure recovery:
        #
        #     rank frozen V12 continuation
        #             +
        #     rank expanded recovery pool
        #
        # The final selector protects nearly
        # the entire V12 Top-K and gives only
        # a bounded slot budget to new deep
        # candidates.

        if (
            should_use_protected_recovery(
                state=state,
                exploration=exploration,
            )
        ):

            baseline_ranked = (
                self._rank_for_plan(
                    state,
                    exploration=True,
                    plan_override=(
                        v12_retrieval_plan(
                            exploration=True
                        )
                    ),
                    compiled_intent=compiled_intent,
                )
            )

            expanded_ranked = (
                self._rank_for_plan(
                    state,
                    exploration=True,
                    plan_override=(
                        retrieval_plan(
                            state=state,
                            exploration=True,
                        )
                    ),
                    compiled_intent=compiled_intent,
                )
            )

            ranked = (
                select_protected_recovery(
                    baseline_ranked=(
                        baseline_ranked
                    ),
                    expanded_ranked=(
                        expanded_ranked
                    ),
                    top_k=top_k,
                )
            )

            # Clarification has already been
            # exhausted on this branch.
            #
            # Retain the V12 pool for any
            # downstream dialogue diagnostics.
            question_candidates = (
                baseline_ranked[
                    :30
                ]
            )

        else:

            ranked = (
                self._rank_for_plan(
                    state,
                    exploration=(
                        exploration
                    ),
                    compiled_intent=compiled_intent,
                )
            )

            question_candidates = (
                ranked[
                    :30
                ]
            )

        # ----------------------------------
        # 3. RETURN TOP K
        # ----------------------------------

        recommendations = [
            {
                "parent_asin":
                    candidate[
                        "parent_asin"
                    ]
            }

            for candidate
            in ranked[
                :top_k
            ]
        ]

        # ----------------------------------
        # 4. REMEMBER SHOWN PRODUCTS
        # ----------------------------------

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
        # 5. CHOOSE NEXT QUESTION
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
                "prompt_tokens":
                    0,

                "completion_tokens":
                    0,
            },
        }