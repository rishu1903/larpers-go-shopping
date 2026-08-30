from __future__ import annotations

import unittest

import src.fusion as fusion

from src.fusion import (
    category_coverage,
    configure_semantic_exploration_weight,
    normalized_rrf,
    semantic_exploration_bonus,
)

from src.reranker import (
    rerank_candidates,
    rerank_for_exploration,
)

from src.state import (
    Evidence,
    SessionState,
)


class SemanticExplorationFusionTest(
    unittest.TestCase
):

    def setUp(
        self,
    ) -> None:
        self.original_weight = (
            fusion
            .SEMANTIC_EXPLORATION_WEIGHT
        )

    def tearDown(
        self,
    ) -> None:
        configure_semantic_exploration_weight(
            self.original_weight
        )

    def _state(
        self,
    ) -> SessionState:
        state = SessionState(
            user_profile={}
        )

        state.category_text = (
            "Walking Shoes"
        )

        state.evidence = [
            Evidence(
                turn=2,
                text=(
                    "comfortable for "
                    "wet pavements"
                ),
            )
        ]

        return state

    def test_normalized_rrf_is_bounded_and_decreases_with_rank(
        self,
    ) -> None:
        self.assertEqual(
            normalized_rrf(1),
            1.0,
        )

        self.assertGreater(
            normalized_rrf(10),
            normalized_rrf(100),
        )

        self.assertEqual(
            normalized_rrf(None),
            0.0,
        )

    def test_category_coverage_blocks_cross_category_drift(
        self,
    ) -> None:
        state = self._state()

        compatible = {
            "title":
                "Trail Walking Shoes",

            "categories":
                "Walking Shoes",
        }

        unrelated = {
            "title":
                "Leather Wallet",

            "categories":
                "Wallets",
        }

        self.assertEqual(
            category_coverage(
                compatible,
                state,
            ),
            1.0,
        )

        self.assertEqual(
            category_coverage(
                unrelated,
                state,
            ),
            0.0,
        )

    def test_bonus_applies_only_to_semantic_only_candidates(
        self,
    ) -> None:
        configure_semantic_exploration_weight(
            1.0
        )

        state = self._state()

        base = {
            "title":
                "Walking Shoes",

            "categories":
                "Walking Shoes",
        }

        semantic_only = {
            **base,
            "source":
                "semantic",
        }

        hybrid = {
            **base,
            "source":
                "hybrid",
        }

        self.assertGreater(
            semantic_exploration_bonus(
                semantic_only,
                state,
                semantic_rank=10,
            ),
            0.0,
        )

        self.assertEqual(
            semantic_exploration_bonus(
                hybrid,
                state,
                semantic_rank=10,
            ),
            0.0,
        )

    def test_bonus_rejects_category_incompatible_semantic_candidate(
        self,
    ) -> None:
        configure_semantic_exploration_weight(
            1.0
        )

        state = self._state()

        candidate = {
            "title":
                "Leather Wallet",

            "categories":
                "Wallets",

            "source":
                "semantic",
        }

        self.assertEqual(
            semantic_exploration_bonus(
                candidate,
                state,
                semantic_rank=1,
            ),
            0.0,
        )

    def test_zero_weight_preserves_existing_exploration_order(
        self,
    ) -> None:
        configure_semantic_exploration_weight(
            0.0
        )

        state = self._state()

        candidates = [
            {
                "parent_asin":
                    "lexical",

                "title":
                    "Walking Shoes",

                "categories":
                    "Walking Shoes",

                "searchable_text":
                    (
                        "Walking Shoes "
                        "wet pavements"
                    ),

                "rating_number":
                    10,

                "semantic_score":
                    0.0,

                "source":
                    "lexical",
            },
            {
                "parent_asin":
                    "semantic",

                "title":
                    "Walking Shoes",

                "categories":
                    "Walking Shoes",

                "searchable_text":
                    "Walking Shoes",

                "rating_number":
                    1,

                "semantic_score":
                    0.95,

                "source":
                    "semantic",
            },
        ]

        ranked = rerank_for_exploration(
            candidates,
            state,
        )

        self.assertEqual(
            ranked[0][
                "parent_asin"
            ],
            "lexical",
        )

    def test_positive_weight_can_promote_near_tie_semantic_candidate(
        self,
    ) -> None:
        configure_semantic_exploration_weight(
            1.0
        )

        state = SessionState(
            user_profile={}
        )

        state.category_text = (
            "Robes"
        )

        candidates = [
            {
                "parent_asin":
                    "lexical",

                "title":
                    "Robe",

                "categories":
                    "Robes",

                "searchable_text":
                    "Robes",

                "rating_number":
                    1,

                "semantic_score":
                    0.0,

                "source":
                    "lexical",
            },
            {
                "parent_asin":
                    "semantic",

                "title":
                    "Robe",

                "categories":
                    "Robes",

                "searchable_text":
                    "Robes",

                "rating_number":
                    100,

                "semantic_score":
                    0.95,

                "source":
                    "semantic",
            },
        ]

        ranked = rerank_for_exploration(
            candidates,
            state,
        )

        self.assertEqual(
            ranked[0][
                "parent_asin"
            ],
            "semantic",
        )

    def test_exploitation_reranker_ignores_semantic_fusion_weight(
        self,
    ) -> None:
        configure_semantic_exploration_weight(
            10.0
        )

        state = SessionState(
            user_profile={}
        )

        state.category_text = (
            "Robes"
        )

        candidates = [
            {
                "parent_asin":
                    "popular",

                "title":
                    "Robe",

                "categories":
                    "Robes",

                "searchable_text":
                    "Robes",

                "rating_number":
                    500,

                "semantic_score":
                    0.0,

                "source":
                    "lexical",
            },
            {
                "parent_asin":
                    "semantic",

                "title":
                    "Robe",

                "categories":
                    "Robes",

                "searchable_text":
                    "Robes",

                "rating_number":
                    1,

                "semantic_score":
                    0.99,

                "source":
                    "semantic",
            },
        ]

        ranked = rerank_candidates(
            candidates,
            state,
        )

        self.assertEqual(
            ranked[0][
                "parent_asin"
            ],
            "popular",
        )


if __name__ == "__main__":
    unittest.main()