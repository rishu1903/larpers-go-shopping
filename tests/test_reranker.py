from __future__ import annotations

import unittest

from src.reranker import (
    rerank_candidates,
)

from src.state import (
    Evidence,
    SessionState,
)


class RerankerTest(
    unittest.TestCase
):

    def test_exact_constraint_match_can_beat_bm25_order(
        self,
    ) -> None:

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
                    "waterproof; "
                    "rubber sole"
                ),
            )
        ]

        candidates = [
            {
                "parent_asin":
                    "generic",

                "title":
                    "Walking Shoes",

                "categories":
                    "Walking Shoes",

                "searchable_text":
                    (
                        "Walking Shoes "
                        "waterproof casual"
                    ),

                # Give the less relevant
                # candidate much greater
                # popularity deliberately.
                "rating_number":
                    5000,
            },
            {
                "parent_asin":
                    "target",

                "title":
                    (
                        "Waterproof "
                        "Walking Shoe"
                    ),

                "categories":
                    "Walking Shoes",

                "searchable_text":
                    (
                        "Walking Shoes "
                        "waterproof "
                        "rubber sole"
                    ),

                "rating_number":
                    10,
            },
        ]

        ranked = rerank_candidates(
            candidates,
            state,
        )

        # Relevance must beat popularity.
        self.assertEqual(
            ranked[0][
                "parent_asin"
            ],
            "target",
        )


    def test_popularity_breaks_equal_relevance_ties(
        self,
    ) -> None:

        state = SessionState(
            user_profile={}
        )

        state.category_text = (
            "Robes"
        )

        state.evidence = [
            Evidence(
                turn=2,
                text=(
                    "100% Polyester; "
                    "Tie closure"
                ),
            )
        ]

        candidates = [
            {
                "parent_asin":
                    "less_popular",

                "title":
                    "Fleece Robe",

                "categories":
                    "Robes",

                "searchable_text":
                    (
                        "Robes "
                        "100% Polyester "
                        "Tie closure"
                    ),

                "rating_number":
                    50,
            },
            {
                "parent_asin":
                    "more_popular",

                "title":
                    "Fleece Robe",

                "categories":
                    "Robes",

                "searchable_text":
                    (
                        "Robes "
                        "100% Polyester "
                        "Tie closure"
                    ),

                "rating_number":
                    5000,
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
            "more_popular",
        )


    def test_bm25_order_is_final_tie_breaker(
        self,
    ) -> None:

        state = SessionState(
            user_profile={}
        )

        state.category_text = (
            "Wallets"
        )

        candidates = [
            {
                "parent_asin":
                    "first",

                "title":
                    "Wallet",

                "categories":
                    "Wallets",

                "searchable_text":
                    "Wallet",

                "rating_number":
                    100,
            },
            {
                "parent_asin":
                    "second",

                "title":
                    "Wallet",

                "categories":
                    "Wallets",

                "searchable_text":
                    "Wallet",

                "rating_number":
                    100,
            },
        ]

        ranked = rerank_candidates(
            candidates,
            state,
        )

        self.assertEqual(
            [
                item[
                    "parent_asin"
                ]
                for item
                in ranked
            ],
            [
                "first",
                "second",
            ],
        )


if __name__ == "__main__":
    unittest.main()