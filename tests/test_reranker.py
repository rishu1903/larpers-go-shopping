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
            },
        ]

        ranked = rerank_candidates(
            candidates,
            state,
        )

        self.assertEqual(
            ranked[0]["parent_asin"],
            "target",
        )


    def test_bm25_order_breaks_true_ties(
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
            },
        ]

        ranked = rerank_candidates(
            candidates,
            state,
        )

        self.assertEqual(
            [
                item["parent_asin"]
                for item in ranked
            ],
            [
                "first",
                "second",
            ],
        )


if __name__ == "__main__":
    unittest.main()