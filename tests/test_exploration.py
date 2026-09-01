from __future__ import annotations

import unittest

from src.reranker import (
    rerank_for_exploration,
)

from src.state import (
    Evidence,
    SessionState,
)


class ExplorationTest(
    unittest.TestCase
):

    def test_additional_no_preference_enables_exploration(
        self,
    ) -> None:

        state = SessionState(
            user_profile={}
        )

        state.update(
            (
                "I don't have an additional "
                "preference for other."
            ),
            4,
        )

        self.assertTrue(
            state.clarification_exhausted
        )

    def test_record_recommendations_tracks_seen_products(
        self,
    ) -> None:

        state = SessionState(
            user_profile={}
        )

        state.record_recommendations(
            [
                "A",
                "B",
                "A",
            ]
        )

        self.assertEqual(
            state.recommended_asins,
            {
                "A",
                "B",
            },
        )

    def test_exploration_keeps_relevance_ahead_of_popularity(
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
                    "popular_but_weaker",

                "title":
                    "Walking Shoes",

                "categories":
                    "Walking Shoes",

                "searchable_text":
                    (
                        "Walking Shoes "
                        "waterproof casual"
                    ),

                "rating_number":
                    50000,
            },
            {
                "parent_asin":
                    "rare_but_relevant",

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
                    1,
            },
        ]

        ranked = (
            rerank_for_exploration(
                candidates,
                state,
            )
        )

        self.assertEqual(
            ranked[0][
                "parent_asin"
            ],
            "rare_but_relevant",
        )

    def test_exploration_prefers_long_tail_inside_equal_relevance_tier(
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
                    "popular",

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
            {
                "parent_asin":
                    "long_tail",

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
                    5,
            },
        ]

        ranked = (
            rerank_for_exploration(
                candidates,
                state,
            )
        )

        self.assertEqual(
            ranked[0][
                "parent_asin"
            ],
            "long_tail",
        )

    def test_exploration_prefers_higher_rating_as_last_resort_tie_break(
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
                    "lower_rated",

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
                    5,

                "average_rating":
                    3.2,
            },
            {
                "parent_asin":
                    "higher_rated",

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
                    5,

                "average_rating":
                    4.8,
            },
        ]

        ranked = (
            rerank_for_exploration(
                candidates,
                state,
            )
        )

        self.assertEqual(
            ranked[0][
                "parent_asin"
            ],
            "higher_rated",
        )


if __name__ == "__main__":
    unittest.main()