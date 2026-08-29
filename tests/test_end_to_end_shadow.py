from __future__ import annotations

import unittest

from scripts.end_to_end_shadow_eval import (
    cumulative_session_metrics,
    normalize_recommendations,
    recommendation_metrics,
    summarize_results,
)


class EndToEndShadowTest(
    unittest.TestCase
):

    def test_normalizes_string_recommendations(
        self,
    ) -> None:

        self.assertEqual(
            normalize_recommendations(
                [
                    "A",
                    "B",
                    "A",
                    "",
                ]
            ),
            [
                "A",
                "B",
            ],
        )


    def test_normalizes_dictionary_recommendations(
        self,
    ) -> None:

        self.assertEqual(
            normalize_recommendations(
                [
                    {
                        "parent_asin":
                            "A"
                    },
                    {
                        "asin":
                            "B"
                    },
                    {
                        "id":
                            "C"
                    },
                ]
            ),
            [
                "A",
                "B",
                "C",
            ],
        )


    def test_recommendation_metrics_uses_any_positive(
        self,
    ) -> None:

        metrics = recommendation_metrics(
            recommendations=[
                "X",
                "Y",
                "B",
                "Z",
            ],
            relevant_asins={
                "A",
                "B",
            },
            top_k=10,
        )

        self.assertTrue(
            metrics[
                "hit"
            ]
        )

        self.assertEqual(
            metrics[
                "first_relevant_rank"
            ],
            3,
        )

        self.assertEqual(
            metrics[
                "reciprocal_rank"
            ],
            0.333333,
        )


    def test_session_keeps_initial_hit_even_if_exploration_misses(
        self,
    ) -> None:

        initial = {
            "hit":
                True,

            "first_relevant_rank":
                2,

            "reciprocal_rank":
                0.5,

            "relevant_hits":
                1,
        }

        exploration = {
            "hit":
                False,

            "first_relevant_rank":
                None,

            "reciprocal_rank":
                0.0,

            "relevant_hits":
                0,
        }

        session = cumulative_session_metrics(
            initial=initial,
            exploration=exploration,
            relevant_count=2,
        )

        self.assertTrue(
            session[
                "hit_by_turn_2"
            ]
        )

        self.assertEqual(
            session[
                "first_hit_turn"
            ],
            1,
        )

        self.assertEqual(
            session[
                "first_hit_reciprocal_rank"
            ],
            0.5,
        )


    def test_session_records_second_turn_rescue(
        self,
    ) -> None:

        initial = {
            "hit":
                False,

            "first_relevant_rank":
                None,

            "reciprocal_rank":
                0.0,

            "relevant_hits":
                0,
        }

        exploration = {
            "hit":
                True,

            "first_relevant_rank":
                5,

            "reciprocal_rank":
                0.2,

            "relevant_hits":
                1,
        }

        session = cumulative_session_metrics(
            initial=initial,
            exploration=exploration,
            relevant_count=4,
        )

        self.assertTrue(
            session[
                "hit_by_turn_2"
            ]
        )

        self.assertEqual(
            session[
                "first_hit_turn"
            ],
            2,
        )

        self.assertEqual(
            session[
                "recall_across_20"
            ],
            0.25,
        )


    def test_summary_counts_cumulative_rescue(
        self,
    ) -> None:

        results = [
            {
                "concept":
                    "waterproof",

                "transition":
                    "rescued_after_exploration",

                "v10_2": {
                    "lexical_first_relevant_rank":
                        None,

                    "semantic_first_relevant_rank":
                        40,
                },

                "initial": {
                    "hit":
                        False,

                    "reciprocal_rank":
                        0.0,

                    "precision_at_10":
                        0.0,

                    "recall_at_10":
                        0.0,

                    "recommendation_count":
                        10,
                },

                "after_exploration": {
                    "hit":
                        True,

                    "reciprocal_rank":
                        0.5,

                    "precision_at_10":
                        0.1,

                    "recall_at_10":
                        0.25,

                    "recommendation_count":
                        10,
                },

                "session": {
                    "hit_by_turn_2":
                        True,

                    "first_hit_turn":
                        2,

                    "first_hit_reciprocal_rank":
                        0.5,

                    "recall_across_20":
                        0.25,

                    "recommendation_overlap":
                        0,
                },
            },

            {
                "concept":
                    "waterproof",

                "transition":
                    "initial_only",

                "v10_2": {
                    "lexical_first_relevant_rank":
                        2,

                    "semantic_first_relevant_rank":
                        None,
                },

                "initial": {
                    "hit":
                        True,

                    "reciprocal_rank":
                        0.5,

                    "precision_at_10":
                        0.1,

                    "recall_at_10":
                        0.25,

                    "recommendation_count":
                        10,
                },

                "after_exploration": {
                    "hit":
                        False,

                    "reciprocal_rank":
                        0.0,

                    "precision_at_10":
                        0.0,

                    "recall_at_10":
                        0.0,

                    "recommendation_count":
                        10,
                },

                "session": {
                    "hit_by_turn_2":
                        True,

                    "first_hit_turn":
                        1,

                    "first_hit_reciprocal_rank":
                        0.5,

                    "recall_across_20":
                        0.25,

                    "recommendation_overlap":
                        0,
                },
            },
        ]

        summary = summarize_results(
            results
        )

        session = summary[
            "session_by_turn_2"
        ]

        self.assertEqual(
            session[
                "cumulative_hit_rate_by_turn_2"
            ],
            1.0,
        )

        self.assertEqual(
            session[
                "first_turn_hits"
            ],
            1,
        )

        self.assertEqual(
            session[
                "second_turn_rescues"
            ],
            1,
        )

        self.assertEqual(
            session[
                "rescue_rate_among_initial_misses"
            ],
            1.0,
        )


if __name__ == "__main__":
    unittest.main()