from __future__ import annotations

import unittest

from src.intent import (
    ShoppingIntent,
)

from src.reranker import (
    configure_buying_relevance_labels,
    rerank_candidates,
    rerank_for_exploration,
)

from src.state import (
    Evidence,
    SessionState,
)


def _color_label_candidates() -> list[dict]:
    """
    Two candidates for the same "color: red"
    evidence chunk: one matches the color but is
    unpopular, the other only mentions "color" (a
    different one) but is very popular. Under plain
    token-coverage scoring both tie at 1-of-2 chunk
    tokens present, so popularity decides -- which
    wrongly promotes the mismatched candidate.
    """

    return [
        {
            "parent_asin":
                "wrong_color_but_popular",

            "title":
                "Running Shoe",

            "categories":
                "Shoes",

            "searchable_text":
                "Shoes Color Blue Running Shoe",

            "rating_number":
                5000,
        },
        {
            "parent_asin":
                "true_match",

            "title":
                "Red Running Shoe",

            "categories":
                "Shoes",

            "searchable_text":
                "Shoes Red Running Shoe",

            "rating_number":
                10,
        },
    ]


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


    def test_bm25_order_beats_popularity_with_no_evidence_yet(
        self,
    ) -> None:

        state = SessionState(
            user_profile={}
        )

        state.category_text = (
            "Robes"
        )

        # No state.evidence set -- mirrors turn 1
        # of a browsing session before any
        # constraint has been disclosed.

        candidates = [
            {
                "parent_asin":
                    "retrieved_first",

                "title":
                    "Fleece Robe",

                "categories":
                    "Robes",

                "searchable_text":
                    "Robes",

                "rating_number":
                    50,
            },
            {
                "parent_asin":
                    "more_popular_but_retrieved_second",

                "title":
                    "Fleece Robe",

                "categories":
                    "Robes",

                "searchable_text":
                    "Robes",

                "rating_number":
                    5000,
            },
        ]

        ranked = rerank_candidates(
            candidates,
            state,
        )

        # With no evidence, BM25/retrieval order
        # should win over raw popularity.
        self.assertEqual(
            ranked[0][
                "parent_asin"
            ],
            "retrieved_first",
        )


    def test_popularity_still_breaks_ties_once_evidence_exists(
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
                text="Tie closure",
            )
        ]

        candidates = [
            {
                "parent_asin":
                    "retrieved_first",

                "title":
                    "Fleece Robe",

                "categories":
                    "Robes",

                "searchable_text":
                    "Robes Tie closure",

                "rating_number":
                    50,
            },
            {
                "parent_asin":
                    "more_popular_but_retrieved_second",

                "title":
                    "Fleece Robe",

                "categories":
                    "Robes",

                "searchable_text":
                    "Robes Tie closure",

                "rating_number":
                    5000,
            },
        ]

        ranked = rerank_candidates(
            candidates,
            state,
        )

        # Once evidence exists, popularity should
        # still win over retrieval order -- V15's
        # change must be a no-op here.
        self.assertEqual(
            ranked[0][
                "parent_asin"
            ],
            "more_popular_but_retrieved_second",
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


    def test_buying_intent_strips_attribute_label_for_exact_match(
        self,
    ) -> None:

        state = SessionState(
            user_profile={}
        )

        state.intent = (
            ShoppingIntent.BUYING
        )

        state.evidence = [
            Evidence(
                turn=2,
                text="color: red",
            )
        ]

        ranked = rerank_candidates(
            _color_label_candidates(),
            state,
        )

        # Once the "color:" label is stripped, the
        # bare "red" chunk exact-matches the true
        # target's text, beating the popular but
        # wrong-colored decoy.
        self.assertEqual(
            ranked[0][
                "parent_asin"
            ],
            "true_match",
        )


    def test_non_buying_intent_relevance_unaffected(
        self,
    ) -> None:

        state = SessionState(
            user_profile={}
        )

        state.intent = (
            ShoppingIntent.BROWSING
        )

        state.evidence = [
            Evidence(
                turn=2,
                text="color: red",
            )
        ]

        ranked = rerank_candidates(
            _color_label_candidates(),
            state,
        )

        # Without buying intent, label stripping
        # never applies -- both candidates tie on
        # partial coverage and popularity decides,
        # exactly as before this experiment.
        self.assertEqual(
            ranked[0][
                "parent_asin"
            ],
            "wrong_color_but_popular",
        )


    def test_exploration_path_never_strips_attribute_labels(
        self,
    ) -> None:

        evidence = [
            Evidence(
                turn=2,
                text="color: red",
            )
        ]

        buying_state = SessionState(
            user_profile={}
        )

        buying_state.intent = (
            ShoppingIntent.BUYING
        )

        buying_state.evidence = evidence

        browsing_state = SessionState(
            user_profile={}
        )

        browsing_state.intent = (
            ShoppingIntent.BROWSING
        )

        browsing_state.evidence = evidence

        # Exploration mode never passes
        # strip_attribute_labels to
        # _candidate_relevance, so its ranking must
        # be identical regardless of state.intent --
        # proving Browsing/exploration ranking is
        # provably unaffected by this experiment.
        self.assertEqual(
            [
                item["parent_asin"]
                for item in rerank_for_exploration(
                    _color_label_candidates(),
                    buying_state,
                )
            ],
            [
                item["parent_asin"]
                for item in rerank_for_exploration(
                    _color_label_candidates(),
                    browsing_state,
                )
            ],
        )


    def test_buying_only_mode_excludes_override_turns(
        self,
    ) -> None:

        configure_buying_relevance_labels(
            "buying_only"
        )

        try:

            state = SessionState(
                user_profile={}
            )

            state.intent = (
                ShoppingIntent.BUYING
            )

            state.override_seen = True

            state.evidence = [
                Evidence(
                    turn=2,
                    text="color: red",
                )
            ]

            ranked = rerank_candidates(
                _color_label_candidates(),
                state,
            )

            self.assertEqual(
                ranked[0][
                    "parent_asin"
                ],
                "wrong_color_but_popular",
            )

        finally:

            configure_buying_relevance_labels(
                "both"
            )


    def test_override_only_mode_fires_after_override(
        self,
    ) -> None:

        configure_buying_relevance_labels(
            "override_only"
        )

        try:

            state = SessionState(
                user_profile={}
            )

            state.intent = (
                ShoppingIntent.BUYING
            )

            state.override_seen = True

            state.evidence = [
                Evidence(
                    turn=2,
                    text="color: red",
                )
            ]

            ranked = rerank_candidates(
                _color_label_candidates(),
                state,
            )

            self.assertEqual(
                ranked[0][
                    "parent_asin"
                ],
                "true_match",
            )

        finally:

            configure_buying_relevance_labels(
                "both"
            )


    def test_off_mode_disables_stripping_entirely(
        self,
    ) -> None:

        configure_buying_relevance_labels(
            "off"
        )

        try:

            state = SessionState(
                user_profile={}
            )

            state.intent = (
                ShoppingIntent.BUYING
            )

            state.evidence = [
                Evidence(
                    turn=2,
                    text="color: red",
                )
            ]

            ranked = rerank_candidates(
                _color_label_candidates(),
                state,
            )

            self.assertEqual(
                ranked[0][
                    "parent_asin"
                ],
                "wrong_color_but_popular",
            )

        finally:

            configure_buying_relevance_labels(
                "both"
            )


if __name__ == "__main__":
    unittest.main()