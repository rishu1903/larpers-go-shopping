from __future__ import annotations

import unittest

from src.dialogue import (
    choose_clarification,
)

from src.hard_constraints import (
    BudgetConstraint,
)

from src.intent import (
    ShoppingIntent,
)

from src.questions import (
    choose_candidate_attribute,
)

from src.state import (
    Evidence,
    SessionState,
)


class DialogueRobustnessTest(
    unittest.TestCase
):

    def test_generic_dissatisfaction_does_not_force_buying(
        self,
    ) -> None:

        state = SessionState(
            user_profile={}
        )

        state.update(
            (
                "I am still exploring "
                "jackets"
            ),
            1,
        )

        state.update(
            (
                "Those options are not "
                "quite right yet."
            ),
            2,
        )

        self.assertEqual(
            state.intent,
            ShoppingIntent.BROWSING,
        )


    def test_explicit_preference_moves_browsing_to_buying(
        self,
    ) -> None:

        state = SessionState(
            user_profile={}
        )

        state.update(
            (
                "I am still exploring "
                "jackets"
            ),
            1,
        )

        state.update(
            (
                "I would prefer something "
                "waterproof"
            ),
            2,
        )

        self.assertEqual(
            state.intent,
            ShoppingIntent.BUYING,
        )

        self.assertEqual(
            state.intent_history,
            [
                (
                    1,
                    "browsing",
                ),
                (
                    2,
                    "buying",
                ),
            ],
        )


    def test_override_keeps_later_budget_but_drops_stale_turn_one_evidence(
        self,
    ) -> None:

        state = SessionState(
            user_profile={}
        )

        state.update(
            (
                "I'm looking for Shoes. "
                "A key requirement is: "
                "leather."
            ),
            1,
        )

        state.update(
            (
                "My budget is "
                "$100."
            ),
            2,
        )

        state.update(
            (
                "Actually, ignore my earlier "
                "preference. What I need is: "
                "waterproof."
            ),
            3,
        )

        self.assertNotIn(
            "leather",
            state.active_text().lower(),
        )

        self.assertIn(
            "waterproof",
            state.active_text().lower(),
        )

        self.assertEqual(
            state.budget_constraint,
            BudgetConstraint(
                max_price=100.0,
            ),
        )

        self.assertEqual(
            state.budget_source_turn,
            2,
        )


    def test_measurement_does_not_become_budget_then_real_budget_does(
        self,
    ) -> None:

        state = SessionState(
            user_profile={}
        )

        state.update(
            (
                "I'm looking for "
                "Watches."
            ),
            1,
        )

        state.update(
            (
                "For that, what matters is: "
                "fits up to 8-inch wrist "
                "circumference."
            ),
            2,
        )

        self.assertIsNone(
            state.budget_constraint
        )

        state.update(
            (
                "Keep the price "
                "under $80."
            ),
            3,
        )

        self.assertEqual(
            state.budget_constraint,
            BudgetConstraint(
                max_price=80.0,
            ),
        )


    def test_specific_no_preference_does_not_exhaust_dialogue(
        self,
    ) -> None:

        state = SessionState(
            user_profile={}
        )

        state.update(
            (
                "I'm looking for "
                "Shoes."
            ),
            1,
        )

        state.update(
            (
                "I don't have a preference "
                "for material."
            ),
            2,
        )

        self.assertIn(
            "material",
            state.no_preference,
        )

        self.assertFalse(
            state.clarification_exhausted
        )


    def test_broad_other_no_preference_exhausts_dialogue(
        self,
    ) -> None:

        state = SessionState(
            user_profile={}
        )

        state.update(
            (
                "I'm looking for "
                "Shoes."
            ),
            1,
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


    def test_turn_ten_never_asks_another_question(
        self,
    ) -> None:

        state = SessionState(
            user_profile={}
        )

        (
            attribute,
            message,
        ) = choose_clarification(
            state=state,
            turn=10,
            candidates=[],
        )

        self.assertIsNone(
            attribute
        )

        self.assertTrue(
            message
        )


    def test_recommendation_memory_deduplicates_seen_products(
        self,
    ) -> None:

        state = SessionState(
            user_profile={}
        )

        state.record_recommendations(
            [
                "A",
                "B",
            ]
        )

        state.record_recommendations(
            [
                "B",
                "C",
            ]
        )

        self.assertEqual(
            state.recommended_asins,
            {
                "A",
                "B",
                "C",
            },
        )


    def test_current_session_value_prevents_profile_reasking_same_dimension(
        self,
    ) -> None:

        state = SessionState(
            user_profile={
                "preference_tags": [
                    "material",
                ]
            }
        )

        state.evidence = [
            Evidence(
                turn=2,
                text="cotton",
            )
        ]

        candidates = [
            {
                "searchable_text":
                    "cotton black shirt"
            },

            {
                "searchable_text":
                    "polyester white shirt"
            },

            {
                "searchable_text":
                    "nylon red shirt"
            },

            {
                "searchable_text":
                    "leather blue shirt"
            },
        ]

        self.assertNotEqual(
            choose_candidate_attribute(
                state=state,
                candidates=candidates,
                turn=4,
            ),
            "material",
        )


if __name__ == "__main__":
    unittest.main()