from __future__ import annotations

import unittest

from src.questions import (
    choose_candidate_attribute,
)

from src.state import (
    Evidence,
    SessionState,
)


class QuestionPolicyTest(
    unittest.TestCase
):

    def test_early_turns_use_broad_discovery(
        self,
    ) -> None:

        state = SessionState(
            user_profile={}
        )

        candidates = [
            {
                "searchable_text":
                    "cotton black"
            },
            {
                "searchable_text":
                    "polyester white"
            },
        ]

        self.assertEqual(
            choose_candidate_attribute(
                state,
                candidates,
                turn=2,
            ),
            "other",
        )


    def test_late_turn_chooses_high_information_material(
        self,
    ) -> None:

        state = SessionState(
            user_profile={}
        )

        # Color is identical.
        #
        # Material varies heavily.
        #
        # Therefore material should have
        # greater information value.
        candidates = [
            {
                "searchable_text":
                    "cotton black shirt"
            },
            {
                "searchable_text":
                    "polyester black shirt"
            },
            {
                "searchable_text":
                    "leather black shirt"
            },
            {
                "searchable_text":
                    "nylon black shirt"
            },
        ]

        self.assertEqual(
            choose_candidate_attribute(
                state,
                candidates,
                turn=4,
            ),
            "material",
        )


    def test_known_material_is_not_asked_again(
        self,
    ) -> None:

        state = SessionState(
            user_profile={}
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
                    "leather red shirt"
            },
            {
                "searchable_text":
                    "nylon blue shirt"
            },
        ]

        # Material is already known.
        #
        # Color becomes the next useful
        # discriminator.
        self.assertEqual(
            choose_candidate_attribute(
                state,
                candidates,
                turn=4,
            ),
            "color",
        )


    def test_already_asked_attribute_is_not_repeated(
        self,
    ) -> None:

        state = SessionState(
            user_profile={}
        )

        # No evidence reveals material, but the
        # agent already spent a turn asking about
        # it -- it should not be asked again.
        state.record_question(
            "material"
        )

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
                    "leather red shirt"
            },
            {
                "searchable_text":
                    "nylon blue shirt"
            },
        ]

        self.assertEqual(
            choose_candidate_attribute(
                state,
                candidates,
                turn=4,
            ),
            "color",
        )


    def test_overloaded_ambiguous_pool_stays_on_broad_discovery(
        self,
    ) -> None:

        state = SessionState(
            user_profile={}
        )

        # A large, genuinely ambiguous pool --
        # none of the candidates mention any
        # recognised attribute value at all, so
        # there is no real signal to differentiate
        # on, regardless of pool size. Clarification
        # should stay broad ("other") rather than
        # guess at a dimension with zero evidence.
        candidates = [
            {
                "searchable_text":
                    "premium quality item"
            }

            for _
            in range(20)
        ]

        self.assertEqual(
            choose_candidate_attribute(
                state,
                candidates,
                turn=2,
            ),
            "other",
        )


    def test_specific_no_preference_does_not_exhaust_all_clarification(
        self,
    ) -> None:

        state = SessionState(
            user_profile={}
        )

        state.update(
            (
                "I don't have an additional "
                "preference for material."
            ),
            4,
        )

        self.assertIn(
            "material",
            state.no_preference,
        )

        self.assertFalse(
            state.clarification_exhausted
        )

        # A failed broad clarification means
        # there is genuinely nothing more
        # available to ask.
        state.update(
            (
                "I don't have an additional "
                "preference for other."
            ),
            5,
        )

        self.assertTrue(
            state.clarification_exhausted
        )


if __name__ == "__main__":
    unittest.main()