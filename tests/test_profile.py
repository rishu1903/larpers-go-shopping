from __future__ import annotations

import unittest

from src.profile import (
    affinity_for_attribute,
    attribute_affinity,
    preference_tags,
)

from src.questions import (
    choose_candidate_attribute,
)

from src.state import (
    Evidence,
    SessionState,
)


class ProfilePersonalizationTest(
    unittest.TestCase
):

    def test_controlled_tags_map_to_safe_dimensions(
        self,
    ) -> None:

        profile = {
            "preference_tags": [
                "material",
                "fit",
                "comfort",
            ]
        }

        affinities = (
            attribute_affinity(
                profile
            )
        )

        self.assertEqual(
            affinities[
                "material"
            ],
            1.0,
        )

        self.assertEqual(
            affinities[
                "size"
            ],
            1.0,
        )

        self.assertGreater(
            affinities[
                "feature"
            ],
            0.0,
        )


    def test_summary_and_rating_style_do_not_create_preferences(
        self,
    ) -> None:

        profile = {
            "average_prior_rating":
                5.0,

            "rating_style":
                "usually positive",

            "summary":
                (
                    "This text mentions cotton, "
                    "black, slim fit and waterproof."
                ),

            "preference_tags":
                [],
        }

        self.assertEqual(
            preference_tags(
                profile
            ),
            [],
        )

        self.assertEqual(
            attribute_affinity(
                profile
            ),
            {},
        )


    def test_general_shopping_tag_creates_no_assumption(
        self,
    ) -> None:

        profile = {
            "preference_tags": [
                "general shopping",
            ]
        }

        self.assertEqual(
            attribute_affinity(
                profile
            ),
            {},
        )


    def test_profile_breaks_only_near_candidate_tie(
        self,
    ) -> None:

        state = SessionState(
            user_profile={
                "preference_tags": [
                    "style",
                ]
            }
        )

        # Color and style both divide the
        # candidates well.
        #
        # Color has a slightly stronger static
        # question prior, so without profile
        # personalization color would win.
        #
        # Style is still within the 90% safety
        # gate, therefore the profile may safely
        # break the near tie.
        candidates = [
            {
                "searchable_text":
                    "black slim shirt"
            },
            {
                "searchable_text":
                    "white regular shirt"
            },
            {
                "searchable_text":
                    "black regular shirt"
            },
            {
                "searchable_text":
                    "white slim shirt"
            },
        ]

        self.assertEqual(
            choose_candidate_attribute(
                state=state,
                candidates=candidates,
                turn=4,
            ),
            "style",
        )


    def test_profile_does_not_override_much_stronger_current_information(
        self,
    ) -> None:

        state = SessionState(
            user_profile={
                "preference_tags": [
                    "style",
                ]
            }
        )

        # Material is highly diverse.
        #
        # Style provides little useful
        # separation.
        candidates = [
            {
                "searchable_text":
                    "cotton slim shirt"
            },
            {
                "searchable_text":
                    "polyester slim shirt"
            },
            {
                "searchable_text":
                    "leather slim shirt"
            },
            {
                "searchable_text":
                    "nylon regular shirt"
            },
        ]

        self.assertEqual(
            choose_candidate_attribute(
                state=state,
                candidates=candidates,
                turn=4,
            ),
            "material",
        )


    def test_current_session_preference_beats_profile_history(
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
                    "leather red shirt"
            },
            {
                "searchable_text":
                    "nylon blue shirt"
            },
        ]

        # Material matters historically,
        # but the current shopper has already
        # specified cotton.
        #
        # We must not ask material again.
        self.assertEqual(
            choose_candidate_attribute(
                state=state,
                candidates=candidates,
                turn=4,
            ),
            "color",
        )


    def test_direct_future_profile_tag_is_supported_without_value_assumption(
        self,
    ) -> None:

        profile = {
            "preference_tags": [
                "color",
            ]
        }

        self.assertEqual(
            affinity_for_attribute(
                profile,
                "color",
            ),
            1.0,
        )

        self.assertEqual(
            affinity_for_attribute(
                profile,
                "material",
            ),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()