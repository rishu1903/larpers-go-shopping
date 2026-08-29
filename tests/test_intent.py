from __future__ import annotations

import unittest

from src.intent import (
    ShoppingIntent,
    infer_intent,
)

from src.retrieval import (
    should_use_semantic,
)

from src.state import (
    SessionState,
)


class IntentRoutingTest(
    unittest.TestCase
):

    def test_exploring_language_routes_to_browsing(
        self,
    ) -> None:

        (
            intent,
            confidence,
        ) = infer_intent(
            user_message=(
                "I'm looking for boots, "
                "but I'm still exploring."
            ),
            turn=1,
            current=None,
        )

        self.assertEqual(
            intent,
            ShoppingIntent.BROWSING,
        )

        self.assertGreater(
            confidence,
            0.80,
        )


    def test_hard_requirement_routes_to_buying(
        self,
    ) -> None:

        (
            intent,
            confidence,
        ) = infer_intent(
            user_message=(
                "I'm looking for boots. "
                "A key requirement is: "
                "waterproof."
            ),
            turn=1,
            current=None,
        )

        self.assertEqual(
            intent,
            ShoppingIntent.BUYING,
        )

        self.assertGreater(
            confidence,
            0.80,
        )


    def test_simulator_narrowing_transitions_browsing_to_buying(
        self,
    ) -> None:

        state = SessionState(
            user_profile={}
        )

        state.update(
            (
                "I'm looking for boots, "
                "but I'm still exploring."
            ),
            1,
        )

        self.assertEqual(
            state.intent,
            ShoppingIntent.BROWSING,
        )

        state.update(
            (
                "For that, what matters is: "
                "waterproof; rubber sole."
            ),
            2,
        )

        self.assertEqual(
            state.intent,
            ShoppingIntent.BUYING,
        )


    def test_i_prefer_transitions_browsing_to_buying(
        self,
    ) -> None:

        state = SessionState(
            user_profile={}
        )

        state.update(
            (
                "I'm open to ideas "
                "for a jacket."
            ),
            1,
        )

        self.assertEqual(
            state.intent,
            ShoppingIntent.BROWSING,
        )

        state.update(
            (
                "I prefer something "
                "waterproof."
            ),
            2,
        )

        self.assertEqual(
            state.intent,
            ShoppingIntent.BUYING,
        )


    def test_id_prefer_transitions_browsing_to_buying(
        self,
    ) -> None:

        state = SessionState(
            user_profile={}
        )

        state.update(
            (
                "I'm still exploring "
                "jackets."
            ),
            1,
        )

        state.update(
            (
                "I'd prefer something "
                "waterproof."
            ),
            2,
        )

        self.assertEqual(
            state.intent,
            ShoppingIntent.BUYING,
        )


    def test_i_would_prefer_transitions_browsing_to_buying(
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

        self.assertEqual(
            state.intent,
            ShoppingIntent.BROWSING,
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


    def test_i_would_like_can_narrow_browsing(
        self,
    ) -> None:

        state = SessionState(
            user_profile={}
        )

        state.update(
            (
                "I'm just browsing "
                "for shoes."
            ),
            1,
        )

        state.update(
            (
                "I would like something "
                "waterproof."
            ),
            2,
        )

        self.assertEqual(
            state.intent,
            ShoppingIntent.BUYING,
        )


    def test_override_forces_buying(
        self,
    ) -> None:

        state = SessionState(
            user_profile={}
        )

        state.update(
            (
                "I'm looking for shirts, "
                "but I'm still exploring."
            ),
            1,
        )

        state.update(
            (
                "Actually, ignore my earlier "
                "preference. What I need is: "
                "cotton."
            ),
            3,
        )

        self.assertEqual(
            state.intent,
            ShoppingIntent.BUYING,
        )


    def test_buying_does_not_enable_dense_route_just_because_lexical_pool_is_small(
        self,
    ) -> None:

        state = SessionState(
            user_profile={}
        )

        state.intent = (
            ShoppingIntent.BUYING
        )

        self.assertFalse(
            should_use_semantic(
                state=state,
                lexical_count=20,
                exploration=False,
            )
        )


    def test_sparse_browsing_pool_enables_dense_route(
        self,
    ) -> None:

        state = SessionState(
            user_profile={}
        )

        state.intent = (
            ShoppingIntent.BROWSING
        )

        self.assertTrue(
            should_use_semantic(
                state=state,
                lexical_count=20,
                exploration=False,
            )
        )


    def test_strong_browsing_lexical_pool_preserves_precision_path(
        self,
    ) -> None:

        state = SessionState(
            user_profile={}
        )

        state.intent = (
            ShoppingIntent.BROWSING
        )

        self.assertFalse(
            should_use_semantic(
                state=state,
                lexical_count=100,
                exploration=False,
            )
        )


    def test_exploration_always_enables_dense_route(
        self,
    ) -> None:

        state = SessionState(
            user_profile={}
        )

        state.intent = (
            ShoppingIntent.BUYING
        )

        self.assertTrue(
            should_use_semantic(
                state=state,
                lexical_count=100,
                exploration=True,
            )
        )


if __name__ == "__main__":
    unittest.main()