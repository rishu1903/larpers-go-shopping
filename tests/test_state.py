from __future__ import annotations

import unittest

from src.state import SessionState


class SessionStateTest(unittest.TestCase):

    def test_accumulates_positive_evidence(self) -> None:

        state = SessionState(
            user_profile={}
        )

        state.update(
            "I'm looking for Women's Shoes, "
            "but I'm still exploring.",
            1,
        )

        state.update(
            "For that, what matters is: "
            "leather; waterproof.",
            2,
        )

        self.assertIn(
            "Women's Shoes",
            state.active_text(),
        )

        self.assertIn(
            "leather",
            state.active_text(),
        )

        self.assertIn(
            "waterproof",
            state.active_text(),
        )

    def test_override_removes_stale_first_turn_but_keeps_later_evidence(
        self,
    ) -> None:

        state = SessionState(
            user_profile={}
        )

        state.update(
            "I'm looking for running shoes. "
            "black style",
            1,
        )

        state.update(
            "For that, what matters is: waterproof.",
            2,
        )

        state.update(
            "Actually, ignore my earlier preference. "
            "What I need is: casual white sneakers.",
            3,
        )

        active = state.active_text().lower()

        self.assertNotIn(
            "black style",
            active,
        )

        self.assertIn(
            "waterproof",
            active,
        )

        self.assertIn(
            "casual white sneakers",
            active,
        )

    def test_no_preference_is_recorded_but_not_added_to_search_text(
        self,
    ) -> None:

        state = SessionState(
            user_profile={}
        )

        state.update(
            "I'm looking for boots, "
            "but I'm still exploring.",
            1,
        )

        state.update(
            "I don't have a preference for color; "
            "please use your judgment.",
            2,
        )

        self.assertIn(
            "color",
            state.no_preference,
        )

        self.assertNotIn(
            "judgment",
            state.active_text().lower(),
        )


if __name__ == "__main__":
    unittest.main()