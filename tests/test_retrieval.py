from __future__ import annotations

import unittest

from src.retrieval import (
    build_expression,
)

from src.state import (
    Evidence,
    SessionState,
)


class RetrievalExpressionTest(
    unittest.TestCase
):

    def test_category_and_evidence_are_scoped_to_different_fields(
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

        expression = (
            build_expression(
                state
            )
        )

        self.assertIn(
            "{title categories}",
            expression,
        )

        self.assertIn(
            '"walking"',
            expression,
        )

        self.assertIn(
            '"shoes"',
            expression,
        )

        self.assertIn(
            (
                "{title features "
                "details description store}"
            ),
            expression,
        )

        self.assertIn(
            '"waterproof"',
            expression,
        )

        self.assertIn(
            '"rubber"',
            expression,
        )

        self.assertIn(
            '"sole"',
            expression,
        )

    def test_category_only_query_still_builds_valid_expression(
        self,
    ) -> None:

        state = SessionState(
            user_profile={}
        )

        state.category_text = (
            "Wallets"
        )

        expression = (
            build_expression(
                state
            )
        )

        self.assertIn(
            "{title categories}",
            expression,
        )

        self.assertIn(
            '"wallets"',
            expression,
        )

        self.assertNotIn(
            (
                "{title features "
                "details description store}"
            ),
            expression,
        )


if __name__ == "__main__":
    unittest.main()