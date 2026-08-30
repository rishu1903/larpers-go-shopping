from __future__ import annotations

import unittest

import src.orchestration as orchestration

from src.orchestration import (
    configure_failure_orchestration,
    retrieval_plan,
    select_protected_recovery,
    should_use_protected_recovery,
    v12_retrieval_plan,
)

from src.state import (
    SessionState,
)


class FailureAwareOrchestrationTest(
    unittest.TestCase
):

    def setUp(
        self,
    ) -> None:

        self.original_enabled = (
            orchestration
            .FAILURE_AWARE_ORCHESTRATION_ENABLED
        )

        self.original_step = (
            orchestration
            .FAILURE_DEPTH_STEP
        )

        self.original_recovery_slots = (
            orchestration
            .FAILURE_RECOVERY_SLOTS
        )

    def tearDown(
        self,
    ) -> None:

        configure_failure_orchestration(
            enabled=(
                self.original_enabled
            ),
            depth_step=(
                self.original_step
            ),
            recovery_slots=(
                self.original_recovery_slots
            ),
        )

    def test_v12_depths_are_preserved_when_disabled(
        self,
    ) -> None:

        configure_failure_orchestration(
            enabled=False,
            depth_step=1.0,
        )

        state = SessionState(
            user_profile={}
        )

        state.miss_streak = 2

        plan = retrieval_plan(
            state=state,
            exploration=True,
        )

        self.assertEqual(
            plan.lexical_limit,
            500,
        )

        self.assertEqual(
            plan.semantic_limit,
            250,
        )

    def test_v12_plan_is_explicitly_reconstructable(
        self,
    ) -> None:

        precision = (
            v12_retrieval_plan(
                exploration=False
            )
        )

        exploration = (
            v12_retrieval_plan(
                exploration=True
            )
        )

        self.assertEqual(
            (
                precision.lexical_limit,
                precision.semantic_limit,
            ),
            (
                100,
                100,
            ),
        )

        self.assertEqual(
            (
                exploration.lexical_limit,
                exploration.semantic_limit,
            ),
            (
                500,
                250,
            ),
        )

    def test_precision_path_is_never_expanded(
        self,
    ) -> None:

        configure_failure_orchestration(
            enabled=True,
            depth_step=1.0,
        )

        state = SessionState(
            user_profile={}
        )

        state.miss_streak = 2

        plan = retrieval_plan(
            state=state,
            exploration=False,
        )

        self.assertEqual(
            (
                plan.lexical_limit,
                plan.semantic_limit,
            ),
            (
                100,
                100,
            ),
        )

        self.assertEqual(
            plan.strategy,
            "precision",
        )

    def test_failure_depth_expands_only_after_miss_signal(
        self,
    ) -> None:

        configure_failure_orchestration(
            enabled=True,
            depth_step=0.50,
        )

        state = SessionState(
            user_profile={}
        )

        baseline = retrieval_plan(
            state=state,
            exploration=True,
        )

        self.assertEqual(
            (
                baseline.lexical_limit,
                baseline.semantic_limit,
            ),
            (
                500,
                250,
            ),
        )

        state.miss_streak = 1

        recovery = retrieval_plan(
            state=state,
            exploration=True,
        )

        self.assertEqual(
            (
                recovery.lexical_limit,
                recovery.semantic_limit,
            ),
            (
                750,
                375,
            ),
        )

    def test_failure_depth_is_capped(
        self,
    ) -> None:

        configure_failure_orchestration(
            enabled=True,
            depth_step=0.50,
        )

        state = SessionState(
            user_profile={}
        )

        state.miss_streak = 10

        plan = retrieval_plan(
            state=state,
            exploration=True,
        )

        self.assertEqual(
            plan.failure_level,
            2,
        )

        self.assertEqual(
            (
                plan.lexical_limit,
                plan.semantic_limit,
            ),
            (
                1000,
                500,
            ),
        )

    def test_protected_recovery_requires_failure_state(
        self,
    ) -> None:

        configure_failure_orchestration(
            enabled=True,
            depth_step=0.75,
            recovery_slots=1,
        )

        state = SessionState(
            user_profile={}
        )

        self.assertFalse(
            should_use_protected_recovery(
                state=state,
                exploration=True,
            )
        )

        state.miss_streak = 1

        self.assertTrue(
            should_use_protected_recovery(
                state=state,
                exploration=True,
            )
        )

        self.assertFalse(
            should_use_protected_recovery(
                state=state,
                exploration=False,
            )
        )

    def test_protected_recovery_keeps_nine_v12_items_and_one_new_item(
        self,
    ) -> None:

        configure_failure_orchestration(
            enabled=True,
            depth_step=0.75,
            recovery_slots=1,
        )

        baseline = [
            {
                "parent_asin":
                    f"B{index}"
            }

            for index
            in range(
                12
            )
        ]

        expanded = [
            {
                "parent_asin":
                    "X"
            },
            {
                "parent_asin":
                    "B3"
            },
            {
                "parent_asin":
                    "Y"
            },
        ]

        selected = (
            select_protected_recovery(
                baseline_ranked=(
                    baseline
                ),
                expanded_ranked=(
                    expanded
                ),
                top_k=10,
            )
        )

        self.assertEqual(
            [
                item[
                    "parent_asin"
                ]

                for item
                in selected
            ],
            [
                "B0",
                "B1",
                "B2",
                "B3",
                "B4",
                "B5",
                "B6",
                "B7",
                "B8",
                "X",
            ],
        )

    def test_protected_recovery_fills_from_v12_when_no_new_candidate(
        self,
    ) -> None:

        configure_failure_orchestration(
            enabled=True,
            recovery_slots=1,
        )

        baseline = [
            {
                "parent_asin":
                    f"B{index}"
            }

            for index
            in range(
                12
            )
        ]

        expanded = [
            {
                "parent_asin":
                    "B3"
            },
            {
                "parent_asin":
                    "B5"
            },
        ]

        selected = (
            select_protected_recovery(
                baseline_ranked=(
                    baseline
                ),
                expanded_ranked=(
                    expanded
                ),
                top_k=10,
            )
        )

        self.assertEqual(
            [
                item[
                    "parent_asin"
                ]

                for item
                in selected
            ],
            [
                f"B{index}"

                for index
                in range(
                    10
                )
            ],
        )

    def test_failure_reply_records_previous_recommendations(
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

        state.update(
            (
                "Those options are not quite right yet. "
                "Ask me about one specific attribute."
            ),
            4,
        )

        self.assertEqual(
            state.miss_streak,
            1,
        )

        self.assertEqual(
            state.failure_events,
            [
                (
                    0,
                    4,
                )
            ],
        )

        self.assertEqual(
            state.failed_recommendations(),
            {
                "A",
                "B",
            },
        )

        self.assertNotIn(
            "options",
            state.active_text().lower(),
        )

    def test_override_starts_new_failure_epoch(
        self,
    ) -> None:

        state = SessionState(
            user_profile={}
        )

        state.record_recommendations(
            [
                "A",
            ]
        )

        state.update(
            (
                "Those options are not quite right yet. "
                "Ask me about one specific attribute."
            ),
            2,
        )

        state.update(
            (
                "Actually, ignore my earlier preference. "
                "What I need is: leather."
            ),
            3,
        )

        self.assertEqual(
            state.intent_epoch,
            1,
        )

        self.assertEqual(
            state.miss_streak,
            0,
        )

        self.assertEqual(
            state.last_recommendations,
            [],
        )

        self.assertEqual(
            state.failed_recommendations(
                epoch=0
            ),
            {
                "A",
            },
        )

        self.assertEqual(
            state.failed_recommendations(),
            set(),
        )


if __name__ == "__main__":
    unittest.main()