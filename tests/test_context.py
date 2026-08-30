from __future__ import annotations

import unittest

import src.context as context

from src.context import (
    build_distilled_context,
    configure_context_distillation,
    retrieval_active_text,
    retrieval_evidence_text,
    should_distill_context,
)

from src.retrieval import (
    build_expression,
)

from src.state import (
    SessionState,
)


class StructuredContextDistillationTest(
    unittest.TestCase
):

    def setUp(
        self,
    ) -> None:

        self.original_enabled = (
            context
            .STRUCTURED_CONTEXT_DISTILLATION_ENABLED
        )

    def tearDown(
        self,
    ) -> None:

        configure_context_distillation(
            self.original_enabled
        )

    def test_disabled_mode_preserves_v13_evidence(
        self,
    ) -> None:

        configure_context_distillation(
            False
        )

        state = SessionState(
            user_profile={}
        )

        state.update(
            "I'm looking for shirts.",
            1,
        )

        state.record_question(
            "material"
        )

        state.update(
            (
                "For that, what matters is: "
                "cotton."
            ),
            2,
        )

        self.assertEqual(
            retrieval_evidence_text(
                state
            ),
            "cotton",
        )

        self.assertEqual(
            retrieval_active_text(
                state
            ),
            "shirts cotton",
        )

    def test_selective_override_restores_valid_slot(
        self,
    ) -> None:

        state = SessionState(
            user_profile={}
        )

        state.update(
            (
                "I'm looking for shirts. "
                "A key requirement is: "
                "cotton; black."
            ),
            1,
        )

        state.update(
            (
                "Actually, ignore my earlier "
                "preference. What I need is: "
                "blue."
            ),
            2,
        )

        distilled = (
            build_distilled_context(
                state
            )
        )

        self.assertEqual(
            distilled.mode,
            "structured",
        )

        self.assertIn(
            "cotton",
            distilled.active_text.lower(),
        )

        self.assertIn(
            "blue",
            distilled.active_text.lower(),
        )

        self.assertNotIn(
            "black",
            distilled.active_text.lower(),
        )

    def test_unknown_free_text_is_preserved(
        self,
    ) -> None:

        state = SessionState(
            user_profile={}
        )

        state.update(
            "I'm looking for dresses.",
            1,
        )

        state.update(
            (
                "For that, what matters is: "
                "suitable for a formal dinner."
            ),
            2,
        )

        distilled = (
            build_distilled_context(
                state
            )
        )

        self.assertIn(
            "suitable for a formal dinner",
            distilled.active_text.lower(),
        )

    def test_cleared_slot_is_not_restored(
        self,
    ) -> None:

        state = SessionState(
            user_profile={}
        )

        state.update(
            "I'm looking for shirts.",
            1,
        )

        state.record_question(
            "material"
        )

        state.update(
            (
                "For that, what matters is: "
                "cotton."
            ),
            2,
        )

        state.record_question(
            "material"
        )

        state.update(
            (
                "I don't have a preference "
                "for material."
            ),
            3,
        )

        distilled = (
            build_distilled_context(
                state
            )
        )

        self.assertEqual(
            distilled.mode,
            "structured",
        )

        self.assertNotIn(
            "cotton",
            distilled.active_text.lower(),
        )

    def test_multiple_active_features_are_retained(
        self,
    ) -> None:

        state = SessionState(
            user_profile={}
        )

        state.update(
            (
                "I'm looking for jackets. "
                "A key requirement is: "
                "waterproof and lightweight."
            ),
            1,
        )

        distilled = (
            build_distilled_context(
                state
            )
        )

        # No lifecycle mutation occurred, so this
        # deliberately stays on V13.
        self.assertEqual(
            distilled.mode,
            "legacy_fallback",
        )

        lowered = (
            distilled
            .active_text
            .lower()
        )

        self.assertIn(
            "waterproof",
            lowered,
        )

        self.assertIn(
            "lightweight",
            lowered,
        )

    def test_budget_is_not_injected_into_text_retrieval(
        self,
    ) -> None:

        state = SessionState(
            user_profile={}
        )

        state.update(
            (
                "I'm looking for boots. "
                "My budget is under $80."
            ),
            1,
        )

        distilled = (
            build_distilled_context(
                state
            )
        )

        self.assertNotIn(
            "up to $80",
            distilled.active_values,
        )

    def test_weak_inference_is_not_injected(
        self,
    ) -> None:

        state = SessionState(
            user_profile={}
        )

        state.update(
            "I'm looking for jackets.",
            1,
        )

        state.slot_state.set_slot(
            attribute="feature",
            value="warm",
            source_turn=1,
            intent_epoch=0,
            confidence=0.5,
            strength="soft",
            provenance="weak_inference",
            mode="append",
        )

        distilled = (
            build_distilled_context(
                state
            )
        )

        self.assertNotIn(
            "warm",
            distilled.active_values,
        )

    def test_enabled_expression_uses_active_context(
        self,
    ) -> None:

        configure_context_distillation(
            True
        )

        state = SessionState(
            user_profile={}
        )

        state.update(
            (
                "I'm looking for shirts. "
                "A key requirement is: "
                "cotton; black."
            ),
            1,
        )

        state.update(
            (
                "Actually, ignore my earlier "
                "preference. What I need is: "
                "blue."
            ),
            2,
        )

        expression = (
            build_expression(
                state
            )
            .lower()
        )

        self.assertIn(
            '"cotton"',
            expression,
        )

        self.assertIn(
            '"blue"',
            expression,
        )

        self.assertNotIn(
            '"black"',
            expression,
        )

    # ================================================================
    # V14B.1 SAFETY GATING
    # ================================================================

    def test_no_lifecycle_change_stays_on_v13(
        self,
    ) -> None:

        state = SessionState(
            user_profile={}
        )

        state.update(
            (
                "I'm looking for jackets. "
                "A key requirement is: "
                "waterproof and lightweight."
            ),
            1,
        )

        self.assertFalse(
            should_distill_context(
                state
            )
        )

        distilled = (
            build_distilled_context(
                state
            )
        )

        self.assertEqual(
            distilled.mode,
            "legacy_fallback",
        )

        self.assertEqual(
            distilled.fallback_reason,
            "no_structured_lifecycle_change",
        )

        self.assertEqual(
            distilled.active_text,
            state.active_text(),
        )

    def test_unknown_override_falls_back_to_v13(
        self,
    ) -> None:

        state = SessionState(
            user_profile={}
        )

        state.update(
            (
                "I'm looking for shirts. "
                "A key requirement is: "
                "cotton; black."
            ),
            1,
        )

        state.update(
            (
                "Actually, ignore my earlier "
                "preference. What I need is: "
                "charcoal heather."
            ),
            2,
        )

        self.assertFalse(
            should_distill_context(
                state
            )
        )

        distilled = (
            build_distilled_context(
                state
            )
        )

        self.assertEqual(
            distilled.mode,
            "legacy_fallback",
        )

        self.assertEqual(
            distilled.fallback_reason,
            "unresolved_override",
        )

        self.assertEqual(
            distilled.active_text,
            state.active_text(),
        )

        self.assertEqual(
            distilled.active_text,
            "shirts charcoal heather",
        )

        self.assertNotIn(
            "cotton",
            distilled.active_text.lower(),
        )

        self.assertNotIn(
            "black",
            distilled.active_text.lower(),
        )

    def test_known_override_activates_distillation(
        self,
    ) -> None:

        state = SessionState(
            user_profile={}
        )

        state.update(
            (
                "I'm looking for shirts. "
                "A key requirement is: "
                "cotton; black."
            ),
            1,
        )

        state.update(
            (
                "Actually, ignore my earlier "
                "preference. What I need is: "
                "blue."
            ),
            2,
        )

        self.assertTrue(
            should_distill_context(
                state
            )
        )

        distilled = (
            build_distilled_context(
                state
            )
        )

        self.assertEqual(
            distilled.mode,
            "structured",
        )

        self.assertEqual(
            distilled.fallback_reason,
            None,
        )


if __name__ == "__main__":
    unittest.main()