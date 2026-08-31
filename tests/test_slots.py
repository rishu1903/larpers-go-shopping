from __future__ import annotations

import unittest

from src.slots import (
    SlotState,
    detect_explicit_slots,
)

from src.state import (
    SessionState,
)


class StructuredSlotStateTest(
    unittest.TestCase
):

    def test_slot_state_supersedes_only_same_attribute(
        self,
    ) -> None:

        slots = SlotState()

        slots.set_slot(
            attribute="material",
            value="cotton",
            source_turn=1,
            intent_epoch=0,
            confidence=1.0,
            strength="soft",
            provenance="question_answer",
        )

        slots.set_slot(
            attribute="color",
            value="black",
            source_turn=2,
            intent_epoch=0,
            confidence=1.0,
            strength="soft",
            provenance="question_answer",
        )

        slots.set_slot(
            attribute="color",
            value="blue",
            source_turn=3,
            intent_epoch=0,
            confidence=1.0,
            strength="soft",
            provenance="override",
        )

        self.assertEqual(
            slots.active_snapshot(),
            {
                "material":
                    "cotton",

                "color":
                    "blue",
            },
        )

        color_history = (
            slots.history_for(
                "color"
            )
        )

        self.assertEqual(
            color_history[
                0
            ].status,
            "superseded",
        )

        self.assertEqual(
            color_history[
                1
            ].status,
            "active",
        )

    def test_slot_state_clear_preserves_other_attributes(
        self,
    ) -> None:

        slots = SlotState()

        slots.set_slot(
            attribute="material",
            value="cotton",
            source_turn=1,
            intent_epoch=0,
            confidence=1.0,
            strength="soft",
            provenance="question_answer",
        )

        slots.set_slot(
            attribute="color",
            value="black",
            source_turn=1,
            intent_epoch=0,
            confidence=1.0,
            strength="soft",
            provenance="question_answer",
        )

        slots.clear_slot(
            attribute="material",
            source_turn=2,
            provenance="no_preference",
        )

        self.assertEqual(
            slots.active_snapshot(),
            {
                "color":
                    "black",
            },
        )

    def test_explicit_slot_confidence_does_not_decay(
        self,
    ) -> None:

        slots = SlotState()

        observation = (
            slots.set_slot(
                attribute="material",
                value="cotton",
                source_turn=1,
                intent_epoch=0,
                confidence=1.0,
                strength="soft",
                provenance="question_answer",
            )
        )

        self.assertEqual(
            slots.effective_confidence(
                observation,
                current_turn=10,
            ),
            1.0,
        )

    def test_weak_inference_confidence_decays(
        self,
    ) -> None:

        slots = SlotState()

        observation = (
            slots.set_slot(
                attribute="feature",
                value="warm",
                source_turn=1,
                intent_epoch=0,
                confidence=1.0,
                strength="soft",
                provenance="weak_inference",
            )
        )

        later = (
            slots.effective_confidence(
                observation,
                current_turn=5,
            )
        )

        self.assertLess(
            later,
            1.0,
        )

        self.assertGreater(
            later,
            0.0,
        )

    def test_question_answer_binds_arbitrary_material_value(
        self,
    ) -> None:

        state = SessionState(
            user_profile={}
        )

        state.update(
            "I'm looking for sweaters.",
            1,
        )

        state.record_question(
            "material"
        )

        state.update(
            (
                "For that, what matters is: "
                "merino blend."
            ),
            2,
        )

        self.assertEqual(
            state.active_slots()[
                "material"
            ],
            "merino blend",
        )

    def test_question_answer_binds_arbitrary_brand_value(
        self,
    ) -> None:

        state = SessionState(
            user_profile={}
        )

        state.update(
            "I'm looking for shoes.",
            1,
        )

        state.record_question(
            "brand"
        )

        state.update(
            (
                "For that, what matters is: "
                "North Ridge."
            ),
            2,
        )

        self.assertEqual(
            state.active_slots()[
                "brand"
            ],
            "North Ridge",
        )

    def test_other_answer_extracts_multiple_safe_slots(
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
            "other"
        )

        state.update(
            (
                "For that, what matters is: "
                "cotton; black."
            ),
            2,
        )

        snapshot = (
            state.active_slots()
        )

        self.assertEqual(
            snapshot[
                "material"
            ],
            "cotton",
        )

        self.assertEqual(
            snapshot[
                "color"
            ],
            "black",
        )

    def test_specific_no_preference_clears_slot(
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

        self.assertNotIn(
            "material",
            state.active_slots(),
        )

    def test_targeted_override_replaces_only_detected_slot(
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
            "color"
        )

        state.update(
            (
                "For that, what matters is: "
                "black."
            ),
            3,
        )

        state.update(
            (
                "Actually, ignore my earlier "
                "preference. What I need is: "
                "blue."
            ),
            4,
        )

        snapshot = (
            state.active_slots()
        )

        self.assertEqual(
            snapshot[
                "material"
            ],
            "cotton",
        )

        self.assertEqual(
            snapshot[
                "color"
            ],
            "blue",
        )

        color_history = (
            state.slot_history(
                "color"
            )
        )

        self.assertEqual(
            color_history[
                0
            ].value,
            "black",
        )

        self.assertEqual(
            color_history[
                0
            ].status,
            "superseded",
        )

        self.assertEqual(
            color_history[
                1
            ].value,
            "blue",
        )

        self.assertEqual(
            color_history[
                1
            ].intent_epoch,
            1,
        )

    def test_unknown_override_does_not_erase_structured_slots(
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
                "merino blend."
            ),
            2,
        )

        state.update(
            (
                "Actually, ignore my earlier "
                "preference. What I need is: "
                "something distinctive."
            ),
            3,
        )

        self.assertEqual(
            state.active_slots()[
                "material"
            ],
            "merino blend",
        )

    def test_budget_constraint_is_mirrored_as_hard_slot(
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

        budget = (
            state.slot_state
            .active_slot(
                "budget"
            )
        )

        self.assertIsNotNone(
            budget
        )

        assert budget is not None

        self.assertEqual(
            budget.value,
            "up to $80",
        )

        self.assertEqual(
            budget.strength,
            "hard",
        )

        self.assertEqual(
            budget.provenance,
            "budget_parser",
        )

    def test_category_is_recorded_as_hard_slot(
        self,
    ) -> None:

        state = SessionState(
            user_profile={}
        )

        state.update(
            (
                "I'm looking for "
                "Hiking Boots."
            ),
            1,
        )

        category = (
            state.slot_state
            .active_slot(
                "category"
            )
        )

        self.assertIsNotNone(
            category
        )

        assert category is not None

        self.assertEqual(
            category.value,
            "Hiking Boots",
        )

        self.assertEqual(
            category.strength,
            "hard",
        )

    def test_slot_observation_does_not_change_active_text(
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
            "color"
        )

        state.update(
            (
                "For that, what matters is: "
                "black."
            ),
            3,
        )

        state.update(
            (
                "Actually, ignore my earlier "
                "preference. What I need is: "
                "blue."
            ),
            4,
        )

        self.assertEqual(
            state.active_text(),
            "shirts cotton black blue",
        )

        self.assertEqual(
            state.active_slots()[
                "color"
            ],
            "blue",
        )

    def test_record_question_none_clears_pending_attribute(
        self,
    ) -> None:

        state = SessionState(
            user_profile={}
        )

        state.record_question(
            "material"
        )

        self.assertEqual(
            state.last_asked_attribute,
            "material",
        )

        state.record_question(
            None
        )

        self.assertIsNone(
            state.last_asked_attribute
        )

    # ================================================================
    # V14A.1 MULTI-VALUE TESTS
    # ================================================================

    def test_detector_retains_multiple_features_from_same_message(
        self,
    ) -> None:

        detected = (
            detect_explicit_slots(
                (
                    "I need something waterproof, "
                    "lightweight and breathable."
                )
            )
        )

        features = [
            item.value.lower()

            for item
            in detected

            if (
                item.attribute
                == "feature"
            )
        ]

        self.assertEqual(
            features,
            [
                "waterproof",
                "lightweight",
                "breathable",
            ],
        )

    def test_detector_recognizes_safe_feature_synonyms(
        self,
    ) -> None:

        detected = detect_explicit_slots(
            (
                "insulation, slip-resistant, anti-slip, "
                "nonslip and pocket"
            )
        )

        features = [
            item.value.casefold()
            for item in detected
            if item.attribute == "feature"
        ]

        self.assertIn("insulation", features)
        self.assertIn("slip-resistant", features)
        self.assertIn("anti-slip", features)
        self.assertIn("nonslip", features)
        self.assertIn("pocket", features)

    def test_additive_feature_does_not_erase_existing_feature(
        self,
    ) -> None:

        state = SessionState(
            user_profile={}
        )

        state.update(
            (
                "I'm looking for jackets. "
                "A key requirement is: "
                "waterproof."
            ),
            1,
        )

        state.record_question(
            "other"
        )

        state.update(
            (
                "For that, what matters is: "
                "lightweight."
            ),
            2,
        )

        values = (
            state.active_slot_values()
        )

        self.assertEqual(
            values[
                "feature"
            ],
            [
                "waterproof",
                "lightweight",
            ],
        )

    def test_multiple_features_can_be_active_from_one_other_reply(
        self,
    ) -> None:

        state = SessionState(
            user_profile={}
        )

        state.update(
            "I'm looking for jackets.",
            1,
        )

        state.record_question(
            "other"
        )

        state.update(
            (
                "For that, what matters is: "
                "waterproof; lightweight."
            ),
            2,
        )

        self.assertEqual(
            state.active_slot_values()[
                "feature"
            ],
            [
                "waterproof",
                "lightweight",
            ],
        )

    def test_feature_override_replaces_all_old_feature_values(
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

        state.update(
            (
                "Actually, ignore my earlier "
                "preference. What I need is: "
                "breathable and hooded."
            ),
            3,
        )

        values = (
            state.active_slot_values()
        )

        self.assertEqual(
            values[
                "feature"
            ],
            [
                "breathable",
                "hooded",
            ],
        )

        self.assertEqual(
            values[
                "material"
            ],
            [
                "cotton",
            ],
        )

    def test_no_preference_clears_all_multi_value_features(
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

        state.record_question(
            "feature"
        )

        state.update(
            (
                "I don't have a preference "
                "for feature."
            ),
            2,
        )

        self.assertNotIn(
            "feature",
            state.active_slot_values(),
        )


if __name__ == "__main__":
    unittest.main()