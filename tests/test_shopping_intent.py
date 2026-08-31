from __future__ import annotations

import unittest

from src.shopping_intent import compile_shopping_intent
from src.state import SessionState


class ShoppingIntentCompilerTest(unittest.TestCase):

    def compile(self, message: str):
        state = SessionState(user_profile={})
        state.update(message, 1)
        return compile_shopping_intent(
            state=state,
            user_message=message,
            turn=1,
        )

    def values(self, compiled, bucket: str, attribute: str) -> list[str]:
        signals = getattr(compiled, bucket).get(attribute, ())
        return [signal.value for signal in signals]

    def test_explicit_feature_is_preserved(self) -> None:
        compiled = self.compile(
            "I'm looking for boots. A key requirement is: waterproof."
        )
        self.assertEqual(compiled.product_type.value, "boots")
        self.assertIn(
            "waterproof",
            self.values(compiled, "explicit_attributes", "feature"),
        )

    def test_implicit_waterproof_language_is_inferred_not_explicit(self) -> None:
        compiled = self.compile(
            "I'm looking for boots. I want something that keeps my feet dry."
        )
        self.assertNotIn(
            "waterproof",
            self.values(compiled, "explicit_attributes", "feature"),
        )
        self.assertIn(
            "waterproof",
            self.values(compiled, "inferred_attributes", "feature"),
        )

    def test_negated_explicit_feature_is_not_positive(self) -> None:
        compiled = self.compile(
            "I'm looking for jackets. I don't want anything insulated."
        )
        self.assertNotIn(
            "insulated",
            self.values(compiled, "explicit_attributes", "feature"),
        )
        self.assertIn(
            "insulated",
            self.values(compiled, "exclusions", "feature"),
        )

    def test_without_hood_becomes_exclusion(self) -> None:
        compiled = self.compile(
            "I'm looking for jackets without a hood."
        )
        self.assertEqual(compiled.product_type.value, "jackets")
        self.assertIn(
            "hooded",
            self.values(compiled, "exclusions", "feature"),
        )

    def test_unknown_short_exclusion_is_retained(self) -> None:
        compiled = self.compile(
            "I'm looking for boots without laces."
        )
        self.assertIn(
            "laces",
            self.values(compiled, "exclusions", "feature"),
        )

    def test_budget_stays_structured_hard_constraint(self) -> None:
        compiled = self.compile(
            "I'm looking for black boots under $80."
        )
        self.assertEqual(compiled.product_type.value, "boots")
        self.assertIsNotNone(compiled.budget)
        self.assertEqual(compiled.budget.max_price, 80.0)
        self.assertIn(
            "black",
            self.values(compiled, "explicit_attributes", "color"),
        )

    def test_product_type_removes_explicit_modifiers(self) -> None:
        compiled = self.compile(
            "I need black hiking boots under $100."
        )
        self.assertEqual(compiled.product_type.value, "boots")

    def test_running_paraphrase_is_inferred_use_case(self) -> None:
        compiled = self.compile(
            "I'm looking for T-shirts. Something intended for repeated fast-paced exercise."
        )
        self.assertIn(
            "running",
            self.values(compiled, "inferred_attributes", "use_case"),
        )

    def test_explicit_alias_is_canonicalized(self) -> None:
        compiled = self.compile(
            "I'm looking for shoes. A key requirement is: non-slip."
        )
        self.assertIn(
            "non_slip",
            self.values(compiled, "explicit_attributes", "feature"),
        )

    def test_negative_signal_blocks_same_inference(self) -> None:
        compiled = self.compile(
            "I'm looking for jackets. I don't want insulated; I need something that reduces heat loss."
        )
        self.assertIn(
            "insulated",
            self.values(compiled, "exclusions", "feature"),
        )
        self.assertNotIn(
            "insulated",
            self.values(compiled, "inferred_attributes", "feature"),
        )

    def test_browsing_mode_is_preserved(self) -> None:
        compiled = self.compile(
            "I'm looking for jackets, but I'm still exploring."
        )
        self.assertEqual(compiled.mode.value, "browsing")

    def test_compiler_does_not_mutate_slot_history(self) -> None:
        message = "I'm looking for jackets. I don't want anything insulated."
        state = SessionState(user_profile={})
        state.update(message, 1)
        before = [
            (
                item.attribute,
                item.value,
                item.status,
                item.provenance,
            )
            for item in state.slot_state.history
        ]

        compile_shopping_intent(
            state=state,
            user_message=message,
            turn=1,
        )

        after = [
            (
                item.attribute,
                item.value,
                item.status,
                item.provenance,
            )
            for item in state.slot_state.history
        ]
        self.assertEqual(before, after)

    def test_product_type_falls_back_to_category_after_negated_modifiers(self) -> None:
        compiled = self.compile(
            "Boots. I want waterproof but not insulated"
        )
        self.assertIsNotNone(compiled.product_type)
        self.assertEqual(compiled.product_type.value, "boots")
        self.assertIn(
            "waterproof",
            self.values(compiled, "explicit_attributes", "feature"),
        )
        self.assertIn(
            "insulated",
            self.values(compiled, "exclusions", "feature"),
        )

    def test_not_only_is_not_misread_as_exclusion(self) -> None:
        compiled = self.compile(
            "Jackets. I want not only waterproof but also lightweight"
        )
        self.assertEqual(compiled.exclusions, {})
        self.assertEqual(compiled.product_type.value, "jackets")
        self.assertIn(
            "waterproof",
            self.values(compiled, "explicit_attributes", "feature"),
        )
        self.assertIn(
            "lightweight",
            self.values(compiled, "explicit_attributes", "feature"),
        )

    def test_safe_feature_synonyms_are_canonicalized(self) -> None:
        cases = (
            ("Jackets. I do not want insulation", "insulated"),
            ("Shoes. I do not want slip-resistant soles", "non_slip"),
            ("Shoes. I do not want anti-slip soles", "non_slip"),
            ("Jackets. I do not want a pocket", "pockets"),
        )

        for message, expected in cases:
            with self.subTest(message=message):
                compiled = self.compile(message)
                self.assertIn(
                    expected,
                    self.values(compiled, "exclusions", "feature"),
                )


if __name__ == "__main__":
    unittest.main()
