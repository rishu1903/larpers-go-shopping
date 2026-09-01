from __future__ import annotations

import unittest

from src.hard_constraints import (
    REMOVE_BUDGET,
    BudgetConstraint,
    apply_budget_constraint,
    coerce_price,
    parse_budget_constraint,
)

from src.state import (
    SessionState,
)


class HardConstraintTest(
    unittest.TestCase
):

    def test_parses_upper_budget(
        self,
    ) -> None:

        constraint = (
            parse_budget_constraint(
                "I need something under $80."
            )
        )

        self.assertEqual(
            constraint,
            BudgetConstraint(
                max_price=80.0,
            ),
        )


    def test_parses_lower_budget(
        self,
    ) -> None:

        constraint = (
            parse_budget_constraint(
                "I'd like something at least $50."
            )
        )

        self.assertEqual(
            constraint,
            BudgetConstraint(
                min_price=50.0,
            ),
        )


    def test_parses_budget_range(
        self,
    ) -> None:

        constraint = (
            parse_budget_constraint(
                (
                    "My budget is between "
                    "$50 and $100."
                )
            )
        )

        self.assertEqual(
            constraint,
            BudgetConstraint(
                min_price=50.0,
                max_price=100.0,
            ),
        )


    def test_plain_budget_is_maximum_spend(
        self,
    ) -> None:

        constraint = (
            parse_budget_constraint(
                "My budget is $120."
            )
        )

        self.assertEqual(
            constraint,
            BudgetConstraint(
                max_price=120.0,
            ),
        )


    def test_budget_word_allows_number_without_dollar_sign(
        self,
    ) -> None:

        constraint = (
            parse_budget_constraint(
                "My budget is 120."
            )
        )

        self.assertEqual(
            constraint,
            BudgetConstraint(
                max_price=120.0,
            ),
        )


    def test_price_context_allows_number_without_dollar_sign(
        self,
    ) -> None:

        constraint = (
            parse_budget_constraint(
                "Keep the price under 80."
            )
        )

        self.assertEqual(
            constraint,
            BudgetConstraint(
                max_price=80.0,
            ),
        )


    def test_spending_context_allows_number_without_dollar_sign(
        self,
    ) -> None:

        constraint = (
            parse_budget_constraint(
                "I can spend up to 100."
            )
        )

        self.assertEqual(
            constraint,
            BudgetConstraint(
                max_price=100.0,
            ),
        )


    def test_approximate_price_is_not_hard_constraint(
        self,
    ) -> None:

        constraint = (
            parse_budget_constraint(
                (
                    "Something around "
                    "$100 would be nice."
                )
            )
        )

        self.assertIsNone(
            constraint
        )


    def test_non_monetary_up_to_measurement_is_not_budget(
        self,
    ) -> None:

        constraint = (
            parse_budget_constraint(
                (
                    "Gold-tone 18mm stainless "
                    "steel expansion band fits "
                    "up to 8-inch wrist "
                    "circumference."
                )
            )
        )

        self.assertIsNone(
            constraint
        )


    def test_non_monetary_distance_is_not_budget(
        self,
    ) -> None:

        constraint = (
            parse_budget_constraint(
                (
                    "Water resistant up to "
                    "30 metres."
                )
            )
        )

        self.assertIsNone(
            constraint
        )


    def test_plain_numeric_limit_without_money_context_is_not_budget(
        self,
    ) -> None:

        constraint = (
            parse_budget_constraint(
                "Supports up to 10 devices."
            )
        )

        self.assertIsNone(
            constraint
        )


    def test_price_coercion_handles_catalogue_formats(
        self,
    ) -> None:

        self.assertEqual(
            coerce_price(
                39.99
            ),
            39.99,
        )

        self.assertEqual(
            coerce_price(
                "$39.99"
            ),
            39.99,
        )

        self.assertEqual(
            coerce_price(
                "1,299.50"
            ),
            1299.50,
        )

        self.assertIsNone(
            coerce_price(
                None
            )
        )


    def test_hard_filter_preserves_order_and_removes_violations(
        self,
    ) -> None:

        candidates = [
            {
                "parent_asin":
                    "A",

                "price":
                    45.0,
            },
            {
                "parent_asin":
                    "B",

                "price":
                    150.0,
            },
            {
                "parent_asin":
                    "C",

                "price":
                    75.0,
            },
            {
                "parent_asin":
                    "D",

                "price":
                    None,
            },
        ]

        filtered = (
            apply_budget_constraint(
                candidates,
                BudgetConstraint(
                    max_price=80.0,
                ),
            )
        )

        self.assertEqual(
            [
                item[
                    "parent_asin"
                ]

                for item
                in filtered
            ],
            [
                "A",
                "C",
            ],
        )


    def test_new_budget_replaces_old_budget(
        self,
    ) -> None:

        state = SessionState(
            user_profile={}
        )

        state.update(
            (
                "I'm looking for Shoes. "
                "A key requirement is: "
                "under $120."
            ),
            1,
        )

        self.assertEqual(
            state.budget_constraint,
            BudgetConstraint(
                max_price=120.0,
            ),
        )

        state.update(
            (
                "Actually my budget is "
                "$80."
            ),
            2,
        )

        self.assertEqual(
            state.budget_constraint,
            BudgetConstraint(
                max_price=80.0,
            ),
        )

        self.assertEqual(
            state.budget_source_turn,
            2,
        )


    def test_override_removes_stale_turn_one_budget(
        self,
    ) -> None:

        state = SessionState(
            user_profile={}
        )

        state.update(
            (
                "I'm looking for Shoes. "
                "A key requirement is: "
                "under $100."
            ),
            1,
        )

        self.assertIsNotNone(
            state.budget_constraint
        )

        state.update(
            (
                "Actually, ignore my earlier "
                "preference. What I need is: "
                "waterproof."
            ),
            3,
        )

        self.assertIsNone(
            state.budget_constraint
        )


    def test_override_preserves_later_budget(
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
            "My budget is $100.",
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


    def test_explicit_no_budget_phrase_returns_remove_sentinel(
        self,
    ) -> None:

        result = parse_budget_constraint(
            "Actually, no budget limit -- surprise me."
        )

        self.assertIs(
            result,
            REMOVE_BUDGET,
        )


    def test_explicit_removal_clears_existing_budget(
        self,
    ) -> None:

        state = SessionState(
            user_profile={}
        )

        state.update(
            (
                "I'm looking for Shoes. "
                "A key requirement is: "
                "under $100."
            ),
            1,
        )

        self.assertIsNotNone(
            state.budget_constraint
        )

        state.update(
            "Doesn't matter on price for now.",
            2,
        )

        self.assertIsNone(
            state.budget_constraint
        )

        self.assertIsNone(
            state.budget_source_turn
        )


if __name__ == "__main__":
    unittest.main()