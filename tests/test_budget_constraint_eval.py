from __future__ import annotations

import unittest

from scripts.budget_constraint_eval import CASES, evaluate_case


# Cases that expose real, evidenced gaps in the parser as of the start of
# this branch. Each fix commit removes the entries it resolves; by the
# final commit on this branch this set is empty and
# ``test_all_cases_pass`` below covers the full table unconditionally.
KNOWN_FAILING: frozenset[str] = frozenset(
    {
        "numeric_edge_bare_decimal_budget",
        "numeric_edge_bare_decimal_under",
    }
)

# Cases that are expected to keep failing after every commit on this
# branch: real, evidenced gaps that are explicitly out of scope for this
# change (see each case's inline comment in budget_constraint_eval.py and
# the "remaining failures" section of the final report). Kept here,
# rather than silently dropped, so the gap stays visible and tracked.
ACCEPTED_LIMITATIONS: frozenset[str] = frozenset(
    {
        "negative_control_unrelated_money_context",
    }
)


class BudgetConstraintEvalRegressionTest(unittest.TestCase):

    def test_all_cases_pass(self) -> None:

        expected_to_fail = KNOWN_FAILING | ACCEPTED_LIMITATIONS

        unexpected_failures = []
        unexpected_passes = []

        for case in CASES:

            result = evaluate_case(case)

            if result.passed and case.name in expected_to_fail:
                unexpected_passes.append(case.name)

            if not result.passed and case.name not in expected_to_fail:
                unexpected_failures.append(
                    f"{case.name}: {result.detail}"
                )

        self.assertEqual(
            unexpected_failures,
            [],
            msg="Unexpected failures (not in KNOWN_FAILING/ACCEPTED_LIMITATIONS)",
        )

        self.assertEqual(
            unexpected_passes,
            [],
            msg=(
                "Cases now passing that are still listed in "
                "KNOWN_FAILING/ACCEPTED_LIMITATIONS -- remove them from "
                "the allowlist"
            ),
        )


if __name__ == "__main__":
    unittest.main()
