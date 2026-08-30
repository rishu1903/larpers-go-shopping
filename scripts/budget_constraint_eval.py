"""Parser-focused evaluation suite for the budget hard-constraint parser.

Complements the end-to-end evaluators (``evaluator/local_evaluator.py``,
``scripts/concept_robustness_eval.py``) with slot-level metrics for
``src.hard_constraints.parse_budget_constraint``: precision/recall/F1 for
constraint detection, negation accuracy, numeric accuracy, and multi-turn
state accuracy.

``data/public_set.jsonl`` contains no free-text customer messages -- every
budget phrase the parser sees locally is synthesized by
``evaluator/local_evaluator.py``'s own template functions. This suite
deliberately includes cases phrased independently of those templates
(paraphrases, unseen values, negation, explicit removal) so that a parser
change validated only against the public benchmark cannot silently
overfit to the team's own simulator scaffolding.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from src.hard_constraints import (
    BudgetConstraint,
    parse_budget_constraint,
)
from src.state import SessionState


# --------------------------------------------------
# CASE DEFINITION
# --------------------------------------------------


@dataclass(frozen=True)
class BudgetCase:
    """One evaluation case.

    Single-turn cases set ``text`` and are run directly through
    ``parse_budget_constraint``. Multi-turn cases set ``turns`` and are
    run through a fresh ``SessionState`` to exercise accumulation,
    replacement, removal, and override behaviour.
    """

    name: str
    category: str

    # Single-turn cases.
    text: str | None = None
    expect_constraint: bool = False
    expect_min: float | None = None
    expect_max: float | None = None

    # Multi-turn cases.
    turns: tuple[str, ...] | None = None
    expect_final_none: bool = False
    expect_final_min: float | None = None
    expect_final_max: float | None = None
    expect_final_source_turn: int | None = None


@dataclass
class CaseResult:
    name: str
    category: str
    passed: bool
    detail: str
    ground_truth_positive: bool | None
    predicted_positive: bool | None


# --------------------------------------------------
# CASE TABLE
# --------------------------------------------------
#
# Categories:
#
#   template_replication  evaluator's own phrasings -- regression proof
#   paraphrase             new vocabulary added by this change
#   unseen_numeric          values absent from any existing test/template
#   numeric_edge_case       bare-decimal amounts (new capability)
#   negation                adjacency-negated bounds ("not over $100")
#   negative_control        non-monetary numeric text -- must stay None
#   multi_turn_state        accumulation/replacement/removal/override

CASES: tuple[BudgetCase, ...] = (

    # -- template_replication --------------------------------------

    BudgetCase(
        name="template_approximate_budget",
        category="template_replication",
        text="budget around $80",
        expect_constraint=False,
    ),
    BudgetCase(
        name="template_key_requirement_under",
        category="template_replication",
        text="I'm looking for Shoes. A key requirement is: under $80.",
        expect_constraint=True,
        expect_max=80.0,
    ),
    BudgetCase(
        name="template_what_matters_approximate",
        category="template_replication",
        text="For that, what matters is: budget around $45.99.",
        expect_constraint=False,
    ),
    BudgetCase(
        name="template_override_sentence_budget",
        category="template_replication",
        text=(
            "Actually, ignore my earlier preference. "
            "What I need is: under $60."
        ),
        expect_constraint=True,
        expect_max=60.0,
    ),
    BudgetCase(
        name="template_no_preference_for_budget",
        category="template_replication",
        text="I don't have a preference for budget; please use your judgment.",
        expect_constraint=False,
    ),

    # -- paraphrase (new vocabulary added by this change) ------------

    BudgetCase(
        name="paraphrase_cheaper_than",
        category="paraphrase",
        text="Show me something cheaper than $60.",
        expect_constraint=True,
        expect_max=60.0,
    ),
    BudgetCase(
        name="paraphrase_cap_of",
        category="paraphrase",
        text="I have a cap of $45 on this purchase.",
        expect_constraint=True,
        expect_max=45.0,
    ),
    BudgetCase(
        name="paraphrase_capped_at",
        category="paraphrase",
        text="I'm capped at $90 for this purchase.",
        expect_constraint=True,
        expect_max=90.0,
    ),

    # -- unseen_numeric ------------------------------------------------

    BudgetCase(
        name="unseen_numeric_decimal_under",
        category="unseen_numeric",
        text="Keep it under $73.42.",
        expect_constraint=True,
        expect_max=73.42,
    ),
    BudgetCase(
        name="unseen_numeric_large_comma_budget",
        category="unseen_numeric",
        text="My budget is $2,450.",
        expect_constraint=True,
        expect_max=2450.0,
    ),
    BudgetCase(
        name="unseen_numeric_over",
        category="unseen_numeric",
        text="I'd like something over $215.",
        expect_constraint=True,
        expect_min=215.0,
    ),

    # -- numeric_edge_case (bare-decimal, new capability) ---------------

    BudgetCase(
        name="numeric_edge_bare_decimal_budget",
        category="numeric_edge_case",
        text="Budget: $.75",
        expect_constraint=True,
        expect_max=0.75,
    ),
    BudgetCase(
        name="numeric_edge_bare_decimal_under",
        category="numeric_edge_case",
        text="Keep the price under $.99.",
        expect_constraint=True,
        expect_max=0.99,
    ),

    # -- negation (adjacency-negated bounds) ---------------------------

    BudgetCase(
        name="negation_not_above",
        category="negation",
        text="Not above $50, please.",
        expect_constraint=False,
    ),
    BudgetCase(
        name="negation_not_under",
        category="negation",
        text="Not under $80.",
        expect_constraint=False,
    ),

    # -- negative_control (non-monetary numeric text) -------------------

    BudgetCase(
        name="negative_control_rating_stars",
        category="negative_control",
        text="This blanket is rated up to 5 stars.",
        expect_constraint=False,
    ),
    BudgetCase(
        name="negative_control_can_count",
        category="negative_control",
        text="Holds up to 12 cans of soda.",
        expect_constraint=False,
    ),
    BudgetCase(
        name="negative_control_cable_length",
        category="negative_control",
        text="Extends up to 6 feet of cable.",
        expect_constraint=False,
    ),
    BudgetCase(
        # Known, documented limitation: the money-context gate scans the
        # whole message rather than the text near the numeric match, so
        # an unrelated money-context word ("cost") elsewhere in the
        # message still licenses an unrelated "up to N" phrase to be
        # read as a price bound. Not fixed in this change -- see the
        # final report's "remaining failures" section. Tracked here so
        # the gap stays visible rather than silently unmeasured.
        name="negative_control_unrelated_money_context",
        category="negative_control",
        text="The cost is unclear until checkout, but it holds up to 10 items.",
        expect_constraint=False,
    ),

    # -- multi_turn_state ------------------------------------------------

    BudgetCase(
        name="state_accumulation_then_replace",
        category="multi_turn_state",
        turns=(
            "I'm looking for Shoes. A key requirement is: under $120.",
            "Actually my budget is $80.",
        ),
        expect_final_max=80.0,
        expect_final_source_turn=2,
    ),
    BudgetCase(
        name="state_override_preserves_later_budget",
        category="multi_turn_state",
        turns=(
            "I'm looking for Shoes. A key requirement is: leather.",
            "My budget is $100.",
            (
                "Actually, ignore my earlier preference. "
                "What I need is: waterproof."
            ),
        ),
        expect_final_max=100.0,
        expect_final_source_turn=2,
    ),
    BudgetCase(
        name="state_override_removes_turn_one_budget",
        category="multi_turn_state",
        turns=(
            "I'm looking for Shoes. A key requirement is: under $100.",
            (
                "Actually, ignore my earlier preference. "
                "What I need is: waterproof."
            ),
        ),
        expect_final_none=True,
    ),
    BudgetCase(
        name="state_paraphrased_override_clears_budget",
        category="multi_turn_state",
        turns=(
            "I'm looking for Shoes. A key requirement is: under $100.",
            (
                "On second thought, forget my earlier preference "
                "-- here's what I actually need: waterproof shoes."
            ),
        ),
        expect_final_none=True,
    ),
    BudgetCase(
        name="state_blanket_removal_clears_budget",
        category="multi_turn_state",
        turns=(
            "I'm looking for Shoes. A key requirement is: under $100.",
            "Actually, no budget limit anymore -- just show me good options.",
        ),
        expect_final_none=True,
    ),
    BudgetCase(
        name="state_removal_then_new_budget_reintroduced",
        category="multi_turn_state",
        turns=(
            "I'm looking for Shoes. A key requirement is: under $100.",
            "Doesn't matter on price for now.",
            "Actually my budget is $60.",
        ),
        expect_final_max=60.0,
        expect_final_source_turn=3,
    ),
)


# --------------------------------------------------
# EVALUATION
# --------------------------------------------------


def _matches_expected(
    result: BudgetConstraint | object | None,
    expect_min: float | None,
    expect_max: float | None,
) -> bool:

    return result == BudgetConstraint(
        min_price=expect_min,
        max_price=expect_max,
    )


def evaluate_case(case: BudgetCase) -> CaseResult:

    if case.turns is not None:
        return _evaluate_state_case(case)

    return _evaluate_parser_case(case)


def _evaluate_parser_case(case: BudgetCase) -> CaseResult:

    result = parse_budget_constraint(case.text or "")

    predicted_positive = isinstance(result, BudgetConstraint)

    if case.expect_constraint:
        passed = predicted_positive and _matches_expected(
            result,
            case.expect_min,
            case.expect_max,
        )
    else:
        passed = not predicted_positive

    detail = f"parse_budget_constraint({case.text!r}) -> {result!r}"

    return CaseResult(
        name=case.name,
        category=case.category,
        passed=passed,
        detail=detail,
        ground_truth_positive=case.expect_constraint,
        predicted_positive=predicted_positive,
    )


def _evaluate_state_case(case: BudgetCase) -> CaseResult:

    state = SessionState(user_profile={})

    for turn, message in enumerate(case.turns or (), start=1):
        state.update(message, turn)

    if case.expect_final_none:
        passed = state.budget_constraint is None
    else:
        passed = (
            state.budget_constraint
            == BudgetConstraint(
                min_price=case.expect_final_min,
                max_price=case.expect_final_max,
            )
        )

        if case.expect_final_source_turn is not None:
            passed = (
                passed
                and state.budget_source_turn == case.expect_final_source_turn
            )

    detail = (
        f"final budget_constraint={state.budget_constraint!r} "
        f"source_turn={state.budget_source_turn!r}"
    )

    return CaseResult(
        name=case.name,
        category=case.category,
        passed=passed,
        detail=detail,
        ground_truth_positive=None,
        predicted_positive=None,
    )


def run_all(cases: tuple[BudgetCase, ...] = CASES) -> list[CaseResult]:
    return [evaluate_case(case) for case in cases]


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return round(numerator / denominator, 4)


def summarize_results(results: list[CaseResult]) -> dict:

    by_category: dict[str, list[CaseResult]] = {}
    for result in results:
        by_category.setdefault(result.category, []).append(result)

    category_summary = {
        category: {
            "total": len(items),
            "passed": sum(1 for item in items if item.passed),
            "pass_rate": _rate(
                sum(1 for item in items if item.passed),
                len(items),
            ),
        }
        for category, items in sorted(by_category.items())
    }

    # Detection precision/recall/F1 over single-turn cases only.
    detection_cases = [
        result
        for result in results
        if result.ground_truth_positive is not None
    ]

    true_positive = sum(
        1
        for r in detection_cases
        if r.ground_truth_positive and r.predicted_positive
    )
    false_positive = sum(
        1
        for r in detection_cases
        if not r.ground_truth_positive and r.predicted_positive
    )
    false_negative = sum(
        1
        for r in detection_cases
        if r.ground_truth_positive and not r.predicted_positive
    )
    true_negative = sum(
        1
        for r in detection_cases
        if not r.ground_truth_positive and not r.predicted_positive
    )

    precision = _rate(true_positive, true_positive + false_positive)
    recall = _rate(true_positive, true_positive + false_negative)

    f1 = (
        round(2 * precision * recall / (precision + recall), 4)
        if precision and recall and (precision + recall) > 0
        else None
    )

    negation_cases = [
        result for result in results if result.category == "negation"
    ]
    numeric_cases = [
        result
        for result in results
        if result.category in ("unseen_numeric", "numeric_edge_case")
    ]
    state_cases = [
        result for result in results if result.category == "multi_turn_state"
    ]

    summary = {
        "sample_count": len(results),
        "overall_pass_rate": _rate(
            sum(1 for r in results if r.passed),
            len(results),
        ),
        "by_category": category_summary,
        "detection": {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "true_negative": true_negative,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "false_positive_rate": _rate(
                false_positive,
                false_positive + true_negative,
            ),
            "false_negative_rate": _rate(
                false_negative,
                false_negative + true_positive,
            ),
        },
        "negation_accuracy": _rate(
            sum(1 for r in negation_cases if r.passed),
            len(negation_cases),
        ),
        "numeric_accuracy": _rate(
            sum(1 for r in numeric_cases if r.passed),
            len(numeric_cases),
        ),
        "state_accuracy": _rate(
            sum(1 for r in state_cases if r.passed),
            len(state_cases),
        ),
    }

    return summary


# --------------------------------------------------
# CLI
# --------------------------------------------------


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Parser-focused evaluation suite for the budget "
            "hard-constraint parser."
        )
    )

    parser.add_argument(
        "--output",
        default="experiments/v13_budget_constraint_hardening.json",
    )

    args = parser.parse_args()

    results = run_all()
    summary = summarize_results(results)

    report = {
        "summary": summary,
        "cases": [
            {
                "name": result.name,
                "category": result.category,
                "passed": result.passed,
                "detail": result.detail,
            }
            for result in results
        ],
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print()
    print("Budget constraint parser eval summary")
    print("======================================")
    print(json.dumps(summary, indent=2))
    print()
    print(f"Saved report to: {output}")


if __name__ == "__main__":
    main()
