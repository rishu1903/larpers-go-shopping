from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

from src.state import (
    SessionState,
)


def case_accumulates_independent_slots() -> dict:

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

    snapshot = (
        state.active_slots()
    )

    passed = (
        snapshot.get(
            "material"
        )
        == "merino blend"

        and

        snapshot.get(
            "color"
        )
        == "black"
    )

    return {
        "passed":
            passed,

        "snapshot":
            snapshot,
    }


def case_selective_override() -> dict:

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

    color_history = [
        {
            "value":
                item.value,

            "status":
                item.status,

            "intent_epoch":
                item.intent_epoch,
        }

        for item
        in state.slot_history(
            "color"
        )
    ]

    passed = (
        snapshot.get(
            "material"
        )
        == "cotton"

        and

        snapshot.get(
            "color"
        )
        == "blue"

        and

        len(
            color_history
        )
        == 2

        and

        color_history[
            0
        ][
            "status"
        ]
        == "superseded"

        and

        color_history[
            1
        ][
            "intent_epoch"
        ]
        == 1
    )

    return {
        "passed":
            passed,

        "snapshot":
            snapshot,

        "color_history":
            color_history,
    }


def case_no_preference_clears_only_one_slot() -> dict:

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

    state.record_question(
        "material"
    )

    state.update(
        (
            "I don't have a preference "
            "for material."
        ),
        4,
    )

    snapshot = (
        state.active_slots()
    )

    passed = (
        "material"
        not in snapshot

        and

        snapshot.get(
            "color"
        )
        == "black"
    )

    return {
        "passed":
            passed,

        "snapshot":
            snapshot,
    }


def case_arbitrary_material_from_question() -> dict:

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
            "alpaca blend."
        ),
        2,
    )

    value = (
        state.active_slots()
        .get(
            "material"
        )
    )

    return {
        "passed":
            (
                value
                == "alpaca blend"
            ),

        "material":
            value,
    }


def case_arbitrary_brand_from_question() -> dict:

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

    value = (
        state.active_slots()
        .get(
            "brand"
        )
    )

    return {
        "passed":
            (
                value
                == "North Ridge"
            ),

        "brand":
            value,
    }


def case_other_detects_multiple_slots() -> dict:

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

    return {
        "passed":
            (
                snapshot.get(
                    "material"
                )
                == "cotton"

                and

                snapshot.get(
                    "color"
                )
                == "black"
            ),

        "snapshot":
            snapshot,
    }


def case_budget_is_structured() -> dict:

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

    passed = (
        budget is not None

        and

        budget.value
        == "up to $80"

        and

        budget.strength
        == "hard"
    )

    return {
        "passed":
            passed,

        "budget":
            (
                None
                if budget is None

                else {
                    "value":
                        budget.value,

                    "strength":
                        budget.strength,

                    "provenance":
                        budget.provenance,
                }
            ),
    }


def case_unknown_override_is_safe() -> dict:

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

    before = (
        state.active_slots()
        .copy()
    )

    state.update(
        (
            "Actually, ignore my earlier "
            "preference. What I need is: "
            "something distinctive."
        ),
        3,
    )

    after = (
        state.active_slots()
        .copy()
    )

    return {
        "passed":
            (
                before.get(
                    "material"
                )
                == "merino blend"

                and

                after.get(
                    "material"
                )
                == "merino blend"
            ),

        "before":
            before,

        "after":
            after,
    }


def case_failure_epoch_does_not_delete_valid_slots() -> dict:

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

    state.record_recommendations(
        [
            "A",
            "B",
        ]
    )

    state.update(
        (
            "Those options are not quite "
            "right yet. Ask me about one "
            "specific attribute."
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

    passed = (
        state.intent_epoch
        == 1

        and

        state.miss_streak
        == 0

        and

        snapshot.get(
            "material"
        )
        == "cotton"

        and

        snapshot.get(
            "color"
        )
        == "blue"
    )

    return {
        "passed":
            passed,

        "intent_epoch":
            state.intent_epoch,

        "miss_streak":
            state.miss_streak,

        "snapshot":
            snapshot,
    }


def case_v13_active_text_is_preserved() -> dict:

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

    active_text = (
        state.active_text()
    )

    snapshot = (
        state.active_slots()
    )

    passed = (
        active_text
        == "shirts cotton black blue"

        and

        snapshot.get(
            "color"
        )
        == "blue"
    )

    return {
        "passed":
            passed,

        "production_active_text":
            active_text,

        "structured_snapshot":
            snapshot,
    }


def case_multiple_features_accumulate() -> dict:

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
            "lightweight; breathable."
        ),
        2,
    )

    values = (
        state.active_slot_values()
    )

    expected = [
        "waterproof",
        "lightweight",
        "breathable",
    ]

    return {
        "passed":
            (
                values.get(
                    "feature"
                )
                == expected
            ),

        "feature_values":
            values.get(
                "feature"
            ),
    }


def case_override_replaces_feature_set_only() -> dict:

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

    return {
        "passed":
            (
                values.get(
                    "feature"
                )
                == [
                    "breathable",
                    "hooded",
                ]

                and

                values.get(
                    "material"
                )
                == [
                    "cotton",
                ]
            ),

        "structured_values":
            values,
    }


def case_clear_removes_all_feature_values() -> dict:

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

    values = (
        state.active_slot_values()
    )

    return {
        "passed":
            (
                "feature"
                not in values
            ),

        "structured_values":
            values,
    }


CASES: tuple[
    tuple[
        str,
        Callable[
            [],
            dict,
        ],
    ],
    ...,
] = (
    (
        "accumulates_independent_slots",
        case_accumulates_independent_slots,
    ),
    (
        "selective_override",
        case_selective_override,
    ),
    (
        "no_preference_clears_only_one_slot",
        case_no_preference_clears_only_one_slot,
    ),
    (
        "arbitrary_material_from_question",
        case_arbitrary_material_from_question,
    ),
    (
        "arbitrary_brand_from_question",
        case_arbitrary_brand_from_question,
    ),
    (
        "other_detects_multiple_slots",
        case_other_detects_multiple_slots,
    ),
    (
        "budget_is_structured",
        case_budget_is_structured,
    ),
    (
        "unknown_override_is_safe",
        case_unknown_override_is_safe,
    ),
    (
        "failure_epoch_does_not_delete_valid_slots",
        case_failure_epoch_does_not_delete_valid_slots,
    ),
    (
        "v13_active_text_is_preserved",
        case_v13_active_text_is_preserved,
    ),
    (
        "multiple_features_accumulate",
        case_multiple_features_accumulate,
    ),
    (
        "override_replaces_feature_set_only",
        case_override_replaces_feature_set_only,
    ),
    (
        "clear_removes_all_feature_values",
        case_clear_removes_all_feature_values,
    ),
)


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate V14 structured "
            "conversation-state lifecycle "
            "without using public labels."
        )
    )

    parser.add_argument(
        "--output",
        default=(
            "experiments/"
            "v14a_context_state.json"
        ),
    )

    args = parser.parse_args()

    results: list[
        dict
    ] = []

    for (
        name,
        evaluator,
    ) in CASES:

        detail = (
            evaluator()
        )

        passed = bool(
            detail.pop(
                "passed"
            )
        )

        results.append(
            {
                "case":
                    name,

                "passed":
                    passed,

                "detail":
                    detail,
            }
        )

        status = (
            "PASS"
            if passed
            else "FAIL"
        )

        print(
            f"{status:4}  {name}"
        )

    passed_count = sum(
        1

        for item
        in results

        if item[
            "passed"
        ]
    )

    total_count = len(
        results
    )

    pass_rate = (
        passed_count
        /
        total_count
    )

    report = {
        "benchmark":
            (
                "V14 structured context "
                "state evaluation"
            ),

        "uses_public_labels":
            False,

        "changes_production_ranking":
            False,

        "summary": {
            "case_count":
                total_count,

            "passed":
                passed_count,

            "failed":
                (
                    total_count
                    -
                    passed_count
                ),

            "pass_rate":
                round(
                    pass_rate,
                    6,
                ),
        },

        "capabilities": {
            "incremental_slot_accumulation":
                True,

            "question_grounded_binding":
                True,

            "multi_value_attributes":
                True,

            "selective_override_observation":
                True,

            "specific_slot_clearing":
                True,

            "intent_epoch_tracking":
                True,

            "confidence_lifecycle":
                True,

            "legacy_retrieval_text_preserved":
                True,
        },

        "cases":
            results,
    }

    output_path = Path(
        args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()

    print(
        "V14 structured context evaluation"
    )

    print(
        "================================="
    )

    print(
        f"Cases:     {total_count}"
    )

    print(
        f"Passed:    {passed_count}"
    )

    print(
        (
            "Pass rate: "
            f"{pass_rate:.2%}"
        )
    )

    print()

    print(
        (
            "Saved report to: "
            f"{output_path}"
        )
    )


if __name__ == "__main__":
    main()