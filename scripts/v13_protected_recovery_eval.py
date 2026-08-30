from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from scripts.concept_robustness_eval import (
    load_catalog,
)

from scripts.v13_failure_orchestration_eval import (
    add_control_comparison,
    metric_text,
    parse_steps,
    reconstruct_residual_cases,
    run_case,
    summarize_run,
)

from src.orchestration import (
    configure_failure_orchestration,
)

from starter.agent import (
    Agent,
)


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate V13 protected failure "
            "recovery: preserve the V12 "
            "continuation while reserving a "
            "small Top-K slot budget for deeper "
            "recovery candidates."
        )
    )

    parser.add_argument(
        "--catalog",
        default=(
            "data/catalog.jsonl"
        ),
    )

    parser.add_argument(
        "--source",
        default=(
            "experiments/"
            "v10_2_concept_robustness.json"
        ),
    )

    parser.add_argument(
        "--v12",
        default=(
            "experiments/"
            "v12_end_to_end_shadow.json"
        ),
    )

    parser.add_argument(
        "--steps",
        default=(
            "0,0.25,0.5,0.75,1.0"
        ),
    )

    parser.add_argument(
        "--recovery-slots",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--max-turn",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--output",
        default=(
            "experiments/"
            "v13_protected_recovery.json"
        ),
    )

    args = parser.parse_args()

    if (
        args.max_turn < 3
        or
        args.max_turn > 10
    ):
        raise ValueError(
            "--max-turn must be "
            "between 3 and 10"
        )

    if (
        args.recovery_slots < 0
        or
        args.recovery_slots > 10
    ):
        raise ValueError(
            "--recovery-slots must "
            "be between 0 and 10"
        )

    steps = parse_steps(
        args.steps
    )

    catalog_path = Path(
        args.catalog
    )

    source_path = Path(
        args.source
    )

    v12_path = Path(
        args.v12
    )

    with source_path.open(
        encoding="utf-8"
    ) as handle:

        source_report = (
            json.load(
                handle
            )
        )

    with v12_path.open(
        encoding="utf-8"
    ) as handle:

        v12_report = (
            json.load(
                handle
            )
        )

    products = (
        load_catalog(
            catalog_path
        )
    )

    residual_cases = (
        reconstruct_residual_cases(
            catalog_products=(
                products
            ),
            source_report=(
                source_report
            ),
            v12_report=(
                v12_report
            ),
        )
    )

    expected_residual_count = int(
        v12_report[
            "summary"
        ][
            "session_by_turn_2"
        ][
            "misses_after_turn_2"
        ]
    )

    if (
        len(
            residual_cases
        )
        != expected_residual_count
    ):
        raise RuntimeError(
            (
                "V12 residual reconstruction "
                "does not match the stored "
                "miss count"
            )
        )

    if (
        args.limit
        is not None
    ):

        if args.limit <= 0:
            raise ValueError(
                "--limit must be positive"
            )

        residual_cases = (
            residual_cases[
                :args.limit
            ]
        )

    print(
        (
            f"Reconstructed "
            f"{len(residual_cases)} "
            "V12 residual shadow cases."
        )
    )

    print(
        "Building production agent once..."
    )

    agent = Agent(
        catalog_path=(
            catalog_path
        )
    )

    runs: list[
        dict
    ] = []

    try:

        for step in steps:

            configure_failure_orchestration(
                enabled=(
                    step > 0.0
                ),
                depth_step=(
                    step
                ),
                recovery_slots=(
                    args.recovery_slots
                ),
            )

            print()

            print(
                (
                    "Evaluating protected "
                    "recovery depth step "
                    f"{step:g}..."
                )
            )

            started = (
                time.perf_counter()
            )

            results: list[
                dict
            ] = []

            for (
                index,
                case,
            ) in enumerate(
                residual_cases,
                start=1,
            ):

                results.append(
                    run_case(
                        agent=agent,
                        case=case,
                        session_id=(
                            "v13-protected-"
                            f"{step:g}-"
                            f"{index}"
                        ),
                        max_turn=(
                            args.max_turn
                        ),
                    )
                )

                print(
                    (
                        f"  {index}/"
                        f"{len(residual_cases)}"
                    )
                )

            runtime_seconds = (
                time.perf_counter()
                -
                started
            )

            runs.append(
                {
                    "depth_step":
                        step,

                    "enabled":
                        (
                            step > 0.0
                        ),

                    "recovery_slots":
                        args.recovery_slots,

                    "summary":
                        summarize_run(
                            results=(
                                results
                            ),
                            v12_report=(
                                v12_report
                            ),
                            max_turn=(
                                args.max_turn
                            ),
                            expected_residual_count=(
                                expected_residual_count
                            ),
                            runtime_seconds=(
                                runtime_seconds
                            ),
                        ),

                    "cases":
                        results,
                }
            )

    finally:

        configure_failure_orchestration(
            enabled=False,
            depth_step=0.50,
            recovery_slots=1,
        )

    add_control_comparison(
        runs
    )

    report = {
        "benchmark":
            (
                "V13 protected failure-recovery "
                "shadow ablation"
            ),

        "uses_public_labels":
            False,

        "methodology": {
            "starts_from_v12_residual_misses":
                True,

            "v12_turn_1_and_2_exactly_protected":
                True,

            "explicit_failure_message_used":
                True,

            "step_0_is_same_depth_continuation_control":
                True,

            "later_failure_turns_compute_v12_and_expanded_plans":
                True,

            "v12_recommendation_prefix_is_protected":
                True,

            "recovery_candidates_must_be_absent_from_v12_candidate_pool":
                True,

            "recovery_slots":
                args.recovery_slots,

            "max_turn":
                args.max_turn,

            "shadow_technical_score_is_not_official":
                True,
        },

        "config": {
            "catalog":
                str(
                    catalog_path
                ),

            "source":
                str(
                    source_path
                ),

            "v12":
                str(
                    v12_path
                ),

            "steps":
                steps,

            "recovery_slots":
                args.recovery_slots,

            "evaluated_residual_case_count":
                len(
                    residual_cases
                ),

            "expected_residual_case_count":
                expected_residual_count,

            "total_shadow_cases":
                int(
                    v12_report[
                        "summary"
                    ][
                        "sample_count"
                    ]
                ),
        },

        "runs":
            runs,
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
        "V13 protected failure-recovery sweep"
    )

    print(
        "======================================"
    )

    print(
        "step | rescues | residual rate | "
        "cumulative H@10 | first-hit MRR | "
        "shadow MTTC | +rescue | lost | "
        "faster | slower | seconds"
    )

    for run in runs:

        summary = (
            run[
                "summary"
            ]
        )

        comparison = (
            run.get(
                "comparison_to_step_0",
                {},
            )
        )

        full_metrics = (
            summary.get(
                "full_shadow_metrics"
            )
            or {}
        )

        print(
            f"{run['depth_step']:>4g} | "
            f"{summary['rescued_after_turn_2']:>7} | "
            f"{summary['residual_rescue_rate']:.6f} | "
            f"{metric_text(full_metrics.get('cumulative_hit_rate_by_max_turn')):>16} | "
            f"{metric_text(full_metrics.get('first_hit_mrr')):>13} | "
            f"{metric_text(full_metrics.get('mean_first_hit_turn_with_miss_11')):>11} | "
            f"{comparison.get('incremental_rescues', 0):>7} | "
            f"{comparison.get('lost_rescues', 0):>4} | "
            f"{comparison.get('accelerated_rescues', 0):>6} | "
            f"{comparison.get('delayed_rescues', 0):>6} | "
            f"{summary['runtime_seconds']:.3f}"
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