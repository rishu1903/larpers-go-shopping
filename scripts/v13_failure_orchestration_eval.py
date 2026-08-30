from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import time

from scripts.concept_robustness_eval import (
    CONCEPTS,
    category_key_and_label,
    load_catalog,
    matches_concept,
)
from scripts.end_to_end_shadow_eval import (
    normalize_recommendations,
)
from src.orchestration import (
    configure_failure_orchestration,
    retrieval_plan,
)
from starter.agent import Agent


FAILURE_MESSAGE = (
    "Those options are not quite right yet. "
    "Ask me about one specific attribute."
)

EXHAUSTION_MESSAGE = (
    "I don't have an additional preference for other."
)

MISS_TURN_SENTINEL = 11


def parse_steps(
    raw: str,
) -> list[float]:

    values: list[float] = []

    for part in raw.split(","):

        part = part.strip()

        if not part:
            continue

        value = float(part)

        if value < 0.0:
            raise ValueError(
                "steps must be non-negative"
            )

        if value not in values:
            values.append(value)

    if not values:
        raise ValueError(
            "at least one step is required"
        )

    return values


def first_relevant_rank(
    recommendations: list[str],
    relevant_asins: set[str],
    top_k: int = 10,
) -> int | None:

    for rank, asin in enumerate(
        recommendations[:top_k],
        start=1,
    ):

        if asin in relevant_asins:
            return rank

    return None


def reconstruct_residual_cases(
    *,
    catalog_products: list[dict],
    source_report: dict,
    v12_report: dict,
) -> list[dict]:
    """
    Rebuild complete positive sets only for
    V12's residual misses.

    V10.2 stores a sample of relevant ASINs for
    each case.

    One sampled positive identifies the exact
    benchmark category key.

    We then collect all matching positives for
    only the residual concept/category pairs in
    one catalogue pass.

    This avoids rebuilding all 3,873 valid V10.2
    concept cases.
    """

    residual_ids = {
        str(
            item[
                "case_id"
            ]
        )

        for item
        in v12_report.get(
            "cases",
            [],
        )

        if (
            item.get(
                "transition"
            )
            == "miss_both"
        )
    }

    source_cases = {
        str(
            item[
                "case_id"
            ]
        ):
        item

        for item
        in source_report.get(
            "cases",
            [],
        )

        if (
            str(
                item.get(
                    "case_id"
                )
            )
            in residual_ids
        )
    }

    if (
        set(
            source_cases
        )
        != residual_ids
    ):

        missing = sorted(
            residual_ids
            -
            set(
                source_cases
            )
        )

        raise RuntimeError(
            (
                "Could not find every V12 residual "
                "case in V10.2. "
                f"Missing examples: {missing[:5]}"
            )
        )

    products_by_asin = {
        str(
            product.get(
                "parent_asin"
            )
        ):
        product

        for product
        in catalog_products

        if product.get(
            "parent_asin"
        )
    }

    specs = {
        spec.name:
        spec

        for spec
        in CONCEPTS
    }

    pair_to_case_ids: dict[
        tuple[
            str,
            tuple[str, ...],
        ],
        list[str],
    ] = defaultdict(
        list
    )

    for (
        case_id,
        source_case,
    ) in source_cases.items():

        concept = str(
            source_case[
                "concept"
            ]
        )

        if concept not in specs:

            raise RuntimeError(
                (
                    "Unknown V10.2 concept: "
                    f"{concept}"
                )
            )

        sample = [
            str(
                value
            )

            for value
            in source_case.get(
                "relevant_asin_sample",
                [],
            )
        ]

        if not sample:

            raise RuntimeError(
                (
                    "No positive sample stored "
                    f"for {case_id}"
                )
            )

        anchor = products_by_asin.get(
            sample[0]
        )

        if anchor is None:

            raise RuntimeError(
                (
                    f"Positive anchor {sample[0]} "
                    "is missing from the catalogue"
                )
            )

        (
            category_key,
            category_label,
        ) = category_key_and_label(
            anchor
        )

        if not category_key:

            raise RuntimeError(
                (
                    "Could not reconstruct "
                    f"category for {case_id}"
                )
            )

        if (
            category_label
            != str(
                source_case[
                    "category"
                ]
            )
        ):

            raise RuntimeError(
                (
                    "Category reconstruction "
                    f"drifted for {case_id}: "
                    f"{category_label!r} != "
                    f"{source_case['category']!r}"
                )
            )

        pair_to_case_ids[
            (
                concept,
                category_key,
            )
        ].append(
            case_id
        )

    keys_to_concepts: dict[
        tuple[
            str,
            ...
        ],
        set[str],
    ] = defaultdict(
        set
    )

    for (
        concept,
        category_key,
    ) in pair_to_case_ids:

        keys_to_concepts[
            category_key
        ].add(
            concept
        )

    positives: dict[
        str,
        set[str],
    ] = {
        case_id:
        set()

        for case_id
        in residual_ids
    }

    for product in catalog_products:

        asin = str(
            product.get(
                "parent_asin"
            )
            or ""
        ).strip()

        if not asin:
            continue

        (
            category_key,
            _,
        ) = category_key_and_label(
            product
        )

        concepts = (
            keys_to_concepts.get(
                category_key
            )
        )

        if not concepts:
            continue

        for concept in concepts:

            if not matches_concept(
                specs[
                    concept
                ],
                product,
            ):
                continue

            for case_id in (
                pair_to_case_ids[
                    (
                        concept,
                        category_key,
                    )
                ]
            ):

                positives[
                    case_id
                ].add(
                    asin
                )

    reconstructed: list[
        dict
    ] = []

    # Preserve V12 order so --limit is
    # deterministic.
    for v12_case in v12_report.get(
        "cases",
        [],
    ):

        if (
            v12_case.get(
                "transition"
            )
            != "miss_both"
        ):
            continue

        case_id = str(
            v12_case[
                "case_id"
            ]
        )

        source_case = (
            source_cases[
                case_id
            ]
        )

        relevant_asins = (
            positives[
                case_id
            ]
        )

        expected_count = int(
            source_case[
                "relevant_count"
            ]
        )

        if (
            len(
                relevant_asins
            )
            != expected_count
        ):

            raise RuntimeError(
                (
                    "Positive-set reconstruction "
                    f"drifted for {case_id}: "
                    f"{len(relevant_asins)} != "
                    f"{expected_count}"
                )
            )

        reconstructed.append(
            {
                "case_id":
                    case_id,

                "concept":
                    str(
                        source_case[
                            "concept"
                        ]
                    ),

                "category":
                    str(
                        source_case[
                            "category"
                        ]
                    ),

                "paraphrase":
                    str(
                        source_case[
                            "paraphrase"
                        ]
                    ),

                "relevant_asins":
                    sorted(
                        relevant_asins
                    ),

                "v12_case":
                    v12_case,
            }
        )

    return reconstructed


def expected_v12_recommendations(
    v12_case: dict,
    turn: int,
) -> list[str]:

    if turn == 1:

        section = (
            v12_case[
                "initial"
            ]
        )

    elif turn == 2:

        section = (
            v12_case[
                "after_exploration"
            ]
        )

    else:

        return []

    return (
        normalize_recommendations(
            section.get(
                "recommendations"
            )
        )[:10]
    )


def run_case(
    *,
    agent: Agent,
    case: dict,
    session_id: str,
    max_turn: int,
) -> dict:

    relevant_asins = set(
        case[
            "relevant_asins"
        ]
    )

    agent.reset(
        session_id,
        {},
    )

    user_message = (
        f"I'm looking for {case['category']}. "
        "A key requirement is: "
        f"{case['paraphrase']}."
    )

    turns: list[
        dict
    ] = []

    first_hit_turn: (
        int
        | None
    ) = None

    first_hit_rank: (
        int
        | None
    ) = None

    for turn in range(
        1,
        max_turn + 1,
    ):

        response = (
            agent.respond(
                session_id=(
                    session_id
                ),
                user_message=(
                    user_message
                ),
                turn=turn,
                top_k=10,
            )
        )

        recommendations = (
            normalize_recommendations(
                response.get(
                    "recommendations"
                )
            )
        )

        # Strong safety invariant:
        #
        # not only must Turn 1/2 still miss,
        # the entire recommendation ordering
        # must remain exactly V12.
        if turn <= 2:

            expected = (
                expected_v12_recommendations(
                    case[
                        "v12_case"
                    ],
                    turn,
                )
            )

            if (
                recommendations[:10]
                != expected
            ):

                raise RuntimeError(
                    (
                        "V13 changed protected "
                        f"V12 Turn-{turn} "
                        "recommendations for "
                        f"{case['case_id']}"
                    )
                )

        rank = (
            first_relevant_rank(
                recommendations,
                relevant_asins,
            )
        )

        # Diagnostic access only.
        #
        # This does not affect Agent behaviour.
        state = (
            agent._sessions[
                session_id
            ]
        )

        plan = (
            retrieval_plan(
                state=state,
                exploration=(
                    state
                    .clarification_exhausted
                ),
            )
        )

        turns.append(
            {
                "turn":
                    turn,

                "user_message":
                    user_message,

                "ask_attribute":
                    response.get(
                        "ask_attribute"
                    ),

                "recommendations":
                    recommendations[:10],

                "first_relevant_rank":
                    rank,

                "miss_streak":
                    state.miss_streak,

                "strategy":
                    plan.strategy,

                "lexical_limit":
                    plan.lexical_limit,

                "semantic_limit":
                    plan.semantic_limit,
            }
        )

        if rank is not None:

            first_hit_turn = (
                turn
            )

            first_hit_rank = (
                rank
            )

            break

        if turn == 1:

            user_message = (
                EXHAUSTION_MESSAGE
            )

        else:

            user_message = (
                FAILURE_MESSAGE
            )

    return {
        "case_id":
            case[
                "case_id"
            ],

        "concept":
            case[
                "concept"
            ],

        "category":
            case[
                "category"
            ],

        "paraphrase":
            case[
                "paraphrase"
            ],

        "relevant_count":
            len(
                relevant_asins
            ),

        "first_hit_turn":
            first_hit_turn,

        "first_hit_rank":
            first_hit_rank,

        "first_hit_reciprocal_rank":
            (
                0.0
                if first_hit_rank
                is None

                else round(
                    1.0
                    / first_hit_rank,
                    6,
                )
            ),

        "turns":
            turns,
    }


def summarize_run(
    *,
    results: list[dict],
    v12_report: dict,
    max_turn: int,
    expected_residual_count: int,
    runtime_seconds: float,
) -> dict:

    rescued = [
        item

        for item
        in results

        if (
            item[
                "first_hit_turn"
            ]
            is not None
        )
    ]

    remaining = [
        item

        for item
        in results

        if (
            item[
                "first_hit_turn"
            ]
            is None
        )
    ]

    summary = {
        "evaluated_residual_case_count":
            len(
                results
            ),

        "expected_residual_case_count":
            expected_residual_count,

        "full_residual_set":
            (
                len(
                    results
                )
                == expected_residual_count
            ),

        "rescued_after_turn_2":
            len(
                rescued
            ),

        "residual_rescue_rate":
            (
                round(
                    len(
                        rescued
                    )
                    / len(
                        results
                    ),
                    6,
                )

                if results

                else 0.0
            ),

        "rescues_by_turn": {
            str(
                turn
            ):
            sum(
                1

                for item
                in results

                if (
                    item[
                        "first_hit_turn"
                    ]
                    == turn
                )
            )

            for turn
            in range(
                3,
                max_turn + 1,
            )
        },

        "remaining_within_evaluated":
            len(
                remaining
            ),

        "mean_rescue_reciprocal_rank":
            (
                round(
                    statistics.fmean(
                        float(
                            item[
                                "first_hit_reciprocal_rank"
                            ]
                        )

                        for item
                        in rescued
                    ),
                    6,
                )

                if rescued

                else 0.0
            ),

        "runtime_seconds":
            round(
                runtime_seconds,
                3,
            ),

        "rescued_case_ids": [
            item[
                "case_id"
            ]

            for item
            in rescued
        ],

        "remaining_case_ids": [
            item[
                "case_id"
            ]

            for item
            in remaining
        ],

        "full_shadow_metrics":
            None,
    }

    # Smoke runs should not pretend to be
    # full 130-case metrics.
    if (
        len(
            results
        )
        != expected_residual_count
    ):

        return summary

    v12_cases = (
        v12_report.get(
            "cases",
            [],
        )
    )

    base_successes = [
        item

        for item
        in v12_cases

        if (
            item.get(
                "transition"
            )
            != "miss_both"
        )
    ]

    total_shadow_cases = int(
        v12_report[
            "summary"
        ][
            "sample_count"
        ]
    )

    first_hit_turns = [
        int(
            item[
                "session"
            ][
                "first_hit_turn"
            ]
        )

        for item
        in base_successes
    ]

    first_hit_turns.extend(
        (
            int(
                item[
                    "first_hit_turn"
                ]
            )

            if (
                item[
                    "first_hit_turn"
                ]
                is not None
            )

            else MISS_TURN_SENTINEL
        )

        for item
        in results
    )

    reciprocal_ranks = [
        float(
            item[
                "session"
            ][
                "first_hit_reciprocal_rank"
            ]
        )

        for item
        in base_successes
    ]

    reciprocal_ranks.extend(
        float(
            item[
                "first_hit_reciprocal_rank"
            ]
        )

        for item
        in results
    )

    cumulative_hits = (
        len(
            base_successes
        )
        +
        len(
            rescued
        )
    )

    hit_rate = (
        cumulative_hits
        /
        total_shadow_cases
    )

    first_hit_mrr = (
        statistics.fmean(
            reciprocal_ranks
        )
    )

    mean_turn = (
        statistics.fmean(
            first_hit_turns
        )
    )

    efficiency = max(
        0.0,
        min(
            1.0,
            (
                11.0
                -
                mean_turn
            )
            / 10.0,
        ),
    )

    technical_score = (
        0.50
        * hit_rate

        +

        0.30
        * first_hit_mrr

        +

        0.20
        * efficiency
    )

    summary[
        "full_shadow_metrics"
    ] = {
        "cumulative_hits_by_max_turn":
            cumulative_hits,

        "cumulative_hit_rate_by_max_turn":
            round(
                hit_rate,
                6,
            ),

        "first_hit_mrr":
            round(
                first_hit_mrr,
                6,
            ),

        "mean_first_hit_turn_with_miss_11":
            round(
                mean_turn,
                6,
            ),

        "efficiency_analogue":
            round(
                efficiency,
                6,
            ),

        "technical_score_analogue":
            round(
                technical_score,
                6,
            ),
    }

    return summary


def add_control_comparison(
    runs: list[dict],
) -> None:

    control = next(
        (
            run

            for run
            in runs

            if (
                float(
                    run[
                        "depth_step"
                    ]
                )
                == 0.0
            )
        ),
        None,
    )

    if control is None:
        return

    control_cases = {
        item[
            "case_id"
        ]:
        item

        for item
        in control[
            "cases"
        ]
    }

    for run in runs:

        incremental: list[
            str
        ] = []

        lost: list[
            str
        ] = []

        accelerated: list[
            str
        ] = []

        delayed: list[
            str
        ] = []

        if run is not control:

            for item in run[
                "cases"
            ]:

                case_id = (
                    item[
                        "case_id"
                    ]
                )

                baseline_turn = (
                    control_cases[
                        case_id
                    ][
                        "first_hit_turn"
                    ]
                )

                candidate_turn = (
                    item[
                        "first_hit_turn"
                    ]
                )

                if (
                    baseline_turn
                    is None

                    and

                    candidate_turn
                    is not None
                ):

                    incremental.append(
                        case_id
                    )

                elif (
                    baseline_turn
                    is not None

                    and

                    candidate_turn
                    is None
                ):

                    lost.append(
                        case_id
                    )

                elif (
                    baseline_turn
                    is not None

                    and

                    candidate_turn
                    is not None
                ):

                    if (
                        candidate_turn
                        < baseline_turn
                    ):

                        accelerated.append(
                            case_id
                        )

                    elif (
                        candidate_turn
                        > baseline_turn
                    ):

                        delayed.append(
                            case_id
                        )

        run[
            "comparison_to_step_0"
        ] = {
            "incremental_rescues":
                len(
                    incremental
                ),

            "lost_rescues":
                len(
                    lost
                ),

            "accelerated_rescues":
                len(
                    accelerated
                ),

            "delayed_rescues":
                len(
                    delayed
                ),

            "incremental_rescue_case_ids":
                incremental,

            "lost_rescue_case_ids":
                lost,

            "accelerated_case_ids":
                accelerated,

            "delayed_case_ids":
                delayed,
        }


def metric_text(
    value: object,
) -> str:

    if value is None:
        return "n/a"

    return (
        f"{float(value):.6f}"
    )


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate V13 failure-aware "
            "retrieval-depth escalation on "
            "V12's residual label-free "
            "shadow misses."
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
            "v13_failure_orchestration.json"
        ),
    )

    args = parser.parse_args()

    if (
        args.max_turn < 3
        or
        args.max_turn > 10
    ):

        raise ValueError(
            (
                "--max-turn must be "
                "between 3 and 10"
            )
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

    if args.limit is not None:

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
        f"Reconstructed "
        f"{len(residual_cases)} "
        "V12 residual shadow cases."
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
            )

            print()

            print(
                (
                    "Evaluating failure "
                    f"depth step {step:g}..."
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
                            f"v13-"
                            f"{step:g}-"
                            f"{index}"
                        ),
                        max_turn=(
                            args.max_turn
                        ),
                    )
                )

                print(
                    f"  {index}/"
                    f"{len(residual_cases)}"
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

        # Never leave the process with
        # experimental production routing enabled.
        configure_failure_orchestration(
            enabled=False,
            depth_step=0.50,
        )

    add_control_comparison(
        runs
    )

    report = {
        "benchmark":
            (
                "V13 failure-aware adaptive "
                "orchestration shadow ablation"
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

            "only_later_exploration_depth_changes":
                True,

            "miss_turn_sentinel_for_efficiency_analogue":
                MISS_TURN_SENTINEL,

            "shadow_technical_score_is_not_official":
                True,

            "max_turn":
                args.max_turn,
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
        "V13 failure-aware "
        "orchestration sweep"
    )

    print(
        "====================================="
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
        f"Saved report to: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()