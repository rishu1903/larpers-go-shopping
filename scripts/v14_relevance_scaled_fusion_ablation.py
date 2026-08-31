from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.concept_robustness_eval import (
    load_catalog,
)

from scripts.end_to_end_shadow_eval import (
    evaluate_case,
    reconstruct_selected_cases,
    summarize_results,
)

from src.fusion import (
    configure_semantic_exploration_weight,
)

from starter.agent import Agent


def parse_weights(
    raw: str,
) -> list[float]:
    values: list[float] = []

    for part in raw.split(","):
        part = part.strip()

        if not part:
            continue

        weight = float(
            part
        )

        if weight < 0.0:
            raise ValueError(
                "weights must be non-negative"
            )

        if weight not in values:
            values.append(
                weight
            )

    if not values:
        raise ValueError(
            "at least one weight is required"
        )

    return values


def compact_summary(
    weight: float,
    summary: dict,
) -> dict:
    initial = summary[
        "initial"
    ]

    turn2 = summary[
        "exploration_turn_only"
    ]

    session = summary[
        "session_by_turn_2"
    ]

    semantic_subset = summary[
        "v10_2_semantic_rescue_subset"
    ]

    return {
        "weight":
            weight,

        "initial_hit_rate_at_10":
            initial[
                "hit_rate_at_10"
            ],

        "initial_mrr_at_10":
            initial[
                "mrr_at_10"
            ],

        "turn2_hit_rate_at_10":
            turn2[
                "hit_rate_at_10"
            ],

        "cumulative_hit_rate_by_turn_2":
            session[
                "cumulative_hit_rate_by_turn_2"
            ],

        "second_turn_rescues":
            session[
                "second_turn_rescues"
            ],

        "misses_after_turn_2":
            session[
                "misses_after_turn_2"
            ],

        "rescue_rate_among_initial_misses":
            session[
                "rescue_rate_among_initial_misses"
            ],

        "first_hit_mrr":
            session[
                "first_hit_mrr"
            ],

        "macro_recall_across_20":
            session[
                "macro_recall_across_20"
            ],

        "semantic_only_subset_count":
            semantic_subset[
                "sample_count"
            ],

        "semantic_only_rescued_into_top_10":
            semantic_subset[
                "rescued_into_top_10_on_turn_2"
            ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep semantic-only exploration fusion "
            "weights against the V14 relevance-scaled "
            "bonus formula, using the same label-free "
            "shadow benchmark as the V12 ablation."
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
        "--weights",
        default=(
            "0,0.25,0.5,0.75,"
            "1.0,1.25,1.5"
        ),
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
            "v14_relevance_scaled_fusion_ablation.json"
        ),
    )

    args = parser.parse_args()

    weights = parse_weights(
        args.weights
    )

    catalog_path = Path(
        args.catalog
    )

    source_path = Path(
        args.source
    )

    with source_path.open(
        encoding="utf-8"
    ) as handle:
        source_report = json.load(
            handle
        )

    products = load_catalog(
        catalog_path
    )

    cases = reconstruct_selected_cases(
        catalog_products=products,
        source_report=source_report,
    )

    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError(
                "--limit must be positive"
            )

        cases = cases[
            :args.limit
        ]

    print(
        f"Reconstructed {len(cases)} "
        "shadow cases."
    )

    print(
        "Building production agent once..."
    )

    agent = Agent(
        catalog_path=(
            catalog_path
        )
    )

    runs: list[dict] = []

    baseline_initial: tuple[
        float,
        float,
    ] | None = None

    for weight in weights:
        configure_semantic_exploration_weight(
            weight
        )

        print()
        print(
            f"Evaluating semantic weight "
            f"{weight:g} (V14 relevance-scaled)..."
        )

        results: list[
            dict
        ] = []

        for index, case in enumerate(
            cases,
            start=1,
        ):
            results.append(
                evaluate_case(
                    agent=agent,
                    case=case,
                    top_k=10,
                )
            )

            if (
                index % 20 == 0

                or

                index == len(
                    cases
                )
            ):
                print(
                    f"  {index}/"
                    f"{len(cases)}"
                )

        summary = summarize_results(
            results
        )

        compact = compact_summary(
            weight=weight,
            summary=summary,
        )

        current_initial = (
            compact[
                "initial_hit_rate_at_10"
            ],
            compact[
                "initial_mrr_at_10"
            ],
        )

        if baseline_initial is None:
            baseline_initial = (
                current_initial
            )

        elif (
            current_initial
            != baseline_initial
        ):
            raise RuntimeError(
                "the sweep changed the protected "
                "turn-1 ranking path"
            )

        runs.append(
            {
                "weight":
                    weight,

                "summary":
                    summary,

                "compact":
                    compact,
            }
        )

    # Return the in-process feature flag to the
    # production-safe zero value after the sweep.
    configure_semantic_exploration_weight(
        0.0
    )

    report = {
        "benchmark":
            (
                "V14 relevance-scaled semantic "
                "exploration fusion weight ablation"
            ),

        "uses_public_labels":
            False,

        "source_benchmark":
            str(
                source_path
            ),

        "weights":
            weights,

        "case_count":
            len(
                cases
            ),

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
        "V14 relevance-scaled fusion sweep"
    )
    print(
        "=================================="
    )

    print(
        "weight | initial H@10 | "
        "cumulative H@10 | rescues | "
        "misses | semantic-only rescues | "
        "first-hit MRR"
    )

    for run in runs:
        item = run[
            "compact"
        ]

        print(
            f"{item['weight']:>6g} | "
            f"{item['initial_hit_rate_at_10']:.6f} | "
            f"{item['cumulative_hit_rate_by_turn_2']:.6f} | "
            f"{item['second_turn_rescues']:>7} | "
            f"{item['misses_after_turn_2']:>6} | "
            f"{item['semantic_only_rescued_into_top_10']:>21} | "
            f"{item['first_hit_mrr']:.6f}"
        )

    print()
    print(
        f"Saved report to: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()
