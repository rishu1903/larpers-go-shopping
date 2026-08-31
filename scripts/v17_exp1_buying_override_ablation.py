from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluator.local_evaluator import (
    catalog_index,
    evaluate,
    load_jsonl,
)

from src.reranker import (
    configure_buying_relevance_labels,
)

from starter.agent import Agent


MODES = (
    "off",
    "buying_only",
    "override_only",
    "both",
)


def summary_row(
    mode: str,
    result: dict,
) -> dict:
    scenarios = result[
        "scenario_metrics"
    ]

    return {
        "mode":
            mode,

        "overall_mrr":
            result["mrr"],

        "overall_hit_rate_at_10":
            result["hit_rate_at_10"],

        "recommended_technical_score":
            result["recommended_technical_score"],

        "buying_mrr":
            scenarios["buying"]["mrr"],

        "buying_mttc":
            scenarios["buying"]["mttc"],

        "intent_override_mrr":
            scenarios["intent_override"]["mrr"],

        "intent_override_mttc":
            scenarios["intent_override"]["mttc"],

        "browsing_mrr":
            scenarios["browsing"]["mrr"],

        "boundary_mttc":
            scenarios["boundary"]["mttc"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Ablate the buying/override "
            "relevance-label-stripping modes "
            "against the official evaluator."
        )
    )

    parser.add_argument(
        "--catalog",
        default="data/catalog.jsonl",
    )

    parser.add_argument(
        "--dataset",
        default="data/public_set.jsonl",
    )

    parser.add_argument(
        "--output",
        default=(
            "experiments/"
            "v17_exp1_buying_override_ablation.json"
        ),
    )

    args = parser.parse_args()

    samples = load_jsonl(
        args.dataset
    )

    catalog_ids, categories, products = (
        catalog_index(
            args.catalog
        )
    )

    print(
        "Building production agent once..."
    )

    agent = Agent(
        args.catalog
    )

    runs: list[dict] = []

    for mode in MODES:

        configure_buying_relevance_labels(
            mode
        )

        print(
            f"Evaluating mode={mode}..."
        )

        result = evaluate(
            agent,
            samples,
            catalog_ids,
            categories,
            products,
        )

        runs.append(
            {
                "mode":
                    mode,

                "result":
                    {
                        key: value

                        for key, value
                        in result.items()

                        if key != "sessions"
                    },
            }
        )

    # Return to the production default.
    configure_buying_relevance_labels(
        "both"
    )

    report = {
        "benchmark":
            (
                "Experiment 1: buying/override "
                "relevance-label stripping ablation"
            ),

        "modes":
            list(
                MODES
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
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print(
        "mode           | overall MRR | "
        "TechScore | buying MRR | "
        "override MRR | browsing MRR | "
        "boundary MTTC"
    )

    for run in runs:
        row = summary_row(
            run["mode"],
            run["result"],
        )

        print(
            f"{row['mode']:<14} | "
            f"{row['overall_mrr']:.6f} | "
            f"{row['recommended_technical_score']:.6f} | "
            f"{row['buying_mrr']:.6f} | "
            f"{row['intent_override_mrr']:.6f} | "
            f"{row['browsing_mrr']:.6f} | "
            f"{row['boundary_mttc']:.6f}"
        )

    print()
    print(
        f"Saved report to: {output_path}"
    )


if __name__ == "__main__":
    main()
