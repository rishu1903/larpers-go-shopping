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
    configure_fallback_blend_weights,
)

from starter.agent import Agent


# (bm25_weight, popularity_weight) sweep points.
#
# (1.0, 0.0) reproduces the current V15.1 production
# behaviour exactly (sanity-check control).
#
# (0.0, 1.0) reproduces the pre-V15 popularity-first
# behaviour (reference only -- not expected to win,
# included to bracket the full range).
WEIGHTS = (
    (1.0, 0.0),
    (0.9, 0.1),
    (0.75, 0.25),
    (0.5, 0.5),
    (0.0, 1.0),
)


def summary_row(
    weights: tuple[float, float],
    result: dict,
) -> dict:
    scenarios = result[
        "scenario_metrics"
    ]

    return {
        "bm25_weight":
            weights[0],

        "popularity_weight":
            weights[1],

        "overall_mrr":
            result["mrr"],

        "overall_hit_rate_at_10":
            result["hit_rate_at_10"],

        "recommended_technical_score":
            result["recommended_technical_score"],

        "buying_mrr":
            scenarios["buying"]["mrr"],

        "intent_override_mrr":
            scenarios["intent_override"]["mrr"],

        "browsing_mrr":
            scenarios["browsing"]["mrr"],

        "browsing_mttc":
            scenarios["browsing"]["mttc"],

        "boundary_mttc":
            scenarios["boundary"]["mttc"],

        "intent_override_mttc":
            scenarios["intent_override"]["mttc"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep the zero-evidence fallback "
            "blend weights against the official "
            "evaluator."
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
            "v17_exp2_fallback_blend_ablation.json"
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

    # Isolate Experiment 2 from Experiment 1 --
    # measure the fallback blend on its own against
    # the pure baseline, per "test separately before
    # combining".
    configure_buying_relevance_labels(
        "off"
    )

    runs: list[dict] = []

    for (
        bm25_weight,
        popularity_weight,
    ) in WEIGHTS:

        configure_fallback_blend_weights(
            bm25_weight=bm25_weight,
            popularity_weight=popularity_weight,
        )

        print(
            f"Evaluating "
            f"bm25={bm25_weight:g} "
            f"popularity={popularity_weight:g}..."
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
                "bm25_weight":
                    bm25_weight,

                "popularity_weight":
                    popularity_weight,

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
    configure_fallback_blend_weights(
        bm25_weight=1.0,
        popularity_weight=0.0,
    )

    report = {
        "benchmark":
            (
                "Experiment 2: zero-evidence "
                "fallback blend weight sweep"
            ),

        "weights":
            [
                {
                    "bm25_weight": bm25_weight,
                    "popularity_weight": popularity_weight,
                }

                for (
                    bm25_weight,
                    popularity_weight,
                ) in WEIGHTS
            ],

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
        "alpha | beta | overall MRR | TechScore | "
        "buying MRR | override MRR | browsing MRR | "
        "browsing MTTC | boundary MTTC | override MTTC"
    )

    for run in runs:
        row = summary_row(
            (
                run["bm25_weight"],
                run["popularity_weight"],
            ),
            run["result"],
        )

        print(
            f"{row['bm25_weight']:>5g} | "
            f"{row['popularity_weight']:>4g} | "
            f"{row['overall_mrr']:.6f} | "
            f"{row['recommended_technical_score']:.6f} | "
            f"{row['buying_mrr']:.6f} | "
            f"{row['intent_override_mrr']:.6f} | "
            f"{row['browsing_mrr']:.6f} | "
            f"{row['browsing_mttc']:.6f} | "
            f"{row['boundary_mttc']:.6f} | "
            f"{row['intent_override_mttc']:.6f}"
        )

    print()
    print(
        f"Saved report to: {output_path}"
    )


if __name__ == "__main__":
    main()
