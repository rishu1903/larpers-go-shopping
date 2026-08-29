from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Iterable

from starter.agent import Agent

from scripts.concept_robustness_eval import (
    build_concept_cases,
    load_catalog,
)


# ==================================================
# RECOMMENDATION NORMALIZATION
# ==================================================


def normalize_recommendations(
    value: object,
) -> list[str]:
    """
    Normalize Agent.respond()["recommendations"].

    The competition contract normally represents
    recommendations as parent ASIN strings.

    Dictionary-shaped values are also tolerated
    so this evaluator remains robust if richer
    presentation metadata is added later.
    """

    if not isinstance(
        value,
        list,
    ):
        return []

    result: list[str] = []

    for item in value:

        asin: str | None = None

        if isinstance(
            item,
            str,
        ):

            asin = item.strip()

        elif isinstance(
            item,
            dict,
        ):

            for key in (
                "parent_asin",
                "asin",
                "id",
            ):

                candidate = item.get(
                    key
                )

                if candidate:

                    asin = str(
                        candidate
                    ).strip()

                    break

        if (
            asin
            and asin not in result
        ):

            result.append(
                asin
            )

    return result


# ==================================================
# CASE RECONSTRUCTION
# ==================================================


def reconstruct_selected_cases(
    catalog_products: Iterable[dict],
    source_report: dict,
) -> list[dict]:
    """
    Reconstruct the complete positive sets used
    by V10.2.

    V10.2 stores only a small relevant-ASIN sample
    in its result JSON. The complete deterministic
    positive set is rebuilt from the frozen
    catalogue and joined by case_id.
    """

    config = source_report.get(
        "config",
        {},
    )

    available = build_concept_cases(
        catalog_products,

        min_positives=int(
            config.get(
                "min_positives",
                2,
            )
        ),

        min_category_size=int(
            config.get(
                "min_category_size",
                10,
            )
        ),

        min_negatives=int(
            config.get(
                "min_negatives",
                5,
            )
        ),
    )

    by_id = {
        case[
            "case_id"
        ]: case

        for case
        in available
    }

    reconstructed: list[dict] = []
    missing: list[str] = []

    for result_case in source_report.get(
        "cases",
        [],
    ):

        case_id = result_case.get(
            "case_id"
        )

        full_case = by_id.get(
            case_id
        )

        if full_case is None:

            missing.append(
                str(
                    case_id
                )
            )

            continue

        reconstructed.append(
            {
                **full_case,

                "v10_2_lexical":
                    result_case.get(
                        "lexical",
                        {},
                    ),

                "v10_2_semantic":
                    result_case.get(
                        "semantic",
                        {},
                    ),

                "v10_2_hybrid":
                    result_case.get(
                        "hybrid",
                        {},
                    ),
            }
        )

    if missing:

        raise RuntimeError(
            (
                "Could not reconstruct "
                f"{len(missing)} V10.2 cases. "
                f"Examples: {missing[:5]}"
            )
        )

    return reconstructed


# ==================================================
# RANK METRICS
# ==================================================


def recommendation_metrics(
    recommendations: list[str],
    relevant_asins: set[str],
    top_k: int = 10,
) -> dict:
    """
    Evaluate one recommendation turn against the
    complete concept-positive set.
    """

    ranked = recommendations[
        :top_k
    ]

    relevant_ranks = [
        rank

        for rank, asin
        in enumerate(
            ranked,
            start=1,
        )

        if asin in relevant_asins
    ]

    unique_hits = (
        set(
            ranked
        )
        &
        relevant_asins
    )

    first_rank = (
        relevant_ranks[0]

        if relevant_ranks

        else None
    )

    return {
        "recommendation_count":
            len(
                ranked
            ),

        "hit":
            bool(
                unique_hits
            ),

        "first_relevant_rank":
            first_rank,

        "reciprocal_rank":
            (
                round(
                    1.0
                    / first_rank,
                    6,
                )

                if first_rank
                is not None

                else 0.0
            ),

        "relevant_hits":
            len(
                unique_hits
            ),

        "precision_at_10":
            round(
                len(
                    unique_hits
                )
                / top_k,
                6,
            ),

        "recall_at_10":
            round(
                len(
                    unique_hits
                )
                / len(
                    relevant_asins
                ),
                6,
            ),

        "relevant_ranks":
            relevant_ranks,
    }


def cumulative_session_metrics(
    initial: dict,
    exploration: dict,
    relevant_count: int,
) -> dict:
    """
    Combine both recommendation turns as one
    competition-style session.

    A hit on turn 1 remains a hit even if turn 2
    deliberately shows a different recommendation
    set.

    First-hit reciprocal rank also comes from the
    first successful turn.
    """

    initial_hit = bool(
        initial[
            "hit"
        ]
    )

    exploration_hit = bool(
        exploration[
            "hit"
        ]
    )

    if initial_hit:

        first_hit_turn = 1

        first_relevant_rank = (
            initial[
                "first_relevant_rank"
            ]
        )

        reciprocal_rank = (
            initial[
                "reciprocal_rank"
            ]
        )

    elif exploration_hit:

        first_hit_turn = 2

        first_relevant_rank = (
            exploration[
                "first_relevant_rank"
            ]
        )

        reciprocal_rank = (
            exploration[
                "reciprocal_rank"
            ]
        )

    else:

        first_hit_turn = None
        first_relevant_rank = None
        reciprocal_rank = 0.0

    total_relevant_hits = (
        initial[
            "relevant_hits"
        ]
        +
        exploration[
            "relevant_hits"
        ]
    )

    # The production agent intentionally avoids
    # previously recommended ASINs during
    # exploration, so the two sets are normally
    # disjoint. Cap defensively anyway.
    total_relevant_hits = min(
        total_relevant_hits,
        relevant_count,
    )

    return {
        "hit_by_turn_2":
            (
                initial_hit
                or exploration_hit
            ),

        "first_hit_turn":
            first_hit_turn,

        "first_relevant_rank":
            first_relevant_rank,

        "first_hit_reciprocal_rank":
            reciprocal_rank,

        "relevant_hits_across_20":
            total_relevant_hits,

        "recall_across_20":
            round(
                total_relevant_hits
                / relevant_count,
                6,
            ),
    }


# ==================================================
# END-TO-END AGENT EVALUATION
# ==================================================


def _buying_message(
    case: dict,
) -> str:

    return (
        f"I'm looking for "
        f"{case['category']}. "
        f"A key requirement is: "
        f"{case['paraphrase']}."
    )


def evaluate_case(
    agent: Agent,
    case: dict,
    top_k: int = 10,
) -> dict:
    """
    Exercise the current production agent through
    its public competition API.

    Turn 1:
        buying-style paraphrased query.

    Turn 2:
        no additional preference is available,
        activating the current exploration path.
    """

    session_id = (
        "shadow_"
        + case[
            "case_id"
        ]
    )

    agent.reset(
        session_id=session_id,
        user_profile={},
    )

    first_message = (
        _buying_message(
            case
        )
    )

    first_response = agent.respond(
        session_id=session_id,
        user_message=first_message,
        turn=1,
        top_k=top_k,
    )

    initial_recommendations = (
        normalize_recommendations(
            first_response.get(
                "recommendations",
                [],
            )
        )
    )

    relevant = set(
        case[
            "relevant_asins"
        ]
    )

    initial_metrics = (
        recommendation_metrics(
            initial_recommendations,
            relevant,
            top_k=top_k,
        )
    )

    exploration_message = (
        "I don't have an additional "
        "preference for other."
    )

    exploration_response = agent.respond(
        session_id=session_id,
        user_message=exploration_message,
        turn=2,
        top_k=top_k,
    )

    exploration_recommendations = (
        normalize_recommendations(
            exploration_response.get(
                "recommendations",
                [],
            )
        )
    )

    exploration_metrics = (
        recommendation_metrics(
            exploration_recommendations,
            relevant,
            top_k=top_k,
        )
    )

    session_metrics = (
        cumulative_session_metrics(
            initial=initial_metrics,
            exploration=exploration_metrics,
            relevant_count=len(
                relevant
            ),
        )
    )

    initial_hit = bool(
        initial_metrics[
            "hit"
        ]
    )

    exploration_hit = bool(
        exploration_metrics[
            "hit"
        ]
    )

    if (
        not initial_hit
        and exploration_hit
    ):

        transition = (
            "rescued_after_exploration"
        )

    elif (
        initial_hit
        and not exploration_hit
    ):

        # Not a regression. The target was already
        # exposed on turn 1 and remains a session
        # hit. This label is intentionally neutral.
        transition = (
            "initial_only"
        )

    elif (
        initial_hit
        and exploration_hit
    ):

        transition = (
            "hit_both"
        )

    else:

        transition = (
            "miss_both"
        )

    overlap_count = len(
        set(
            initial_recommendations
        )
        &
        set(
            exploration_recommendations
        )
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

        "buying_message":
            first_message,

        "relevant_count":
            len(
                relevant
            ),

        "v10_2": {
            "lexical_first_relevant_rank":
                case[
                    "v10_2_lexical"
                ].get(
                    "first_relevant_rank"
                ),

            "semantic_first_relevant_rank":
                case[
                    "v10_2_semantic"
                ].get(
                    "first_relevant_rank"
                ),
        },

        "initial": {
            "ask_attribute":
                first_response.get(
                    "ask_attribute"
                ),

            "message":
                first_response.get(
                    "message"
                ),

            "recommendations":
                initial_recommendations,

            **initial_metrics,
        },

        "after_exploration": {
            "ask_attribute":
                exploration_response.get(
                    "ask_attribute"
                ),

            "message":
                exploration_response.get(
                    "message"
                ),

            "recommendations":
                exploration_recommendations,

            **exploration_metrics,
        },

        "session": {
            **session_metrics,

            "recommendation_overlap":
                overlap_count,
        },

        "transition":
            transition,
    }


# ==================================================
# SUMMARY
# ==================================================


def summarize_stage(
    results: list[dict],
    stage: str,
) -> dict:

    if not results:

        return {
            "sample_count": 0
        }

    count = len(
        results
    )

    hit_count = sum(
        1

        for result
        in results

        if result[
            stage
        ][
            "hit"
        ]
    )

    return {
        "sample_count":
            count,

        "hit_rate_at_10":
            round(
                hit_count
                / count,
                6,
            ),

        "mrr_at_10":
            round(
                sum(
                    result[
                        stage
                    ][
                        "reciprocal_rank"
                    ]

                    for result
                    in results
                )
                / count,
                6,
            ),

        "macro_precision_at_10":
            round(
                sum(
                    result[
                        stage
                    ][
                        "precision_at_10"
                    ]

                    for result
                    in results
                )
                / count,
                6,
            ),

        "macro_recall_at_10":
            round(
                sum(
                    result[
                        stage
                    ][
                        "recall_at_10"
                    ]

                    for result
                    in results
                )
                / count,
                6,
            ),

        "mean_recommendation_count":
            round(
                sum(
                    result[
                        stage
                    ][
                        "recommendation_count"
                    ]

                    for result
                    in results
                )
                / count,
                3,
            ),
    }


def summarize_session(
    results: list[dict],
) -> dict:

    if not results:

        return {
            "sample_count": 0
        }

    count = len(
        results
    )

    hits = [
        result

        for result
        in results

        if result[
            "session"
        ][
            "hit_by_turn_2"
        ]
    ]

    misses = (
        count
        - len(
            hits
        )
    )

    first_turn_hits = sum(
        1

        for result
        in results

        if result[
            "session"
        ][
            "first_hit_turn"
        ]
        == 1
    )

    second_turn_rescues = sum(
        1

        for result
        in results

        if result[
            "session"
        ][
            "first_hit_turn"
        ]
        == 2
    )

    initial_misses = (
        count
        - first_turn_hits
    )

    return {
        "sample_count":
            count,

        "cumulative_hit_rate_by_turn_2":
            round(
                len(
                    hits
                )
                / count,
                6,
            ),

        "first_turn_hits":
            first_turn_hits,

        "second_turn_rescues":
            second_turn_rescues,

        "misses_after_turn_2":
            misses,

        "rescue_rate_among_initial_misses":
            round(
                second_turn_rescues
                / initial_misses,
                6,
            )
            if initial_misses
            else 0.0,

        "first_hit_mrr":
            round(
                sum(
                    result[
                        "session"
                    ][
                        "first_hit_reciprocal_rank"
                    ]

                    for result
                    in results
                )
                / count,
                6,
            ),

        "macro_recall_across_20":
            round(
                sum(
                    result[
                        "session"
                    ][
                        "recall_across_20"
                    ]

                    for result
                    in results
                )
                / count,
                6,
            ),

        "mean_recommendation_overlap":
            round(
                sum(
                    result[
                        "session"
                    ][
                        "recommendation_overlap"
                    ]

                    for result
                    in results
                )
                / count,
                3,
            ),
    }


def summarize_results(
    results: list[dict],
) -> dict:

    initial = summarize_stage(
        results,
        "initial",
    )

    exploration = summarize_stage(
        results,
        "after_exploration",
    )

    session = summarize_session(
        results
    )

    transitions = {
        "rescued_after_exploration": 0,
        "initial_only": 0,
        "hit_both": 0,
        "miss_both": 0,
    }

    for result in results:

        transitions[
            result[
                "transition"
            ]
        ] += 1

    by_concept: dict[
        str,
        list[dict],
    ] = defaultdict(
        list
    )

    for result in results:

        by_concept[
            result[
                "concept"
            ]
        ].append(
            result
        )

    concept_summary = {
        concept: {
            "initial":
                summarize_stage(
                    group,
                    "initial",
                ),

            "after_exploration":
                summarize_stage(
                    group,
                    "after_exploration",
                ),

            "session":
                summarize_session(
                    group
                ),
        }

        for concept, group
        in sorted(
            by_concept.items()
        )
    }

    semantic_rescue_cases = [
        result

        for result
        in results

        if (
            result[
                "v10_2"
            ][
                "lexical_first_relevant_rank"
            ]
            is None

            and

            result[
                "v10_2"
            ][
                "semantic_first_relevant_rank"
            ]
            is not None
        )
    ]

    return {
        "sample_count":
            len(
                results
            ),

        "initial":
            initial,

        # Kept as a diagnostic of the new second
        # recommendation set, not as a replacement
        # for session-level success.
        "exploration_turn_only":
            exploration,

        "session_by_turn_2":
            session,

        "transition":
            transitions,

        "v10_2_semantic_rescue_subset": {
            "sample_count":
                len(
                    semantic_rescue_cases
                ),

            "session":
                summarize_session(
                    semantic_rescue_cases
                ),

            "rescued_into_top_10_on_turn_2":
                sum(
                    result[
                        "session"
                    ][
                        "first_hit_turn"
                    ]
                    == 2

                    for result
                    in semantic_rescue_cases
                ),
        },

        "by_concept":
            concept_summary,
    }


# ==================================================
# CLI
# ==================================================


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "End-to-end shadow evaluation of "
            "the production conversational "
            "shopping agent's top-10 session "
            "performance."
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
        "--output",
        default=(
            "experiments/"
            "v11_1_end_to_end_shadow.json"
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    if args.top_k != 10:

        raise ValueError(
            (
                "This benchmark intentionally "
                "evaluates the competition's "
                "top-10 recommendation surface. "
                "--top-k must be 10."
            )
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
        (
            f"Reconstructed "
            f"{len(cases)} "
            "V10.2 cases."
        )
    )

    print(
        "Building production agent..."
    )

    agent = Agent(
        catalog_path=(
            catalog_path
        )
    )

    results: list[dict] = []

    for (
        index,
        case,
    ) in enumerate(
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
            or index == len(
                cases
            )
        ):

            print(
                (
                    f"Evaluated "
                    f"{index}/"
                    f"{len(cases)}"
                )
            )

    summary = summarize_results(
        results
    )

    report = {
        "benchmark":
            (
                "end-to-end production "
                "two-turn shadow evaluation"
            ),

        "uses_public_labels":
            False,

        "source_benchmark":
            str(
                source_path
            ),

        "methodology": {
            "uses_public_agent_api":
                True,

            "initial_turn_is_buying_style":
                True,

            "second_turn_exhausts_other_clarification":
                True,

            "evaluates_final_top_10_each_turn":
                True,

            "session_success_is_cumulative":
                True,

            "first_turn_hit_is_not_invalidated_by_later_diversification":
                True,

            "complete_concept_positive_sets_reconstructed":
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

            "top_k":
                10,

            "case_count":
                len(
                    cases
                ),
        },

        "summary":
            summary,

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
        "V11.1 end-to-end shadow summary"
    )
    print(
        "==============================="
    )

    print(
        json.dumps(
            {
                "sample_count":
                    summary[
                        "sample_count"
                    ],

                "initial":
                    summary[
                        "initial"
                    ],

                "exploration_turn_only":
                    summary[
                        "exploration_turn_only"
                    ],

                "session_by_turn_2":
                    summary[
                        "session_by_turn_2"
                    ],

                "transition":
                    summary[
                        "transition"
                    ],

                "v10_2_semantic_rescue_subset":
                    summary[
                        "v10_2_semantic_rescue_subset"
                    ],
            },
            indent=2,
        )
    )

    print()
    print(
        (
            f"Saved report to: "
            f"{output_path}"
        )
    )


if __name__ == "__main__":
    main()