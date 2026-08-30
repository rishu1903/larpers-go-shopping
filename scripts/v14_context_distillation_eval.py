from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics

from scripts.concept_robustness_eval import (
    category_key_and_label,
    load_catalog,
)

from scripts.end_to_end_shadow_eval import (
    normalize_recommendations,
)

from src.context import (
    build_distilled_context,
    configure_context_distillation,
)

from src.slots import (
    detect_explicit_slots,
)

from starter.agent import (
    Agent,
)


def _text(
    value: object,
) -> str:

    if value is None:
        return ""

    if isinstance(
        value,
        dict,
    ):

        return " ".join(
            f"{key} {item}"

            for (
                key,
                item,
            ) in value.items()
        )

    if isinstance(
        value,
        list,
    ):

        return " ".join(
            str(
                item
            )

            for item
            in value
        )

    return str(
        value
    )


def descriptive_text(
    product: dict,
) -> str:

    return " ".join(
        _text(
            product.get(
                field
            )
        )

        for field
        in (
            "title",
            "features",
            "details",
            "description",
            "store",
        )
    )


def slot_values(
    product: dict,
) -> tuple[
    set[str],
    set[str],
]:

    detected = (
        detect_explicit_slots(
            descriptive_text(
                product
            )
        )
    )

    materials = {
        item.value.lower()

        for item
        in detected

        if (
            item.attribute
            == "material"
        )
    }

    colors = {
        item.value.lower()

        for item
        in detected

        if (
            item.attribute
            == "color"
        )
    }

    return (
        materials,
        colors,
    )


def build_cases(
    products: list[dict],
    max_cases: int,
) -> list[dict]:
    """
    Build catalogue-derived selective-override
    sessions.

    Every relevant product must match:

        category
        retained material
        replacement color

    The stale color is chosen so that it has zero
    overlap with the relevant positive set.
    """

    groups: dict[
        tuple[str, ...],
        dict,
    ] = {}

    for product in products:

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
            category_label,
        ) = category_key_and_label(
            product
        )

        if not category_key:
            continue

        group = (
            groups.setdefault(
                category_key,
                {
                    "label":
                        category_label,

                    "all_asins":
                        set(),

                    "color_asins":
                        defaultdict(
                            set
                        ),

                    "pair_asins":
                        defaultdict(
                            set
                        ),
                },
            )
        )

        group[
            "all_asins"
        ].add(
            asin
        )

        (
            materials,
            colors,
        ) = slot_values(
            product
        )

        for color in colors:

            group[
                "color_asins"
            ][
                color
            ].add(
                asin
            )

        for material in materials:

            for color in colors:

                group[
                    "pair_asins"
                ][
                    (
                        material,
                        color,
                    )
                ].add(
                    asin
                )

    candidates: list[
        dict
    ] = []

    for (
        category_key,
        group,
    ) in groups.items():

        if (
            len(
                group[
                    "all_asins"
                ]
            )
            < 10
        ):
            continue

        for (
            (
                material,
                new_color,
            ),
            relevant_asins,
        ) in (
            group[
                "pair_asins"
            ].items()
        ):

            if (
                len(
                    relevant_asins
                )
                < 2
            ):
                continue

            stale_options: list[
                tuple[
                    int,
                    str,
                ]
            ] = []

            for (
                stale_color,
                stale_asins,
            ) in (
                group[
                    "color_asins"
                ].items()
            ):

                if (
                    stale_color
                    == new_color
                ):
                    continue

                if (
                    len(
                        stale_asins
                    )
                    < 2
                ):
                    continue

                # Require a clean contradiction:
                # none of the defined positives
                # also carry the stale color.
                if (
                    relevant_asins
                    &
                    stale_asins
                ):
                    continue

                stale_options.append(
                    (
                        -len(
                            stale_asins
                        ),
                        stale_color,
                    )
                )

            if not stale_options:
                continue

            stale_options.sort()

            stale_color = (
                stale_options[
                    0
                ][
                    1
                ]
            )

            raw_id = (
                "|".join(
                    (
                        "/".join(
                            category_key
                        ),
                        material,
                        stale_color,
                        new_color,
                    )
                )
            )

            case_id = (
                "v14_"
                +
                hashlib.sha1(
                    raw_id.encode(
                        "utf-8"
                    )
                )
                .hexdigest()[
                    :12
                ]
            )

            candidates.append(
                {
                    "case_id":
                        case_id,

                    "category":
                        group[
                            "label"
                        ],

                    "material":
                        material,

                    "stale_color":
                        stale_color,

                    "new_color":
                        new_color,

                    "relevant_asins":
                        sorted(
                            relevant_asins
                        ),

                    "category_size":
                        len(
                            group[
                                "all_asins"
                            ]
                        ),
                }
            )

    # Deterministic sample independent of catalogue
    # iteration ordering.
    candidates.sort(
        key=lambda item: (
            hashlib.sha1(
                item[
                    "case_id"
                ].encode(
                    "utf-8"
                )
            ).hexdigest()
        )
    )

    return candidates[
        :max_cases
    ]


def first_relevant_rank(
    recommendations: list[str],
    relevant_asins: set[str],
) -> int | None:

    for (
        rank,
        asin,
    ) in enumerate(
        recommendations[
            :10
        ],
        start=1,
    ):

        if (
            asin
            in relevant_asins
        ):

            return rank

    return None


def run_case(
    *,
    agent: Agent,
    case: dict,
    session_id: str,
) -> dict:

    agent.reset(
        session_id,
        {},
    )

    turn_1 = (
        f"I'm looking for "
        f"{case['category']}. "
        "A key requirement is: "
        f"{case['material']}; "
        f"{case['stale_color']}."
    )

    agent.respond(
        session_id=session_id,
        user_message=turn_1,
        turn=1,
        top_k=10,
    )

    turn_2 = (
        "Actually, ignore my earlier "
        "preference. What I need is: "
        f"{case['new_color']}."
    )

    response = (
        agent.respond(
            session_id=session_id,
            user_message=turn_2,
            turn=2,
            top_k=10,
        )
    )

    recommendations = (
        normalize_recommendations(
            response.get(
                "recommendations"
            )
        )[
            :10
        ]
    )

    relevant_asins = set(
        case[
            "relevant_asins"
        ]
    )

    rank = (
        first_relevant_rank(
            recommendations,
            relevant_asins,
        )
    )

    state = (
        agent._sessions[
            session_id
        ]
    )

    distilled = (
        build_distilled_context(
            state
        )
    )

    return {
        "recommendations":
            recommendations,

        "first_relevant_rank":
            rank,

        "hit_at_10":
            (
                rank
                is not None
            ),

        "reciprocal_rank":
            (
                0.0

                if rank
                is None

                else
                1.0
                / rank
            ),

        "legacy_active_text":
            state.active_text(),

        "distilled_active_text":
            distilled.active_text,

        "structured_values":
            state.active_slot_values(),
    }


def summarize(
    results: list[dict],
) -> dict:

    if not results:

        return {
            "sample_count":
                0,

            "hit_rate_at_10":
                0.0,

            "mrr":
                0.0,
        }

    return {
        "sample_count":
            len(
                results
            ),

        "hit_rate_at_10":
            round(
                statistics.fmean(
                    1.0
                    if item[
                        "hit_at_10"
                    ]
                    else 0.0

                    for item
                    in results
                ),
                6,
            ),

        "mrr":
            round(
                statistics.fmean(
                    float(
                        item[
                            "reciprocal_rank"
                        ]
                    )

                    for item
                    in results
                ),
                6,
            ),
    }


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Compare V13 legacy context against "
            "V14 structured context distillation "
            "using catalogue-derived selective "
            "override sessions."
        )
    )

    parser.add_argument(
        "--catalog",
        default=(
            "data/catalog.jsonl"
        ),
    )

    parser.add_argument(
        "--cases",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--output",
        default=(
            "experiments/"
            "v14b_context_distillation.json"
        ),
    )

    args = parser.parse_args()

    if args.cases <= 0:

        raise ValueError(
            "--cases must be positive"
        )

    catalog_path = Path(
        args.catalog
    )

    products = (
        load_catalog(
            catalog_path
        )
    )

    cases = (
        build_cases(
            products,
            max_cases=(
                args.cases
            ),
        )
    )

    if not cases:

        raise RuntimeError(
            (
                "No valid structured-context "
                "cases could be generated."
            )
        )

    print(
        (
            f"Generated "
            f"{len(cases)} "
            "catalogue-derived "
            "selective-override cases."
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

    legacy_results: list[
        dict
    ] = []

    structured_results: list[
        dict
    ] = []

    try:

        configure_context_distillation(
            False
        )

        print()

        print(
            "Evaluating V13 legacy context..."
        )

        for (
            index,
            case,
        ) in enumerate(
            cases,
            start=1,
        ):

            legacy_results.append(
                run_case(
                    agent=agent,
                    case=case,
                    session_id=(
                        f"v14-legacy-{index}"
                    ),
                )
            )

        configure_context_distillation(
            True
        )

        print(
            "Evaluating V14 distilled context..."
        )

        for (
            index,
            case,
        ) in enumerate(
            cases,
            start=1,
        ):

            structured_results.append(
                run_case(
                    agent=agent,
                    case=case,
                    session_id=(
                        f"v14-structured-{index}"
                    ),
                )
            )

    finally:

        configure_context_distillation(
            False
        )

    transitions = {
        "gained":
            0,

        "lost":
            0,

        "hit_both":
            0,

        "miss_both":
            0,
    }

    case_results: list[
        dict
    ] = []

    for (
        case,
        legacy,
        structured,
    ) in zip(
        cases,
        legacy_results,
        structured_results,
    ):

        legacy_hit = bool(
            legacy[
                "hit_at_10"
            ]
        )

        structured_hit = bool(
            structured[
                "hit_at_10"
            ]
        )

        if (
            not legacy_hit

            and

            structured_hit
        ):

            transition = "gained"

        elif (
            legacy_hit

            and

            not structured_hit
        ):

            transition = "lost"

        elif (
            legacy_hit

            and

            structured_hit
        ):

            transition = "hit_both"

        else:

            transition = "miss_both"

        transitions[
            transition
        ] += 1

        case_results.append(
            {
                **case,

                "transition":
                    transition,

                "legacy":
                    legacy,

                "structured":
                    structured,
            }
        )

    legacy_summary = (
        summarize(
            legacy_results
        )
    )

    structured_summary = (
        summarize(
            structured_results
        )
    )

    report = {
        "benchmark":
            (
                "V14B catalogue-derived "
                "structured context distillation"
            ),

        "uses_public_labels":
            False,

        "production_enabled":
            False,

        "methodology": {
            "scenario":
                (
                    "retain one valid material "
                    "while selectively overriding "
                    "a stale color"
                ),

            "relevance":
                (
                    "same-category products whose "
                    "participant-visible metadata "
                    "contains both the retained "
                    "material and replacement color"
                ),

            "stale_color_overlap_with_positive_set":
                0,

            "public_targets_used":
                False,
        },

        "legacy":
            legacy_summary,

        "structured":
            structured_summary,

        "delta": {
            "hit_rate_at_10":
                round(
                    structured_summary[
                        "hit_rate_at_10"
                    ]
                    -
                    legacy_summary[
                        "hit_rate_at_10"
                    ],
                    6,
                ),

            "mrr":
                round(
                    structured_summary[
                        "mrr"
                    ]
                    -
                    legacy_summary[
                        "mrr"
                    ],
                    6,
                ),
        },

        "transitions":
            transitions,

        "cases":
            case_results,
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
        "V14B structured context distillation"
    )

    print(
        "===================================="
    )

    print(
        (
            "Legacy Hit@10:     "
            f"{legacy_summary['hit_rate_at_10']:.6f}"
        )
    )

    print(
        (
            "Structured Hit@10: "
            f"{structured_summary['hit_rate_at_10']:.6f}"
        )
    )

    print(
        (
            "Legacy MRR:        "
            f"{legacy_summary['mrr']:.6f}"
        )
    )

    print(
        (
            "Structured MRR:    "
            f"{structured_summary['mrr']:.6f}"
        )
    )

    print()

    print(
        (
            "Gained:   "
            f"{transitions['gained']}"
        )
    )

    print(
        (
            "Lost:     "
            f"{transitions['lost']}"
        )
    )

    print(
        (
            "Hit both: "
            f"{transitions['hit_both']}"
        )
    )

    print(
        (
            "Miss both:"
            f" {transitions['miss_both']}"
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