from __future__ import annotations

import unittest

from scripts.concept_robustness_eval import (
    CONCEPTS,
    build_concept_cases,
    category_allowed,
    category_key_and_label,
    query_has_leakage,
    select_cases,
    specific_category_parts,
    summarize_results,
)


def _spec(
    name: str,
):
    return next(
        spec
        for spec
        in CONCEPTS
        if spec.name == name
    )


class ConceptRobustnessTest(
    unittest.TestCase
):

    def test_generic_root_does_not_count_as_footwear(
        self,
    ) -> None:

        product = {
            "categories": [
                "Clothing, Shoes & Jewelry",
                "Men",
            ]
        }

        parts = specific_category_parts(
            product
        )

        self.assertEqual(
            parts,
            [],
        )

        self.assertFalse(
            category_allowed(
                _spec(
                    "non_slip"
                ),
                parts,
            )
        )


    def test_generic_only_category_has_no_benchmark_key(
        self,
    ) -> None:

        product = {
            "categories": [
                "Clothing, Shoes & Jewelry",
                "Women",
            ]
        }

        (
            key,
            label,
        ) = category_key_and_label(
            product
        )

        self.assertEqual(
            key,
            (),
        )

        self.assertEqual(
            label,
            "",
        )


    def test_non_slip_accepts_real_footwear(
        self,
    ) -> None:

        self.assertTrue(
            category_allowed(
                _spec(
                    "non_slip"
                ),
                [
                    "Shoes",
                    "Fashion Sneakers",
                ],
            )
        )


    def test_non_slip_rejects_bra_category(
        self,
    ) -> None:

        self.assertFalse(
            category_allowed(
                _spec(
                    "non_slip"
                ),
                [
                    "Bras",
                    "Everyday Bras",
                ],
            )
        )


    def test_waterproof_rejects_keyrings(
        self,
    ) -> None:

        self.assertFalse(
            category_allowed(
                _spec(
                    "waterproof"
                ),
                [
                    "Accessories",
                    (
                        "Keyrings, Keychains "
                        "& Charms"
                    ),
                ],
            )
        )


    def test_waterproof_accepts_boots(
        self,
    ) -> None:

        self.assertTrue(
            category_allowed(
                _spec(
                    "waterproof"
                ),
                [
                    "Shoes",
                    "Boots",
                ],
            )
        )


    def test_running_rejects_lingerie(
        self,
    ) -> None:

        self.assertFalse(
            category_allowed(
                _spec(
                    "running"
                ),
                [
                    "Lingerie",
                    (
                        "Bustiers & "
                        "Corsets"
                    ),
                ],
            )
        )


    def test_running_rejects_cycling_category(
        self,
    ) -> None:

        self.assertFalse(
            category_allowed(
                _spec(
                    "running"
                ),
                [
                    "Athletic",
                    "Cycling",
                ],
            )
        )


    def test_running_accepts_shorts(
        self,
    ) -> None:

        self.assertTrue(
            category_allowed(
                _spec(
                    "running"
                ),
                [
                    "Active",
                    "Shorts",
                ],
            )
        )


    def test_winter_rejects_flip_flops(
        self,
    ) -> None:

        self.assertFalse(
            category_allowed(
                _spec(
                    "winter"
                ),
                [
                    "Sandals",
                    "Flip-Flops",
                ],
            )
        )


    def test_winter_accepts_cold_weather_gloves(
        self,
    ) -> None:

        self.assertTrue(
            category_allowed(
                _spec(
                    "winter"
                ),
                [
                    "Gloves & Mittens",
                    (
                        "Cold Weather "
                        "Gloves"
                    ),
                ],
            )
        )


    def test_running_category_leakage_is_rejected(
        self,
    ) -> None:

        spec = _spec(
            "running"
        )

        self.assertTrue(
            query_has_leakage(
                spec,
                (
                    "Running Road Running; "
                    "something suitable "
                    "for jogging"
                ),
            )
        )

        self.assertFalse(
            query_has_leakage(
                spec,
                (
                    "Shoes Athletic; "
                    "something suitable "
                    "for jogging"
                ),
            )
        )


    def test_group_relevance_uses_all_matching_products(
        self,
    ) -> None:

        products = [
            {
                "parent_asin":
                    "A",

                "title":
                    "Waterproof trail boot",

                "features": [
                    (
                        "Waterproof "
                        "membrane"
                    )
                ],

                "categories": [
                    "Clothing, Shoes & Jewelry",
                    "Men",
                    "Shoes",
                    "Boots",
                ],
            },

            {
                "parent_asin":
                    "B",

                "title":
                    "Rain boot",

                "features": [
                    "Waterproof upper"
                ],

                "categories": [
                    "Clothing, Shoes & Jewelry",
                    "Men",
                    "Shoes",
                    "Boots",
                ],
            },

            {
                "parent_asin":
                    "C",

                "title":
                    "Leather boot",

                "features": [
                    "Leather upper"
                ],

                "categories": [
                    "Clothing, Shoes & Jewelry",
                    "Men",
                    "Shoes",
                    "Boots",
                ],
            },

            {
                "parent_asin":
                    "D",

                "title":
                    "Suede boot",

                "features": [
                    "Suede upper"
                ],

                "categories": [
                    "Clothing, Shoes & Jewelry",
                    "Men",
                    "Shoes",
                    "Boots",
                ],
            },

            {
                "parent_asin":
                    "E",

                "title":
                    "Work boot",

                "features": [
                    "Steel toe"
                ],

                "categories": [
                    "Clothing, Shoes & Jewelry",
                    "Men",
                    "Shoes",
                    "Boots",
                ],
            },
        ]

        cases = build_concept_cases(
            products,

            specs=(
                _spec(
                    "waterproof"
                ),
            ),

            min_positives=2,

            min_category_size=5,

            min_negatives=3,
        )

        self.assertEqual(
            len(
                cases
            ),
            3,
        )

        for case in cases:

            self.assertEqual(
                set(
                    case[
                        "relevant_asins"
                    ]
                ),
                {
                    "A",
                    "B",
                },
            )


    def test_generated_cases_have_multiple_paraphrases_without_leakage(
        self,
    ) -> None:

        products = [
            {
                "parent_asin":
                    "A",

                "title":
                    "Waterproof boot",

                "features": [
                    "Waterproof"
                ],

                "categories": [
                    "Shoes",
                    "Boots",
                ],
            },

            {
                "parent_asin":
                    "B",

                "title":
                    (
                        "Waterproof "
                        "work boot"
                    ),

                "features": [
                    "Waterproof"
                ],

                "categories": [
                    "Shoes",
                    "Boots",
                ],
            },

            {
                "parent_asin":
                    "C",

                "title":
                    "Leather boot",

                "features": [
                    "Leather"
                ],

                "categories": [
                    "Shoes",
                    "Boots",
                ],
            },

            {
                "parent_asin":
                    "D",

                "title":
                    "Suede boot",

                "features": [
                    "Suede"
                ],

                "categories": [
                    "Shoes",
                    "Boots",
                ],
            },

            {
                "parent_asin":
                    "E",

                "title":
                    "Work boot",

                "features": [
                    "Steel toe"
                ],

                "categories": [
                    "Shoes",
                    "Boots",
                ],
            },
        ]

        spec = _spec(
            "waterproof"
        )

        cases = build_concept_cases(
            products,

            specs=(
                spec,
            ),

            min_positives=2,

            min_category_size=5,

            min_negatives=3,
        )

        self.assertEqual(
            len(
                cases
            ),
            3,
        )

        self.assertEqual(
            len(
                {
                    case[
                        "paraphrase"
                    ]
                    for case
                    in cases
                }
            ),
            3,
        )

        for case in cases:

            self.assertFalse(
                query_has_leakage(
                    spec,
                    case[
                        "query"
                    ],
                )
            )


    def test_selection_is_deterministic_and_balanced(
        self,
    ) -> None:

        cases = [
            {
                "case_id":
                    f"a{i}",

                "concept":
                    "a",

                "category_key": [
                    "x",
                    str(
                        i
                    ),
                ],

                "paraphrase_index":
                    1,
            }

            for i in range(
                4
            )
        ] + [
            {
                "case_id":
                    f"b{i}",

                "concept":
                    "b",

                "category_key": [
                    "y",
                    str(
                        i
                    ),
                ],

                "paraphrase_index":
                    1,
            }

            for i in range(
                4
            )
        ]

        first = select_cases(
            cases,
            max_cases=6,
            seed=2026,
        )

        second = select_cases(
            cases,
            max_cases=6,
            seed=2026,
        )

        self.assertEqual(
            first,
            second,
        )

        self.assertEqual(
            sum(
                case[
                    "concept"
                ]
                == "a"
                for case
                in first
            ),
            3,
        )

        self.assertEqual(
            sum(
                case[
                    "concept"
                ]
                == "b"
                for case
                in first
            ),
            3,
        )


    def test_summary_measures_semantic_rescue(
        self,
    ) -> None:

        results = [
            {
                "concept":
                    "waterproof",

                "lexical": {
                    "first_relevant_rank":
                        None,

                    "hit_at_10":
                        False,

                    "hit_at_50":
                        False,

                    "hit_at_100":
                        False,

                    "precision_at_10":
                        0.0,

                    "precision_at_50":
                        0.0,

                    "precision_at_100":
                        0.0,

                    "recall_at_10":
                        0.0,

                    "recall_at_50":
                        0.0,

                    "recall_at_100":
                        0.0,
                },

                "semantic": {
                    "first_relevant_rank":
                        5,

                    "hit_at_10":
                        True,

                    "hit_at_50":
                        True,

                    "hit_at_100":
                        True,

                    "precision_at_10":
                        0.1,

                    "precision_at_50":
                        0.02,

                    "precision_at_100":
                        0.01,

                    "recall_at_10":
                        0.5,

                    "recall_at_50":
                        0.5,

                    "recall_at_100":
                        0.5,
                },

                "hybrid": {
                    "hit_at_10":
                        True,

                    "hit_at_50":
                        True,

                    "hit_at_100":
                        True,

                    "candidate_recall_at_10":
                        0.5,

                    "candidate_recall_at_50":
                        0.5,

                    "candidate_recall_at_100":
                        0.5,

                    "candidate_count_at_10":
                        20,

                    "candidate_count_at_50":
                        100,

                    "candidate_count_at_100":
                        200,
                },
            }
        ]

        summary = summarize_results(
            results,
            depth=100,
        )

        self.assertEqual(
            summary[
                "complementarity"
            ][
                "semantic_rescue_cases"
            ],
            1,
        )

        self.assertEqual(
            summary[
                "lexical"
            ][
                "hit_rate_at_100"
            ],
            0.0,
        )

        self.assertEqual(
            summary[
                "semantic"
            ][
                "hit_rate_at_100"
            ],
            1.0,
        )


if __name__ == "__main__":
    unittest.main()