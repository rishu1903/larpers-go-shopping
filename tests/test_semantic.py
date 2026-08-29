from __future__ import annotations

import unittest

from src.semantic import (
    SemanticRetriever,
)


class SemanticRetrieverTest(
    unittest.TestCase
):

    @classmethod
    def setUpClass(
        cls,
    ) -> None:

        cls.retriever = (
            SemanticRetriever()
        )

    def test_semantic_assets_align_with_catalogue(
        self,
    ) -> None:

        self.assertEqual(
            len(
                self.retriever
                .asins
            ),
            50_000,
        )

        self.assertEqual(
            (
                self.retriever
                .embeddings
                .shape
            ),
            (
                50_000,
                96,
            ),
        )

    def test_search_returns_unique_hits_in_descending_score_order(
        self,
    ) -> None:

        hits = (
            self.retriever
            .search(
                (
                    "waterproof "
                    "winter gloves"
                ),
                top_n=5,
            )
        )

        self.assertEqual(
            len(hits),
            5,
        )

        self.assertEqual(
            len(
                {
                    asin
                    for asin, _
                    in hits
                }
            ),
            5,
        )

        self.assertTrue(
            all(
                score_a
                >= score_b

                for (
                    _,
                    score_a,
                ), (
                    _,
                    score_b,
                )

                in zip(
                    hits,
                    hits[1:],
                )
            )
        )


if __name__ == "__main__":
    unittest.main()