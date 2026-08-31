from __future__ import annotations

import unittest

from src.query_expansion import expand_query_text


class QueryExpansionTest(unittest.TestCase):

    def test_paraphrase_gains_canonical_waterproof_terms(
        self,
    ) -> None:

        expanded = expand_query_text(
            "something suitable for occasional wet conditions"
        )

        self.assertIn("waterproof", expanded)
        self.assertIn("water resistant", expanded)
        self.assertIn(
            "something suitable for occasional wet conditions",
            expanded,
        )

    def test_original_text_is_never_removed_or_altered(
        self,
    ) -> None:

        original = "resists moisture during everyday use"

        expanded = expand_query_text(original)

        self.assertTrue(expanded.startswith(original))

    def test_already_canonical_text_still_gains_synonyms(
        self,
    ) -> None:

        expanded = expand_query_text(
            "a fully waterproof jacket"
        )

        self.assertIn("water resistant", expanded)
        self.assertIn("weatherproof", expanded)

    def test_unrelated_text_is_returned_unchanged(
        self,
    ) -> None:

        original = "comfortable running shoes with good arch support"

        self.assertEqual(
            expand_query_text(original),
            original,
        )

    def test_expansion_is_idempotent(
        self,
    ) -> None:

        once = expand_query_text(
            "something that resists moisture"
        )

        twice = expand_query_text(once)

        self.assertEqual(once, twice)


if __name__ == "__main__":
    unittest.main()
