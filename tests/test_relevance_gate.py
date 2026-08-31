import unittest

from src.relevance_gate import (
    candidate_violates_observed_exclusion,
    filter_observed_exclusions,
)
from src.shopping_intent import compile_shopping_intent
from src.state import SessionState


def _intent(message: str):
    state = SessionState(user_profile={})
    state.update(user_message=message, turn=1)
    return compile_shopping_intent(
        state=state,
        user_message=message,
        turn=1,
    )


class RelevanceGateTests(unittest.TestCase):
    def test_observed_forbidden_feature_is_rejected(self):
        intent = _intent("Jackets. I do not want hooded")
        candidate = {
            "parent_asin": "A",
            "title": "Warm Hooded Jacket",
            "features": ["Attached hood"],
            "description": [],
            "categories": ["Clothing", "Jackets"],
            "details": {},
        }
        self.assertTrue(
            candidate_violates_observed_exclusion(candidate, intent)
        )

    def test_missing_evidence_is_not_rejected(self):
        intent = _intent("Jackets. I do not want hooded")
        candidate = {
            "parent_asin": "A",
            "title": "Warm Jacket",
            "features": ["Soft shell"],
            "description": [],
            "categories": ["Clothing", "Jackets"],
            "details": {},
        }
        self.assertFalse(
            candidate_violates_observed_exclusion(candidate, intent)
        )

    def test_filter_preserves_relative_order(self):
        intent = _intent("T-Shirts. without polyester")
        candidates = [
            {
                "parent_asin": "A",
                "title": "Cotton Tee",
                "features": ["100% cotton"],
                "description": [],
                "categories": ["Clothing", "T-Shirts"],
                "details": {},
            },
            {
                "parent_asin": "B",
                "title": "Polyester Tee",
                "features": ["100% polyester"],
                "description": [],
                "categories": ["Clothing", "T-Shirts"],
                "details": {},
            },
            {
                "parent_asin": "C",
                "title": "Linen Tee",
                "features": ["linen blend"],
                "description": [],
                "categories": ["Clothing", "T-Shirts"],
                "details": {},
            },
        ]
        filtered = filter_observed_exclusions(candidates, intent)
        self.assertEqual(
            [candidate["parent_asin"] for candidate in filtered],
            ["A", "C"],
        )

    def test_no_exclusions_returns_same_membership(self):
        intent = _intent("black waterproof hiking boots")
        candidates = [
            {"parent_asin": "A"},
            {"parent_asin": "B"},
        ]
        self.assertEqual(
            filter_observed_exclusions(candidates, intent),
            candidates,
        )

    def test_native_detail_evidence_can_trigger_exclusion(self):
        intent = _intent("T-Shirts. without polyester")
        candidate = {
            "parent_asin": "A",
            "title": "Basic Tee",
            "features": [],
            "description": [],
            "categories": ["Clothing", "T-Shirts"],
            "details": {"Material": "Polyester"},
        }
        self.assertTrue(
            candidate_violates_observed_exclusion(candidate, intent)
        )


if __name__ == "__main__":
    unittest.main()
