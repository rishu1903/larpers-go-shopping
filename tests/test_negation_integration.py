from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from starter.agent import Agent


class NegationIntegrationTest(unittest.TestCase):
    def test_agent_does_not_recommend_observed_excluded_feature(self) -> None:
        products = [
            {
                "parent_asin": "P1",
                "title": "Hooded Jacket",
                "features": ["hooded", "zip closure"],
                "description": [],
                "price": 50.0,
                "categories": ["Clothing", "Jackets"],
                "details": {},
                "average_rating": 4.0,
                "rating_number": 100,
                "store": "Store A",
            },
            {
                "parent_asin": "P2",
                "title": "Everyday Jacket",
                "features": ["zip closure"],
                "description": [],
                "price": 55.0,
                "categories": ["Clothing", "Jackets"],
                "details": {},
                "average_rating": 4.0,
                "rating_number": 90,
                "store": "Store B",
            },
            {
                "parent_asin": "P3",
                "title": "Lightweight Jacket",
                "features": ["lightweight"],
                "description": [],
                "price": 60.0,
                "categories": ["Clothing", "Jackets"],
                "details": {},
                "average_rating": 4.0,
                "rating_number": 80,
                "store": "Store C",
            },
        ]

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "catalog.jsonl"
            path.write_text(
                "".join(json.dumps(product) + "\n" for product in products),
                encoding="utf-8",
            )

            agent = Agent(path)
            agent.reset("session", {})
            response = agent.respond(
                "session",
                "Jackets. I do not want hooded",
                1,
                10,
            )

            asins = [item["parent_asin"] for item in response["recommendations"]]
            self.assertNotIn("P1", asins)
            self.assertEqual(asins, ["P2", "P3"])
            self.assertNotIn(
                "hooded",
                agent._sessions["session"].active_slot_values().get("feature", []),
            )


if __name__ == "__main__":
    unittest.main()
