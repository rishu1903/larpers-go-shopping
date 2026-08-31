from __future__ import annotations

import unittest

from src.product_profile import build_product_profile
from src.shopping_intent import canonicalize_attribute_value


class ProductProfileTest(unittest.TestCase):

    def product(self, **overrides) -> dict:
        base = {
            "parent_asin": "A1",
            "title": "Women's Black Waterproof Hiking Boots",
            "features": [
                "Leather upper",
                "Breathable lining",
                "Non-slip sole",
            ],
            "description": [
                "A lightweight boot designed for trails.",
            ],
            "price": "$79.99",
            "categories": [
                "Clothing",
                "Women",
                "Shoes",
                "Outdoor",
                "Hiking & Trekking",
            ],
            "details": {
                "Department": "Womens",
                "Manufacturer": "Acme Manufacturing",
                "Brand": "Acme",
                "Material": "Leather",
                "Color": "Black",
                "Special Feature": "Water-Resistant",
            },
            "average_rating": 4.5,
            "rating_number": 100,
            "store": "Acme Store",
        }
        base.update(overrides)
        return base

    def test_core_native_fields_are_preserved(self) -> None:
        profile = build_product_profile(self.product())

        self.assertEqual(profile.parent_asin, "A1")
        self.assertEqual(profile.department, "women")
        self.assertEqual(profile.manufacturer, "Acme Manufacturing")
        self.assertEqual(profile.native_brand, "Acme")
        self.assertEqual(profile.store, "Acme Store")
        self.assertEqual(profile.price, 79.99)
        self.assertEqual(profile.leaf_category, "Hiking & Trekking")

    def test_controlled_attributes_are_normalized(self) -> None:
        profile = build_product_profile(self.product())

        self.assertIn("leather", profile.values("material"))
        self.assertIn("black", profile.values("color"))
        self.assertIn("waterproof", profile.values("feature"))
        self.assertIn("water_resistant", profile.values("feature"))
        self.assertIn("breathable", profile.values("feature"))
        self.assertIn("non_slip", profile.values("feature"))
        self.assertIn("lightweight", profile.values("feature"))
        self.assertIn("hiking", profile.values("use_case"))

    def test_same_value_merges_provenance(self) -> None:
        profile = build_product_profile(self.product())
        leather = next(
            signal
            for signal in profile.attributes["material"]
            if signal.value == "leather"
        )

        self.assertIn("features", leather.sources)
        self.assertIn("details:Material", leather.sources)
        self.assertTrue(leather.native)
        self.assertGreaterEqual(leather.support_count, 2)

    def test_store_is_not_attribute_evidence(self) -> None:
        profile = build_product_profile(
            self.product(
                title="Plain boots",
                features=[],
                description=[],
                categories=["Clothing", "Shoes", "Boots"],
                details={},
                store="Black Leather Outdoor Store",
            )
        )

        self.assertNotIn("black", profile.values("color"))
        self.assertNotIn("leather", profile.values("material"))
        self.assertNotIn("outdoor", profile.values("use_case"))

    def test_irrelevant_details_are_not_scanned_for_attributes(self) -> None:
        profile = build_product_profile(
            self.product(
                title="Plain boots",
                features=[],
                description=[],
                categories=["Clothing", "Shoes", "Boots"],
                details={
                    "Package Dimensions": "wide package",
                    "Item model number": "BLACK-LEATHER-1",
                },
            )
        )

        self.assertNotIn("wide", profile.values("size"))
        self.assertNotIn("black", profile.values("color"))
        self.assertNotIn("leather", profile.values("material"))

    def test_categories_contribute_to_use_case_not_color_or_material(self) -> None:
        profile = build_product_profile(
            self.product(
                title="Plain shoe",
                features=[],
                description=[],
                categories=["Black", "Leather", "Running"],
                details={},
            )
        )

        self.assertIn("running", profile.values("use_case"))
        self.assertNotIn("black", profile.values("color"))
        self.assertNotIn("leather", profile.values("material"))

    def test_shared_canonicalizer_matches_query_side_vocabulary(self) -> None:
        self.assertEqual(
            canonicalize_attribute_value("water-resistant"),
            "water_resistant",
        )
        self.assertEqual(
            canonicalize_attribute_value("jogging"),
            "running",
        )
        self.assertEqual(
            canonicalize_attribute_value("non slip"),
            "non_slip",
        )

    def test_missing_optional_fields_remain_unknown(self) -> None:
        profile = build_product_profile(
            {
                "parent_asin": "A2",
                "title": "Plain T-Shirt",
                "features": [],
                "description": [],
                "price": None,
                "categories": ["Clothing", "T-Shirts"],
                "details": {},
                "store": None,
            }
        )

        self.assertIsNone(profile.department)
        self.assertIsNone(profile.manufacturer)
        self.assertIsNone(profile.native_brand)
        self.assertIsNone(profile.store)
        self.assertIsNone(profile.price)

    def test_safe_catalogue_feature_synonyms_share_canonical_values(self) -> None:
        profile = build_product_profile(
            self.product(
                title="Plain product",
                features=[
                    "200g insulation",
                    "Slip-resistant rubber outsole",
                    "Anti-slip tread",
                    "Interior pocket",
                ],
                description=[],
                categories=["Clothing", "Shoes"],
                details={},
            )
        )

        self.assertIn("insulated", profile.values("feature"))
        self.assertIn("non_slip", profile.values("feature"))
        self.assertIn("pockets", profile.values("feature"))


if __name__ == "__main__":
    unittest.main()
