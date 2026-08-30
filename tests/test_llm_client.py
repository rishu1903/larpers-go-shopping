from __future__ import annotations

import unittest

from src.llm_client import (
    NullLLMClient,
    SlotOp,
    _parse_ops,
)


class NullLLMClientTest(unittest.TestCase):

    def test_always_returns_no_ops(self):

        client = NullLLMClient()

        self.assertEqual(
            client.extract_state_ops(
                current_state={"category": "jacket", "preferences": []},
                user_message="nevermind, i want it to be gold",
            ),
            [],
        )


class ParseOpsTest(unittest.TestCase):
    """
    Exercises the defensive parsing a real provider's
    raw tool response goes through. A provider is
    untrusted input: malformed or partially-invalid
    entries must be dropped individually, never crash
    the caller.
    """

    def test_valid_set_op_parses(self):

        parsed = _parse_ops(
            [
                {
                    "op": "set",
                    "component": "Zipper",
                    "attribute_type": "color",
                    "value": "gold",
                    "polarity": "include",
                }
            ]
        )

        self.assertEqual(len(parsed), 1)
        self.assertEqual(
            parsed[0],
            SlotOp(
                op="set",
                component="zipper",
                attribute_type="color",
                value="gold",
                polarity="include",
            ),
        )

    def test_valid_clear_op_does_not_require_value(self):

        parsed = _parse_ops(
            [
                {
                    "op": "clear",
                    "component": "garment",
                    "attribute_type": "color",
                }
            ]
        )

        self.assertEqual(len(parsed), 1)
        self.assertIsNone(parsed[0].value)

    def test_not_a_list_returns_empty(self):

        self.assertEqual(_parse_ops(None), [])
        self.assertEqual(_parse_ops("set color gold"), [])
        self.assertEqual(_parse_ops({"op": "set"}), [])

    def test_unknown_attribute_type_is_dropped(self):

        parsed = _parse_ops(
            [
                {
                    "op": "set",
                    "component": "zipper",
                    "attribute_type": "hardware_finish",
                    "value": "gold",
                }
            ]
        )

        self.assertEqual(parsed, [])

    def test_set_without_value_is_dropped(self):

        parsed = _parse_ops(
            [
                {
                    "op": "set",
                    "component": "zipper",
                    "attribute_type": "color",
                }
            ]
        )

        self.assertEqual(parsed, [])

    def test_missing_component_is_dropped(self):

        parsed = _parse_ops(
            [
                {
                    "op": "set",
                    "attribute_type": "color",
                    "value": "gold",
                }
            ]
        )

        self.assertEqual(parsed, [])

    def test_invalid_polarity_defaults_to_include(self):

        parsed = _parse_ops(
            [
                {
                    "op": "set",
                    "component": "zipper",
                    "attribute_type": "color",
                    "value": "gold",
                    "polarity": "sideways",
                }
            ]
        )

        self.assertEqual(parsed[0].polarity, "include")

    def test_one_bad_entry_does_not_drop_the_others(self):

        parsed = _parse_ops(
            [
                {"op": "not_a_real_op"},
                {
                    "op": "set",
                    "component": "zipper",
                    "attribute_type": "color",
                    "value": "gold",
                },
            ]
        )

        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0].value, "gold")


if __name__ == "__main__":
    unittest.main()
