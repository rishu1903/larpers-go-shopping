from __future__ import annotations

import unittest

from src.llm_client import (
    NullLLMClient,
    SlotOp,
)

from src.state import (
    SessionState,
)

from src.state_tracker import (
    apply_ops,
    snapshot_state,
    update_item_context,
)


class StubLLMClient:
    """
    Deterministic stand-in for a real provider, keyed
    by exact user message text. Lets the state-tracking
    logic (component-scoped purge, override detection)
    be tested without any network access or API key --
    exactly what "design it now, wire credentials later"
    calls for.
    """

    def __init__(
        self,
        ops_by_message: dict[str, list[SlotOp]],
    ) -> None:

        self._ops_by_message = ops_by_message
        self.calls: list[tuple[dict, str]] = []

    def extract_state_ops(
        self,
        current_state: dict,
        user_message: str,
    ) -> list[SlotOp]:

        self.calls.append(
            (current_state, user_message)
        )

        return self._ops_by_message.get(
            user_message,
            [],
        )


# --------------------------------------------------
# THE MOTIVATING EXAMPLE
# --------------------------------------------------
#
# Turn 1: "I want a jacket"                 -> category
# Turn 2: "The zipper need to be silver."   -> stale pref
#                                               on turn 2,
#                                               not turn 1
# Turn 3: "nevermind, i want it to be gold" -> must replace
#                                               ONLY the
#                                               zipper color,
#                                               not the
#                                               category

_TURN2_MESSAGE = "The zipper need to be silver."
_TURN3_MESSAGE = "nevermind, i want it to be gold"


def _jacket_zipper_client() -> StubLLMClient:

    return StubLLMClient(
        {
            _TURN2_MESSAGE: [
                SlotOp(
                    op="set",
                    component="zipper",
                    attribute_type="color",
                    value="silver",
                ),
            ],
            _TURN3_MESSAGE: [
                SlotOp(
                    op="set",
                    component="zipper",
                    attribute_type="color",
                    value="gold",
                ),
            ],
        }
    )


class ComponentScopedOverrideTest(unittest.TestCase):

    def test_override_replaces_component_slot_regardless_of_turn(self):
        # This is the exact case the turn==1-only purge
        # missed: the stale preference is on turn 2.

        client = _jacket_zipper_client()

        state = SessionState(user_profile={})

        state.update("I want a jacket", 1, client=client)
        state.update(_TURN2_MESSAGE, 2, client=client)
        state.update(_TURN3_MESSAGE, 3, client=client)

        active = state.active_text().lower()

        self.assertIn("jacket", active)
        self.assertIn("gold", active)
        self.assertNotIn("silver", active)
        self.assertTrue(state.override_seen)

    def test_category_is_never_a_purge_target(self):
        # "jacket" must survive even though the override
        # happened -- it was never a component slot.

        client = _jacket_zipper_client()

        state = SessionState(user_profile={})

        state.update("I want a jacket", 1, client=client)
        state.update(_TURN2_MESSAGE, 2, client=client)
        state.update(_TURN3_MESSAGE, 3, client=client)

        self.assertIn(
            "jacket",
            state.category_text.lower(),
        )

    def test_unrelated_component_slots_survive_the_override(self):
        # A slot for a DIFFERENT component (sole) must not
        # be touched by an override targeting the zipper.

        client = StubLLMClient(
            {
                "The sole should be rubber.": [
                    SlotOp(
                        op="set",
                        component="sole",
                        attribute_type="material",
                        value="rubber",
                    ),
                ],
                _TURN2_MESSAGE: [
                    SlotOp(
                        op="set",
                        component="zipper",
                        attribute_type="color",
                        value="silver",
                    ),
                ],
                _TURN3_MESSAGE: [
                    SlotOp(
                        op="set",
                        component="zipper",
                        attribute_type="color",
                        value="gold",
                    ),
                ],
            }
        )

        state = SessionState(user_profile={})

        state.update("I want a jacket", 1, client=client)
        state.update("The sole should be rubber.", 2, client=client)
        state.update(_TURN2_MESSAGE, 3, client=client)
        state.update(_TURN3_MESSAGE, 4, client=client)

        active = state.active_text().lower()

        self.assertIn("rubber", active)
        self.assertIn("gold", active)
        self.assertNotIn("silver", active)

    def test_no_client_configured_falls_back_to_existing_behavior(self):
        # Regression guard: with NullLLMClient (the
        # default absent a provider), behavior must stay
        # identical to the pre-existing deterministic
        # path so the rest of the test suite -- including
        # tests/test_override_robustness.py -- keeps
        # passing unmodified.

        state = SessionState(user_profile={})

        state.update(
            "I am looking for running shoes. black style",
            1,
            client=NullLLMClient(),
        )

        state.update(
            "For that, what matters is: waterproof.",
            2,
            client=NullLLMClient(),
        )

        state.update(
            "Actually, ignore my earlier preference. "
            "What I need is: casual white sneakers.",
            3,
            client=NullLLMClient(),
        )

        active = state.active_text().lower()

        self.assertNotIn("black style", active)
        self.assertIn("waterproof", active)
        self.assertIn("casual white sneakers", active)
        self.assertTrue(state.override_seen)


class ApplyOpsTest(unittest.TestCase):

    def test_empty_ops_makes_no_change_and_reports_no_override(self):

        state = SessionState(user_profile={})

        overrode = apply_ops(state=state, ops=[], turn=2)

        self.assertFalse(overrode)
        self.assertEqual(state.evidence, [])

    def test_set_on_empty_slot_is_not_reported_as_an_override(self):
        # Introducing a brand-new preference (nothing to
        # replace) should not be treated as a reversal.

        state = SessionState(user_profile={})

        overrode = apply_ops(
            state=state,
            ops=[
                SlotOp(
                    op="set",
                    component="zipper",
                    attribute_type="color",
                    value="silver",
                ),
            ],
            turn=2,
        )

        self.assertFalse(overrode)
        self.assertEqual(len(state.evidence), 1)
        self.assertEqual(state.evidence[0].component, "zipper")
        self.assertEqual(state.evidence[0].attribute_type, "color")

    def test_clear_removes_matching_slot_without_replacement(self):

        state = SessionState(user_profile={})

        apply_ops(
            state=state,
            ops=[
                SlotOp(
                    op="set",
                    component="zipper",
                    attribute_type="color",
                    value="silver",
                ),
            ],
            turn=2,
        )

        overrode = apply_ops(
            state=state,
            ops=[
                SlotOp(
                    op="clear",
                    component="zipper",
                    attribute_type="color",
                ),
            ],
            turn=3,
        )

        self.assertTrue(overrode)
        self.assertEqual(state.evidence, [])

    def test_garment_component_renders_without_the_component_name(self):

        state = SessionState(user_profile={})

        apply_ops(
            state=state,
            ops=[
                SlotOp(
                    op="set",
                    component="garment",
                    attribute_type="color",
                    value="navy",
                ),
            ],
            turn=1,
        )

        self.assertEqual(state.evidence[0].text, "navy")


class SnapshotStateTest(unittest.TestCase):

    def test_only_component_tagged_evidence_is_reported(self):

        state = SessionState(user_profile={})
        state.category_text = "jacket"

        apply_ops(
            state=state,
            ops=[
                SlotOp(
                    op="set",
                    component="zipper",
                    attribute_type="color",
                    value="silver",
                ),
            ],
            turn=2,
        )

        # Untagged evidence from the deterministic path
        # has no slot identity to report.
        from src.state import Evidence

        state.evidence.append(
            Evidence(turn=3, text="loose fit")
        )

        snapshot = snapshot_state(state)

        self.assertEqual(snapshot["category"], "jacket")
        self.assertEqual(len(snapshot["preferences"]), 1)
        self.assertEqual(
            snapshot["preferences"][0]["component"],
            "zipper",
        )


class UpdateItemContextTest(unittest.TestCase):

    def test_returns_false_when_client_has_nothing_to_say(self):

        state = SessionState(user_profile={})

        applied = update_item_context(
            state=state,
            user_message="sounds good",
            turn=2,
            client=StubLLMClient({}),
        )

        self.assertFalse(applied)
        self.assertEqual(state.evidence, [])


if __name__ == "__main__":
    unittest.main()
