from __future__ import annotations

from typing import TYPE_CHECKING

from src.llm_client import (
    LLMClient,
    SlotOp,
    get_default_client,
)

if TYPE_CHECKING:
    from src.state import SessionState


def snapshot_state(
    state: "SessionState",
) -> dict:
    """
    Serialize the shopper's current component-scoped
    preferences for the LLM prompt.

    Only evidence the LLM path itself tagged with a
    (component, attribute_type) is reported -- plain
    evidence added by the deterministic fallback path
    (component=None) has no slot identity to report
    back, so it is omitted from the snapshot but still
    contributes to active_text() as before.
    """

    slots = [
        {
            "component": item.component,
            "attribute_type": item.attribute_type,
            "value": item.text,
        }

        for item
        in state.evidence

        if (
            item.component
            and
            item.attribute_type
        )
    ]

    return {
        "category": state.category_text,
        "preferences": slots,
    }


def apply_ops(
    state: "SessionState",
    ops: list[SlotOp],
    turn: int,
) -> bool:
    """
    Apply extracted slot operations to session state.

    Removal is keyed on (component, attribute_type)
    identity and scans the ENTIRE evidence list
    regardless of which turn each matching entry was
    originally added on -- this is the fix for the
    turn==1-only purge: a stale preference is found by
    what it is, not by when it was said.

    Returns whether any existing slot was actually
    overwritten or cleared, so the caller can set
    state.override_seen accurately even when the
    shopper used no explicit reversal cue phrase.
    """

    from src.state import Evidence

    if not ops:
        return False

    overrode_something = False

    for op in ops:

        key = (
            op.component,
            op.attribute_type,
        )

        before = len(
            state.evidence
        )

        state.evidence = [
            item

            for item
            in state.evidence

            if (
                item.component,
                item.attribute_type,
            )
            != key
        ]

        if len(state.evidence) < before:
            overrode_something = True

        if op.op == "set":

            rendered = (
                op.value

                if op.component == "garment"

                else f"{op.value} {op.component}"
            )

            state.evidence.append(
                Evidence(
                    turn=turn,
                    text=rendered,
                    component=op.component,
                    attribute_type=op.attribute_type,
                )
            )

    return overrode_something


def update_item_context(
    state: "SessionState",
    user_message: str,
    turn: int,
    client: LLMClient | None = None,
) -> bool:
    """
    Attempt LLM-driven, component-scoped state
    tracking for one turn.

    Returns whether it produced any change. Callers
    should fall back to the existing deterministic
    evidence path when this returns False.
    NullLLMClient (the default when no provider is
    configured) always returns no ops, so this is a
    no-op -- and the pipeline behaves exactly as
    before -- until real credentials are wired in.
    """

    active_client = (
        client
        or get_default_client()
    )

    ops = (
        active_client
        .extract_state_ops(
            snapshot_state(
                state
            ),
            user_message,
        )
    )

    if not ops:
        return False

    overrode = apply_ops(
        state=state,
        ops=ops,
        turn=turn,
    )

    if overrode:

        state.override_seen = True
        state.clarification_exhausted = False

    return True
