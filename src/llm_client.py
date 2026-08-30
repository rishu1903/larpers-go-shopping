from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Protocol


# --------------------------------------------------
# ALLOWED ATTRIBUTE FAMILIES
# --------------------------------------------------
#
# Mirrors the taxonomy already used for clarification
# question selection (src/questions.py ATTRIBUTE_VALUES),
# plus "budget" and "other" for parity with the
# evaluator's own attribute vocabulary
# (evaluator/local_evaluator.py ALLOWED_ATTRIBUTES).

ALLOWED_ATTRIBUTE_TYPES = {
    "material",
    "color",
    "size",
    "style",
    "use_case",
    "feature",
    "budget",
    "other",
}


@dataclass(frozen=True)
class SlotOp:
    """
    One instruction for updating a single
    (component, attribute_type) preference slot.

    "set" replaces whatever value that slot
    currently holds -- wherever in the conversation
    it was originally stated -- with `value`. This
    is what makes override purging component-scoped
    and turn-independent instead of hardcoded to
    turn 1.

    "clear" removes the slot entirely (the shopper
    no longer has a stated preference for it).

    `component` names the part of the item the
    preference is about ("zipper", "sole", ...), or
    "garment" for the item itself when no specific
    part is mentioned. `polarity` distinguishes a
    positive preference from an explicit exclusion
    ("I don't want blue").
    """

    op: str
    component: str
    attribute_type: str
    value: str | None = None
    polarity: str = "include"


class LLMClient(Protocol):
    """
    Provider-agnostic interface for turning a new
    user message, plus the shopper's current
    structured preferences, into slot operations.

    Implementations must fail closed: any provider,
    network, or parsing error must be caught
    internally and result in an empty list, never a
    raised exception. Callers treat an empty list as
    "no LLM-derived change, fall back to the existing
    deterministic path" -- the same contract
    src/override_semantic.py already uses for its
    semantic_override() fallback.
    """

    def extract_state_ops(
        self,
        current_state: dict,
        user_message: str,
    ) -> list[SlotOp]:
        ...


class NullLLMClient:
    """
    Default client when no provider is configured.

    Always returns no operations, so
    SessionState.update() transparently falls back to
    its existing deterministic (turn==1-purge) path.
    This keeps the pipeline's "does not require an
    external paid LLM API" default true (see README
    "Design Principles") until a provider is
    deliberately wired in.
    """

    def extract_state_ops(
        self,
        current_state: dict,
        user_message: str,
    ) -> list[SlotOp]:

        return []


# --------------------------------------------------
# ANTHROPIC TOOL SCHEMA
# --------------------------------------------------

_STATE_UPDATE_TOOL = {
    "name": "update_shopping_state",
    "description": (
        "Update the shopper's structured product "
        "preferences based on their newest message."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "ops": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "op": {
                            "type": "string",
                            "enum": [
                                "set",
                                "clear",
                            ],
                        },
                        "component": {
                            "type": "string",
                            "description": (
                                "The part of the item this "
                                "preference is about, e.g. "
                                "'zipper', 'sole', 'lining'. "
                                "Use 'garment' for the item "
                                "itself when no specific part "
                                "is mentioned."
                            ),
                        },
                        "attribute_type": {
                            "type": "string",
                            "enum": sorted(
                                ALLOWED_ATTRIBUTE_TYPES
                            ),
                        },
                        "value": {
                            "type": "string",
                        },
                        "polarity": {
                            "type": "string",
                            "enum": [
                                "include",
                                "exclude",
                            ],
                        },
                    },
                    "required": [
                        "op",
                        "component",
                        "attribute_type",
                    ],
                },
            },
        },
        "required": [
            "ops",
        ],
    },
}


_SYSTEM_PROMPT = (
    "You track a shopper's product preferences as a set "
    "of (component, attribute) slots. Given the shopper's "
    "current preference state and their newest message, "
    "call update_shopping_state with only the operations "
    "needed to bring the state up to date.\n\n"
    "A component defaults to \"garment\" (the item itself) "
    "unless the message clearly refers to a specific part "
    "(zipper, sole, lining, strap, buckle, ...).\n\n"
    "If the new message contradicts or reverses an earlier "
    "stated value for the same component+attribute -- "
    "whether or not the shopper uses an explicit phrase "
    "like \"ignore\" or \"actually\" -- emit a \"set\" op "
    "that overwrites it, regardless of which turn it was "
    "originally stated on.\n\n"
    "If the shopper says they do NOT want something, emit "
    "polarity=\"exclude\" instead of silently dropping it.\n\n"
    "If the message adds no product preference (small talk, "
    "a plain yes/no, a request to see options), return an "
    "empty ops list."
)


def _parse_ops(
    raw_ops: object,
) -> list[SlotOp]:
    """
    Defensively parse a provider's raw tool input
    into SlotOp instances.

    Any individual malformed entry is skipped rather
    than aborting the whole batch or raising --
    partial, valid ops are still worth applying, and
    a provider returning garbage must never crash a
    session (fail closed).
    """

    if not isinstance(
        raw_ops,
        list,
    ):
        return []

    parsed: list[
        SlotOp
    ] = []

    for entry in raw_ops:

        if not isinstance(
            entry,
            dict,
        ):
            continue

        op = entry.get(
            "op"
        )

        component = entry.get(
            "component"
        )

        attribute_type = entry.get(
            "attribute_type"
        )

        value = entry.get(
            "value"
        )

        polarity = entry.get(
            "polarity",
            "include",
        )

        if op not in (
            "set",
            "clear",
        ):
            continue

        if (
            not isinstance(
                component,
                str,
            )
            or not component.strip()
        ):
            continue

        if (
            attribute_type
            not in ALLOWED_ATTRIBUTE_TYPES
        ):
            continue

        if op == "set" and (
            not isinstance(
                value,
                str,
            )
            or not value.strip()
        ):
            continue

        if polarity not in (
            "include",
            "exclude",
        ):
            polarity = "include"

        parsed.append(
            SlotOp(
                op=op,
                component=(
                    component
                    .strip()
                    .lower()
                ),
                attribute_type=(
                    attribute_type
                ),
                value=(
                    value.strip()
                    if isinstance(
                        value,
                        str,
                    )
                    else None
                ),
                polarity=polarity,
            )
        )

    return parsed


class AnthropicLLMClient:
    """
    Calls the Anthropic Messages API with the
    update_shopping_state tool to extract slot
    operations.

    Fails closed on any error: a missing
    dependency, missing/invalid API key, network
    failure, or malformed response all result in an
    empty op list rather than a raised exception, so
    a session never crashes because the provider is
    unavailable -- SessionState.update() simply falls
    back to the deterministic path for that turn.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-5",
        api_key: str | None = None,
    ) -> None:

        # Imported here, not at module scope, so
        # importing this module never requires the
        # (optional) anthropic dependency to be
        # installed. Only constructing this client
        # does.
        import anthropic

        self.client = anthropic.Anthropic(
            api_key=(
                api_key
                or os.environ.get(
                    "ANTHROPIC_API_KEY"
                )
            )
        )

        self.model = model

    def extract_state_ops(
        self,
        current_state: dict,
        user_message: str,
    ) -> list[SlotOp]:

        try:
            response = (
                self.client
                .messages
                .create(
                    model=self.model,
                    max_tokens=1024,
                    system=_SYSTEM_PROMPT,
                    tools=[
                        _STATE_UPDATE_TOOL,
                    ],
                    tool_choice={
                        "type": "tool",
                        "name": (
                            "update_shopping_state"
                        ),
                    },
                    messages=[
                        {
                            "role": "user",
                            "content": (
                                "Current state:\n"
                                f"{json.dumps(current_state)}"
                                "\n\nNew message:\n"
                                f"\"{user_message}\""
                            ),
                        }
                    ],
                )
            )

            for block in response.content:

                if (
                    getattr(
                        block,
                        "type",
                        None,
                    )
                    == "tool_use"
                ):

                    return _parse_ops(
                        block.input.get(
                            "ops"
                        )
                    )

            return []

        except Exception:

            return []


# --------------------------------------------------
# LAZY DEFAULT CLIENT
# --------------------------------------------------
#
# Mirrors src/override_semantic.py's
# get_detector()/_load_attempted pattern: construct at
# most once, cache a failure as NullLLMClient so a
# missing key/dependency doesn't retry (and silently
# eat latency) on every single turn.

_default_client: LLMClient | None = None
_default_client_loaded = False


def get_default_client() -> LLMClient:
    """
    Return the shared default client: a real provider
    if credentials are configured and loadable,
    otherwise NullLLMClient.

    This is an opt-in capability -- with no
    ANTHROPIC_API_KEY set, the pipeline stays exactly
    as deterministic and network-free as before (see
    README "Design Principles" / V13).
    """

    global _default_client
    global _default_client_loaded

    if _default_client_loaded:
        return _default_client  # type: ignore[return-value]

    _default_client_loaded = True

    if os.environ.get(
        "ANTHROPIC_API_KEY"
    ):

        try:
            _default_client = (
                AnthropicLLMClient()
            )

        except Exception:

            _default_client = (
                NullLLMClient()
            )

    else:

        _default_client = (
            NullLLMClient()
        )

    return _default_client
