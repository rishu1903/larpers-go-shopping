from __future__ import annotations

from dataclasses import dataclass
import re

from src.state import (
    SessionState,
)


# V14 production configuration.
#
# Structured context distillation activates only
# when the slot lifecycle is sufficiently resolved
# to rewrite conversational evidence safely.
#
# Unknown or unresolved intent overrides continue
# to fall back to the proven V13 context path.
STRUCTURED_CONTEXT_DISTILLATION_ENABLED = True


# Budget remains a real structured hard constraint
# rather than textual retrieval evidence.
#
# Category retains its dedicated retrieval field.
TEXTUAL_SLOT_ATTRIBUTES = {
    "material",
    "color",
    "size",
    "style",
    "brand",
    "feature",
    "use_case",
}


# Only explicit, current-session evidence can be
# rewritten into the retrieval query.
#
# Weak inference and historical profile priors must
# never silently become concrete shopping
# requirements.
SEARCH_SAFE_PROVENANCES = {
    "question_answer",
    "explicit_keyword",
    "override",
}


@dataclass(frozen=True)
class DistilledContext:
    category_text: str
    residual_text: str
    evidence_text: str
    active_text: str

    active_values: tuple[str, ...]
    removed_values: tuple[str, ...]

    mode: str
    fallback_reason: str | None


def configure_context_distillation(
    enabled: bool,
) -> None:
    """
    Enable or disable V14 structured-context
    distillation.

    Production uses enabled=True.

    Offline evaluation scripts can still override
    the setting temporarily for reproducible
    ablations.
    """

    global STRUCTURED_CONTEXT_DISTILLATION_ENABLED

    STRUCTURED_CONTEXT_DISTILLATION_ENABLED = (
        bool(
            enabled
        )
    )


def _normalize_space(
    text: str,
) -> str:

    return re.sub(
        r"\s+",
        " ",
        str(
            text
        ),
    ).strip()


def _legacy_evidence_text(
    state: SessionState,
) -> str:

    return _normalize_space(
        " ".join(
            item.text

            for item
            in state.evidence
        )
    )


def _legacy_active_text(
    state: SessionState,
) -> str:

    parts: list[str] = []

    if state.category_text:

        parts.append(
            state.category_text
        )

    evidence = (
        _legacy_evidence_text(
            state
        )
    )

    if evidence:

        parts.append(
            evidence
        )

    return _normalize_space(
        " ".join(
            parts
        )
    )


def _legacy_context(
    state: SessionState,
    *,
    reason: str,
) -> DistilledContext:
    """
    Return exact V13-style textual context.

    This is the safety fallback whenever structured
    lifecycle information is insufficient to prove
    which preference changed.
    """

    evidence_text = (
        _legacy_evidence_text(
            state
        )
    )

    return DistilledContext(
        category_text=(
            state.category_text
        ),

        residual_text=(
            evidence_text
        ),

        evidence_text=(
            evidence_text
        ),

        active_text=(
            _legacy_active_text(
                state
            )
        ),

        active_values=(),

        removed_values=(),

        mode="legacy_fallback",

        fallback_reason=reason,
    )


def _phrase_pattern(
    value: str,
) -> re.Pattern[str]:
    """
    Build a conservative case-insensitive phrase
    matcher.

    Word boundaries prevent values such as:

        red

    from being removed from:

        infrared
    """

    normalized = (
        _normalize_space(
            value
        )
    )

    escaped = re.escape(
        normalized
    )

    escaped = escaped.replace(
        r"\ ",
        r"\s+",
    )

    return re.compile(
        (
            r"(?<!\w)"
            f"{escaped}"
            r"(?!\w)"
        ),
        re.IGNORECASE,
    )


def _remove_phrase(
    text: str,
    value: str,
) -> str:

    if (
        not text
        or
        not value
    ):
        return text

    return (
        _phrase_pattern(
            value
        )
        .sub(
            " ",
            text,
        )
    )


def _clean_residual(
    text: str,
) -> str:
    """
    Clean punctuation left after structured values
    are removed.

    Unknown natural language is intentionally
    retained because it may contain useful product
    evidence outside the controlled slot model.
    """

    cleaned = re.sub(
        r"\s*;\s*",
        " ",
        text,
    )

    cleaned = re.sub(
        r"\s*,\s*",
        " ",
        cleaned,
    )

    cleaned = re.sub(
        r"\s+",
        " ",
        cleaned,
    )

    return cleaned.strip(
        " .;,"
    )


def _safe_observation(
    observation: object,
) -> bool:

    attribute = getattr(
        observation,
        "attribute",
        None,
    )

    provenance = getattr(
        observation,
        "provenance",
        None,
    )

    value = getattr(
        observation,
        "value",
        None,
    )

    return (
        attribute
        in TEXTUAL_SLOT_ATTRIBUTES

        and

        provenance
        in SEARCH_SAFE_PROVENANCES

        and

        isinstance(
            value,
            str,
        )

        and

        bool(
            value.strip()
        )
    )


def _current_override_is_structurally_resolved(
    state: SessionState,
) -> bool:
    """
    Determine whether the most recent intent
    override was understood by the structured
    observer.

    An observation with:

        provenance == "override"
        intent_epoch == current intent epoch

    can only have been created from the explicit
    override message itself.

    This prevents later ordinary evidence from
    accidentally making an earlier unknown
    override appear resolved.
    """

    if not state.override_seen:

        return True

    return any(
        (
            observation.provenance
            == "override"

            and

            observation.intent_epoch
            == state.intent_epoch

            and

            observation.attribute
            in TEXTUAL_SLOT_ATTRIBUTES
        )

        for observation
        in state.slot_state.history
    )


def _has_safe_lifecycle_change(
    state: SessionState,
) -> bool:
    """
    Distillation is useful only when structured
    state can correct something in legacy
    conversation text.

    Examples:

        black -> blue
        cotton -> cleared
        waterproof -> breathable

    Pure accumulation requires no rewriting and
    therefore stays on the proven V13 path.
    """

    for observation in (
        state.slot_state.history
    ):

        if not _safe_observation(
            observation
        ):
            continue

        if observation.status in {
            "superseded",
            "cleared",
        }:

            return True

    return False


def should_distill_context(
    state: SessionState,
) -> bool:
    """
    Conservative V14 activation gate.

    No override / no lifecycle mutation
    ------------------------------------

        Keep V13.

    Known structured replacement or clearing
    ----------------------------------------

        Use V14 distillation.

    Resolved explicit override
    --------------------------

        Use V14 distillation.

    Unknown explicit override
    -------------------------

        Fall back to V13.

    The final rule protects private-set sessions
    containing arbitrary values outside the
    controlled slot vocabulary.
    """

    if (
        state.override_seen

        and

        not
        _current_override_is_structurally_resolved(
            state
        )
    ):

        return False

    if (
        state.override_seen

        and

        _current_override_is_structurally_resolved(
            state
        )
    ):

        return True

    return (
        _has_safe_lifecycle_change(
            state
        )
    )


def build_distilled_context(
    state: SessionState,
) -> DistilledContext:
    """
    Build a compact active shopping context.

    Structured rewriting activates only when the
    lifecycle safety gate allows it.

    Algorithm
    =========

    1. Begin with existing free-text evidence.

    2. Remove every safely identified structured
       value from its original conversational
       location.

    3. Restore only currently active structured
       values.

    4. Preserve every unknown free-text phrase.

    Example
    -------

        Turn 1:
            cotton + black

        Turn 2:
            override black -> blue

    Legacy V13:
            blue

    Structured V14:
            cotton + blue

    Unknown override:

        cotton + black
            ↓
        "charcoal heather"

    falls back to V13 because the affected
    structured dimension cannot be proven safely.
    """

    if not should_distill_context(
        state
    ):

        reason = (
            "unresolved_override"

            if (
                state.override_seen

                and

                not
                _current_override_is_structurally_resolved(
                    state
                )
            )

            else

            "no_structured_lifecycle_change"
        )

        return _legacy_context(
            state,
            reason=reason,
        )

    legacy_evidence = (
        _legacy_evidence_text(
            state
        )
    )

    safe_history = [
        observation

        for observation
        in state.slot_state.history

        if _safe_observation(
            observation
        )
    ]

    removal_values = sorted(
        {
            str(
                observation.value
            ).strip()

            for observation
            in safe_history

            if (
                observation.value
                is not None
            )
        },
        key=lambda value: (
            -len(
                value
            ),
            value.casefold(),
        ),
    )

    residual = (
        legacy_evidence
    )

    for value in removal_values:

        residual = (
            _remove_phrase(
                residual,
                value,
            )
        )

    residual = (
        _clean_residual(
            residual
        )
    )

    active_values: list[
        str
    ] = []

    seen: set[
        str
    ] = set()

    for observation in safe_history:

        if (
            observation.status
            != "active"

            or

            observation.value
            is None
        ):

            continue

        value = (
            _normalize_space(
                observation.value
            )
        )

        key = (
            value.casefold()
        )

        if (
            not value

            or

            key in seen
        ):
            continue

        seen.add(
            key
        )

        active_values.append(
            value
        )

    evidence_parts: list[
        str
    ] = []

    if residual:

        evidence_parts.append(
            residual
        )

    evidence_parts.extend(
        active_values
    )

    evidence_text = (
        _normalize_space(
            " ".join(
                evidence_parts
            )
        )
    )

    active_parts: list[
        str
    ] = []

    if state.category_text:

        active_parts.append(
            state.category_text
        )

    if evidence_text:

        active_parts.append(
            evidence_text
        )

    active_text = (
        _normalize_space(
            " ".join(
                active_parts
            )
        )
    )

    return DistilledContext(
        category_text=(
            state.category_text
        ),

        residual_text=residual,

        evidence_text=evidence_text,

        active_text=active_text,

        active_values=tuple(
            active_values
        ),

        removed_values=tuple(
            removal_values
        ),

        mode="structured",

        fallback_reason=None,
    )


def retrieval_evidence_text(
    state: SessionState,
) -> str:
    """
    Evidence used by lexical retrieval.

    Feature disabled:
        exact V13 behaviour.

    Feature enabled:
        safe structured distillation when possible,
        otherwise exact V13 fallback.
    """

    if (
        not
        STRUCTURED_CONTEXT_DISTILLATION_ENABLED
    ):

        return (
            _legacy_evidence_text(
                state
            )
        )

    return (
        build_distilled_context(
            state
        )
        .evidence_text
    )


def retrieval_active_text(
    state: SessionState,
) -> str:
    """
    Full text used by semantic retrieval.
    """

    if (
        not
        STRUCTURED_CONTEXT_DISTILLATION_ENABLED
    ):

        return (
            state.active_text()
        )

    return (
        build_distilled_context(
            state
        )
        .active_text
    )