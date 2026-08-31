from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import TYPE_CHECKING

from src.hard_constraints import BudgetConstraint
from src.intent import ShoppingIntent
from src.slots import detect_explicit_slots, normalize_slot_value

if TYPE_CHECKING:
    from src.state import SessionState


@dataclass(frozen=True)
class IntentSignal:
    """One normalized piece of shopping intent."""

    value: str
    confidence: float
    source: str
    source_turn: int | None = None
    strength: str = "soft"


@dataclass(frozen=True)
class CompiledShoppingIntent:
    """
    Read-only interpretation of the shopper's current need.

    This is intentionally separate from SessionState:

        SessionState
            conversational memory / lifecycle

        CompiledShoppingIntent
            current shopping interpretation

    The production path currently consumes exclusions from this object.
    Positive attributes remain available for future query planning but do not
    override the proven V14 ranking path.
    """

    mode: ShoppingIntent
    mode_confidence: float
    product_type: IntentSignal | None
    explicit_attributes: dict[str, tuple[IntentSignal, ...]] = field(
        default_factory=dict
    )
    inferred_attributes: dict[str, tuple[IntentSignal, ...]] = field(
        default_factory=dict
    )
    exclusions: dict[str, tuple[IntentSignal, ...]] = field(
        default_factory=dict
    )
    budget: BudgetConstraint | None = None

    def to_dict(self) -> dict:
        def signal_dict(signal: IntentSignal) -> dict:
            return {
                "value": signal.value,
                "confidence": signal.confidence,
                "source": signal.source,
                "source_turn": signal.source_turn,
                "strength": signal.strength,
            }

        return {
            "mode": self.mode.value,
            "mode_confidence": self.mode_confidence,
            "product_type": (
                signal_dict(self.product_type)
                if self.product_type is not None
                else None
            ),
            "explicit_attributes": {
                attribute: [signal_dict(signal) for signal in signals]
                for attribute, signals in self.explicit_attributes.items()
            },
            "inferred_attributes": {
                attribute: [signal_dict(signal) for signal in signals]
                for attribute, signals in self.inferred_attributes.items()
            },
            "exclusions": {
                attribute: [signal_dict(signal) for signal in signals]
                for attribute, signals in self.exclusions.items()
            },
            "budget": (
                {
                    "min_price": self.budget.min_price,
                    "max_price": self.budget.max_price,
                }
                if self.budget is not None
                else None
            ),
        }


_CANONICAL_VALUE_ALIASES = {
    "hood": "hooded",
    "pocket": "pockets",
    "insulation": "insulated",
    "nonslip": "non_slip",
    "slip resistant": "non_slip",
    "slip-resistant": "non_slip",
    "anti slip": "non_slip",
    "anti-slip": "non_slip",
    "water-resistant": "water_resistant",
    "water resistant": "water_resistant",
    "non-slip": "non_slip",
    "non slip": "non_slip",
    "stretchy": "stretch",
    "jog": "running",
    "jogs": "running",
    "jogging": "running",
    "trail": "hiking",
    "trails": "hiking",
}


_IMPLICIT_PATTERNS: tuple[
    tuple[str, str, float, tuple[re.Pattern[str], ...]], ...
] = (
    (
        "feature",
        "waterproof",
        0.86,
        (
            re.compile(r"\bkeep(?:s|ing)? (?:my |your |the )?(?:feet|foot|body|things?|items?) dry\b", re.I),
            re.compile(r"\bkeep water from getting through\b", re.I),
            re.compile(r"\b(?:heavy )?rain without water getting in\b", re.I),
            re.compile(r"\bwater (?:does not|doesn't|won't) get in\b", re.I),
        ),
    ),
    (
        "feature",
        "breathable",
        0.82,
        (
            re.compile(r"\breduce heat buildup\b", re.I),
            re.compile(r"\bventilat(?:ed|ion)\b", re.I),
            re.compile(r"\b(?:good |better )?airflow\b", re.I),
        ),
    ),
    (
        "feature",
        "non_slip",
        0.82,
        (
            re.compile(r"\bsecure traction\b", re.I),
            re.compile(r"\bgrips? well\b", re.I),
            re.compile(r"\breduce sliding\b", re.I),
            re.compile(r"\bslick surfaces?\b", re.I),
        ),
    ),
    (
        "feature",
        "insulated",
        0.80,
        (
            re.compile(r"\bretain warmth\b", re.I),
            re.compile(r"\breduce heat loss\b", re.I),
            re.compile(r"\bextra warmth\b", re.I),
        ),
    ),
    (
        "feature",
        "lightweight",
        0.80,
        (
            re.compile(r"\b(?:does not|doesn't|won't) feel heavy\b", re.I),
            re.compile(r"\bminimal weight\b", re.I),
            re.compile(r"\beasy to (?:wear|carry) for long periods\b", re.I),
        ),
    ),
    (
        "feature",
        "hooded",
        0.82,
        (
            re.compile(r"\bbuilt-in head coverage\b", re.I),
            re.compile(r"\battached covering for (?:my |the )?head\b", re.I),
            re.compile(r"\bcover (?:my |the )?head\b", re.I),
        ),
    ),
    (
        "feature",
        "pockets",
        0.80,
        (
            re.compile(r"\bplaces? to keep (?:a )?(?:phone|small items?)\b", re.I),
            re.compile(r"\bcarry small belongings\b", re.I),
        ),
    ),
    (
        "feature",
        "stretch",
        0.78,
        (
            re.compile(r"\bmoves? comfortably with (?:my |the )?body\b", re.I),
            re.compile(r"\bgives? a little instead of feeling rigid\b", re.I),
            re.compile(r"\bflexibility when i move\b", re.I),
        ),
    ),
    (
        "feature",
        "adjustable",
        0.82,
        (
            re.compile(r"\btighten or loosen\b", re.I),
            re.compile(r"\bfit (?:can|could) be (?:changed|fine-tuned)\b", re.I),
        ),
    ),
    (
        "use_case",
        "running",
        0.80,
        (
            re.compile(r"\bregular jogs? and workouts?\b", re.I),
            re.compile(r"\brepeated fast-paced exercise\b", re.I),
        ),
    ),
    (
        "use_case",
        "hiking",
        0.80,
        (
            re.compile(r"\boutdoor trails?\b", re.I),
            re.compile(r"\brough paths? and trail use\b", re.I),
            re.compile(r"\btrails? and uneven terrain\b", re.I),
        ),
    ),
)


_NEGATION_PATTERNS = (
    re.compile(
        r"\bwithout\s+(?:anything\s+|any\s+|a\s+|an\s+)?"
        r"(?P<value>[^,.;]+?)(?=\s+(?:but|and|or)\b|[,.;]|$)",
        re.I,
    ),
    re.compile(
        r"\b(?:don't|dont|do not)\s+want\s+"
        r"(?:anything\s+|any\s+)?"
        r"(?P<value>[^,.;]+?)(?=\s+(?:but|and|or)\b|[,.;]|$)",
        re.I,
    ),
    re.compile(
        r"\bnot\s+(?P<value>[^,.;]+?)"
        r"(?=\s+(?:but|and|or)\b|[,.;]|$)",
        re.I,
    ),
)


_PRODUCT_REQUEST_PATTERNS = (
    re.compile(r"\b(?:i'm|i am) looking for\s+(?P<value>.+)", re.I),
    re.compile(r"\bi need\s+(?P<value>.+)", re.I),
    re.compile(r"\bi want\s+(?P<value>.+)", re.I),
    re.compile(r"\bfind me\s+(?P<value>.+)", re.I),
    re.compile(r"\bshow me\s+(?P<value>.+)", re.I),
)


_PRODUCT_TYPE_STOP_RE = re.compile(
    r"\s+(?:under|below|over|above|between|with|without|that|which|who|for)\b.*$",
    re.I,
)


def canonicalize_attribute_value(value: str) -> str:
    normalized = normalize_slot_value(value).casefold()
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return _CANONICAL_VALUE_ALIASES.get(normalized, normalized)


def _extract_product_type(
    user_message: str,
    state: SessionState,
    turn: int,
) -> IntentSignal | None:
    candidate: str | None = None

    for pattern in _PRODUCT_REQUEST_PATTERNS:
        match = pattern.search(user_message)
        if match is not None:
            candidate = match.group("value")
            break

    if candidate is None:
        candidate = state.category_text

    if not candidate:
        return None

    candidate = candidate.split(".", 1)[0]
    candidate = _PRODUCT_TYPE_STOP_RE.sub("", candidate).strip()

    # Remove explicit structured modifiers from the product noun phrase.
    for detected in detect_explicit_slots(candidate):
        candidate = re.sub(
            rf"\b{re.escape(detected.value)}\b",
            " ",
            candidate,
            flags=re.I,
        )

    candidate = re.sub(
        r"^(?:some|a|an|the|something|a pair of|pair of)\s+",
        "",
        candidate,
        flags=re.I,
    )
    candidate = re.sub(r"\s+", " ", candidate).strip(" ,.;")

    junk_tokens = {"and", "but", "not", "or", "no", "anything", "any", "only", "also"}
    candidate_tokens = set(re.findall(r"[a-z0-9]+", candidate.casefold()))
    if not candidate or (candidate_tokens and candidate_tokens <= junk_tokens):
        fallback = normalize_slot_value(state.category_text)
        candidate = fallback.strip(" ,.;")

    if not candidate:
        return None

    return IntentSignal(
        value=candidate.casefold(),
        confidence=0.95,
        source="product_request",
        source_turn=turn,
        strength="hard",
    )


def _is_real_exclusion_phrase(phrase: str) -> bool:
    normalized = normalize_slot_value(phrase)
    if not normalized:
        return False

    # Avoid common constructions such as "not only waterproof but also..."
    # where "not" does not express a product exclusion.
    lowered = normalized.casefold()
    if lowered.startswith(("only ", "necessarily ", "just ")):
        return False

    if detect_explicit_slots(normalized):
        return True

    cleaned = re.sub(r"^(?:anything|any|a|an)\s+", "", normalized, flags=re.I)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return bool(cleaned and len(cleaned.split()) <= 3)


def extract_exclusions(
    user_message: str,
    turn: int,
) -> dict[str, tuple[IntentSignal, ...]]:
    grouped: dict[str, list[IntentSignal]] = {}
    seen: set[tuple[str, str]] = set()

    for pattern in _NEGATION_PATTERNS:
        for match in pattern.finditer(user_message):
            phrase = normalize_slot_value(match.group("value"))
            if not _is_real_exclusion_phrase(phrase):
                continue

            detections = detect_explicit_slots(phrase)

            if detections:
                for detected in detections:
                    value = canonicalize_attribute_value(detected.value)
                    key = (detected.attribute, value)
                    if key in seen:
                        continue
                    seen.add(key)
                    grouped.setdefault(detected.attribute, []).append(
                        IntentSignal(
                            value=value,
                            confidence=0.98,
                            source="explicit_negation",
                            source_turn=turn,
                            strength="hard",
                        )
                    )
                continue

            # Preserve a short unknown exclusion such as "laces" without
            # pretending to know a richer catalogue attribute taxonomy yet.
            cleaned = re.sub(r"^(?:anything|any|a|an)\s+", "", phrase, flags=re.I)
            cleaned = re.sub(r"\s+", " ", cleaned).strip().casefold()
            if cleaned and len(cleaned.split()) <= 3:
                key = ("feature", cleaned)
                if key not in seen:
                    seen.add(key)
                    grouped.setdefault("feature", []).append(
                        IntentSignal(
                            value=cleaned,
                            confidence=0.90,
                            source="explicit_negation_raw",
                            source_turn=turn,
                            strength="hard",
                        )
                    )

    return {
        attribute: tuple(signals)
        for attribute, signals in grouped.items()
    }


def positive_retrieval_text(user_message: str) -> str:
    """Remove only confidently parsed negative constraints from search text.

    FTS/BM25 is bag-of-words and otherwise interprets "not insulated" as
    positive evidence for "insulated". Unknown or ambiguous negation is kept.
    """

    text = str(user_message or "")

    for pattern in _NEGATION_PATTERNS:
        def replace(match: re.Match[str]) -> str:
            phrase = normalize_slot_value(match.group("value"))
            return " " if _is_real_exclusion_phrase(phrase) else match.group(0)

        text = pattern.sub(replace, text)

    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;])", r"\1", text)
    return text.strip(" ,.;")


def _active_explicit_attributes(
    state: SessionState,
    exclusions: dict[str, tuple[IntentSignal, ...]],
    user_message: str,
    turn: int,
) -> dict[str, tuple[IntentSignal, ...]]:
    grouped: dict[str, list[IntentSignal]] = {}

    excluded = {
        (attribute, signal.value.casefold())
        for attribute, signals in exclusions.items()
        for signal in signals
    }

    for attribute in state.slot_state.active_value_snapshot():
        if attribute in {"category", "budget"}:
            continue

        for observation in state.slot_state.active_values(attribute):
            if observation.value is None:
                continue

            value = canonicalize_attribute_value(observation.value)
            if (attribute, value.casefold()) in excluded:
                continue

            confidence = state.slot_state.effective_confidence(
                observation,
                current_turn=max(1, turn),
            )

            grouped.setdefault(attribute, []).append(
                IntentSignal(
                    value=value,
                    confidence=round(float(confidence), 6),
                    source=observation.provenance,
                    source_turn=observation.source_turn,
                    strength=observation.strength,
                )
            )

    existing = {
        (attribute, signal.value.casefold())
        for attribute, signals in grouped.items()
        for signal in signals
    }

    # SessionState's turn-1 category handling intentionally preserves
    # the V13 production path, so some explicit modifiers can live only
    # in the original message. Add them to the compiled interpretation
    # without mutating SlotState.
    for detected in detect_explicit_slots(user_message):
        if detected.attribute in {"category", "budget"}:
            continue

        value = canonicalize_attribute_value(detected.value)
        key = (detected.attribute, value.casefold())
        if key in excluded or key in existing:
            continue

        existing.add(key)
        grouped.setdefault(detected.attribute, []).append(
            IntentSignal(
                value=value,
                confidence=float(detected.confidence),
                source="current_message_explicit",
                source_turn=turn,
                strength="soft",
            )
        )

    return {
        attribute: tuple(signals)
        for attribute, signals in grouped.items()
        if signals
    }


def _infer_attributes(
    user_message: str,
    explicit: dict[str, tuple[IntentSignal, ...]],
    exclusions: dict[str, tuple[IntentSignal, ...]],
    turn: int,
) -> dict[str, tuple[IntentSignal, ...]]:
    explicit_values = {
        (attribute, signal.value.casefold())
        for attribute, signals in explicit.items()
        for signal in signals
    }
    excluded_values = {
        (attribute, signal.value.casefold())
        for attribute, signals in exclusions.items()
        for signal in signals
    }

    grouped: dict[str, list[IntentSignal]] = {}

    for attribute, value, confidence, patterns in _IMPLICIT_PATTERNS:
        key = (attribute, value.casefold())
        if key in explicit_values or key in excluded_values:
            continue

        if any(pattern.search(user_message) is not None for pattern in patterns):
            grouped.setdefault(attribute, []).append(
                IntentSignal(
                    value=value,
                    confidence=confidence,
                    source="implicit_language",
                    source_turn=turn,
                    strength="inferred",
                )
            )

    return {
        attribute: tuple(signals)
        for attribute, signals in grouped.items()
    }


def compile_shopping_intent(
    *,
    state: SessionState,
    user_message: str,
    turn: int,
) -> CompiledShoppingIntent:
    """
    Compile a read-only shopping interpretation after SessionState.update().

    V15A deliberately does not mutate SessionState and does not alter
    retrieval, ranking, clarification, or the official Agent API.
    """

    mode = state.intent or ShoppingIntent.BUYING

    exclusions = extract_exclusions(user_message, turn)
    explicit = _active_explicit_attributes(
        state,
        exclusions,
        user_message,
        turn,
    )
    inferred = _infer_attributes(
        user_message,
        explicit,
        exclusions,
        turn,
    )

    return CompiledShoppingIntent(
        mode=mode,
        mode_confidence=float(state.intent_confidence),
        product_type=_extract_product_type(user_message, state, turn),
        explicit_attributes=explicit,
        inferred_attributes=inferred,
        exclusions=exclusions,
        budget=state.budget_constraint,
    )
