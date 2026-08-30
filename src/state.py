from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import re

from src.hard_constraints import (
    BudgetConstraint,
    parse_budget_constraint,
)

from src.intent import (
    ShoppingIntent,
    infer_intent,
)

from src.slots import (
    MULTI_VALUE_ATTRIBUTES,
    STRUCTURED_ATTRIBUTES,
    SlotObservation,
    SlotState,
    detect_explicit_slots,
    normalize_slot_value,
)


_NO_PREFERENCE_RE = re.compile(
    r"i don't have "
    r"(?:an additional |a )?"
    r"preference for ([a-z_]+)",
    re.IGNORECASE,
)


_ADDITIONAL_NO_PREFERENCE_RE = re.compile(
    r"i don't have an additional "
    r"preference for ([a-z_]+)",
    re.IGNORECASE,
)


_FAILURE_REPLY_PREFIX = (
    "those options are not quite right yet"
)


_OVERRIDE_MARKER = (
    "ignore my earlier preference"
)


def _clean_customer_message(
    message: str,
) -> str:
    """
    Remove simulator/dialogue boilerplate while
    preserving searchable product evidence.

    V14A.1 remains observation-only: this function
    intentionally preserves V13 retrieval behaviour.
    """

    text = re.sub(
        r"\s+",
        " ",
        message,
    ).strip()

    lowered = (
        text.lower()
    )

    if lowered.startswith(
        _FAILURE_REPLY_PREFIX
    ):
        return ""

    if lowered.startswith(
        "i don't have a preference for"
    ):
        return ""

    if lowered.startswith(
        "i don't have an additional preference for"
    ):
        return ""

    replacements = (
        (
            r"^I'm looking for\s+",
            "",
        ),
        (
            r"\bA key requirement is:\s*",
            " ",
        ),
        (
            r"^For that, what matters is:\s*",
            "",
        ),
        (
            (
                r"^Actually, "
                r"ignore my earlier preference\.\s*"
                r"What I need is:\s*"
            ),
            "",
        ),
    )

    for (
        pattern,
        replacement,
    ) in replacements:

        text = re.sub(
            pattern,
            replacement,
            text,
            flags=re.IGNORECASE,
        )

    text = re.sub(
        r",?\s*but I'm still exploring\.?$",
        "",
        text,
        flags=re.IGNORECASE,
    )

    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip(
        " ."
    )


def _number_text(
    value: float,
) -> str:

    if float(
        value
    ).is_integer():

        return str(
            int(
                value
            )
        )

    return (
        f"{value:g}"
    )


def _budget_slot_value(
    constraint: BudgetConstraint,
) -> str:

    minimum = (
        constraint.min_price
    )

    maximum = (
        constraint.max_price
    )

    if (
        minimum is not None
        and
        maximum is not None
    ):

        return (
            "$"
            f"{_number_text(minimum)}"
            "-"
            "$"
            f"{_number_text(maximum)}"
        )

    if minimum is not None:

        return (
            "at least $"
            f"{_number_text(minimum)}"
        )

    if maximum is not None:

        return (
            "up to $"
            f"{_number_text(maximum)}"
        )

    return "budget specified"


@dataclass
class Evidence:
    turn: int
    text: str


@dataclass
class SessionState:
    user_profile: dict

    # ----------------------------------
    # ACTIVE SHOPPING INTENT
    # ----------------------------------

    intent: ShoppingIntent | None = None

    intent_confidence: float = 0.0

    intent_history: list[
        tuple[
            int,
            str,
        ]
    ] = field(
        default_factory=list
    )

    # ----------------------------------
    # EXISTING V13 PRODUCT STATE
    # ----------------------------------

    category_text: str = ""

    evidence: list[
        Evidence
    ] = field(
        default_factory=list
    )

    # ----------------------------------
    # STRUCTURED HARD CONSTRAINTS
    # ----------------------------------

    budget_constraint: (
        BudgetConstraint
        | None
    ) = None

    budget_source_turn: (
        int
        | None
    ) = None

    # ----------------------------------
    # V14 STRUCTURED CONTEXT
    # ----------------------------------

    slot_state: SlotState = field(
        default_factory=SlotState
    )

    last_asked_attribute: (
        str
        | None
    ) = None

    # ----------------------------------
    # DIALOGUE STATE
    # ----------------------------------

    no_preference: set[
        str
    ] = field(
        default_factory=set
    )

    asked_attributes: set[
        str
    ] = field(
        default_factory=set
    )

    override_seen: bool = False

    clarification_exhausted: bool = False

    recommended_asins: set[
        str
    ] = field(
        default_factory=set
    )

    # ----------------------------------
    # V13 FAILURE-AWARE STATE
    # ----------------------------------

    intent_epoch: int = 0

    miss_streak: int = 0

    failure_events: list[
        tuple[
            int,
            int,
        ]
    ] = field(
        default_factory=list
    )

    last_recommendations: list[
        str
    ] = field(
        default_factory=list
    )

    failed_recommendations_by_epoch: dict[
        int,
        set[str],
    ] = field(
        default_factory=dict
    )

    def _observe_failure(
        self,
        user_message: str,
        turn: int,
    ) -> None:

        lowered = re.sub(
            r"\s+",
            " ",
            user_message.lower(),
        ).strip()

        if not lowered.startswith(
            _FAILURE_REPLY_PREFIX
        ):
            return

        self.miss_streak += 1

        self.failure_events.append(
            (
                self.intent_epoch,
                turn,
            )
        )

        failed = (
            self
            .failed_recommendations_by_epoch
            .setdefault(
                self.intent_epoch,
                set(),
            )
        )

        failed.update(
            self.last_recommendations
        )

    def _set_detected_slots(
        self,
        *,
        text: str,
        turn: int,
        provenance: str,
        replace_detected_attributes: bool,
    ) -> None:
        """
        Record conservatively detected explicit
        values.

        Normal evidence:
            append multi-valued dimensions;
            replace single-valued dimensions.

        Explicit override:
            replace all active values only for the
            dimensions detected in the override.
        """

        detections = (
            detect_explicit_slots(
                text
            )
        )

        grouped: dict[
            str,
            list[str],
        ] = defaultdict(
            list
        )

        confidence_by_attribute: dict[
            str,
            float,
        ] = {}

        for detected in detections:

            grouped[
                detected.attribute
            ].append(
                detected.value
            )

            confidence_by_attribute[
                detected.attribute
            ] = max(
                confidence_by_attribute.get(
                    detected.attribute,
                    0.0,
                ),
                detected.confidence,
            )

        for (
            attribute,
            values,
        ) in grouped.items():

            confidence = (
                confidence_by_attribute[
                    attribute
                ]
            )

            if replace_detected_attributes:

                self.slot_state.replace_values(
                    attribute=attribute,
                    values=values,
                    source_turn=turn,
                    intent_epoch=(
                        self.intent_epoch
                    ),
                    confidence=confidence,
                    strength="soft",
                    provenance=provenance,
                )

                continue

            mode = (
                "append"

                if (
                    attribute
                    in MULTI_VALUE_ATTRIBUTES
                )

                else

                "replace"
            )

            for value in values:

                self.slot_state.set_slot(
                    attribute=attribute,
                    value=value,
                    source_turn=turn,
                    intent_epoch=(
                        self.intent_epoch
                    ),
                    confidence=confidence,
                    strength="soft",
                    provenance=provenance,
                    mode=mode,
                )

                # Once a single-valued dimension
                # receives the first detected value,
                # any additional same-message match
                # is treated as an additive lexical
                # signal rather than repeatedly
                # replacing within the same message.
                if (
                    attribute
                    not in MULTI_VALUE_ATTRIBUTES
                ):
                    mode = "append"

    def _observe_structured_text(
        self,
        *,
        text: str,
        turn: int,
        pending_attribute: (
            str
            | None
        ),
        is_override: bool,
    ) -> None:

        normalized = (
            normalize_slot_value(
                text
            )
        )

        if not normalized:
            return

        # ----------------------------------
        # EXPLICIT OVERRIDE
        # ----------------------------------

        if is_override:

            self._set_detected_slots(
                text=normalized,
                turn=turn,
                provenance="override",
                replace_detected_attributes=True,
            )

            return

        # ----------------------------------
        # STRUCTURED QUESTION ANSWER
        # ----------------------------------

        if (
            pending_attribute
            in STRUCTURED_ATTRIBUTES
        ):

            strength = "soft"

            if (
                pending_attribute
                == "category"
            ):
                strength = "hard"

            if (
                pending_attribute
                == "budget"

                and

                self.budget_source_turn
                == turn

                and

                self.budget_constraint
                is not None
            ):

                value = (
                    _budget_slot_value(
                        self.budget_constraint
                    )
                )

                strength = "hard"

            else:

                value = normalized

            # A direct answer to a specific
            # structured question is treated as
            # the latest authoritative value for
            # that dimension.
            self.slot_state.set_slot(
                attribute=(
                    pending_attribute
                ),
                value=value,
                source_turn=turn,
                intent_epoch=(
                    self.intent_epoch
                ),
                confidence=1.0,
                strength=strength,
                provenance=(
                    "question_answer"
                ),
                mode="replace",
            )

            return

        # ----------------------------------
        # UNSTRUCTURED / "OTHER" ANSWER
        # ----------------------------------

        self._set_detected_slots(
            text=normalized,
            turn=turn,
            provenance=(
                "explicit_keyword"
            ),
            replace_detected_attributes=False,
        )

    def update(
        self,
        user_message: str,
        turn: int,
    ) -> None:

        # ----------------------------------
        # 0. CONSUME PREVIOUS QUESTION
        # ----------------------------------

        pending_attribute = (
            self.last_asked_attribute
        )

        self.last_asked_attribute = (
            None
        )

        # ----------------------------------
        # 1. FAILURE SIGNAL
        # ----------------------------------

        self._observe_failure(
            user_message=user_message,
            turn=turn,
        )

        # ----------------------------------
        # 2. SHOPPING INTENT
        # ----------------------------------

        previous_intent = (
            self.intent
        )

        (
            self.intent,
            self.intent_confidence,
        ) = infer_intent(
            user_message=user_message,
            turn=turn,
            current=self.intent,
        )

        if (
            self.intent
            != previous_intent
        ):

            self.intent_history.append(
                (
                    turn,
                    self.intent.value,
                )
            )

        # ----------------------------------
        # 3. NO-PREFERENCE TRACKING
        # ----------------------------------

        no_pref = (
            _NO_PREFERENCE_RE.search(
                user_message
            )
        )

        if no_pref:

            attribute = (
                no_pref
                .group(1)
                .lower()
            )

            self.no_preference.add(
                attribute
            )

            if (
                attribute
                in STRUCTURED_ATTRIBUTES
            ):

                self.slot_state.clear_slot(
                    attribute=attribute,
                    source_turn=turn,
                    provenance=(
                        "no_preference"
                    ),
                )

        additional_no_pref = (
            _ADDITIONAL_NO_PREFERENCE_RE
            .search(
                user_message
            )
        )

        if (
            additional_no_pref

            and

            additional_no_pref
            .group(1)
            .lower()
            == "other"
        ):

            self.clarification_exhausted = (
                True
            )

        # ----------------------------------
        # 4. INTENT OVERRIDE
        # ----------------------------------

        lowered_message = (
            user_message.lower()
        )

        is_override = (
            "actually"
            in lowered_message

            and

            _OVERRIDE_MARKER
            in lowered_message
        )

        if is_override:

            self.override_seen = True

            self.intent_epoch += 1

            self.miss_streak = 0

            self.last_recommendations = []

            self.clarification_exhausted = (
                False
            )

            # ----------------------------------
            # LEGACY V13 PRODUCTION BEHAVIOUR
            # ----------------------------------
            #
            # Still preserved exactly during
            # V14A.1.

            self.evidence = [
                item

                for item
                in self.evidence

                if item.turn != 1
            ]

            if (
                self.budget_source_turn
                == 1
            ):

                self.budget_constraint = (
                    None
                )

                self.budget_source_turn = (
                    None
                )

                self.slot_state.clear_slot(
                    attribute="budget",
                    source_turn=turn,
                    provenance=(
                        "legacy_override_reset"
                    ),
                )

        # ----------------------------------
        # 5. STRUCTURED BUDGET
        # ----------------------------------

        parsed_budget = (
            parse_budget_constraint(
                user_message
            )
        )

        if parsed_budget is not None:

            self.budget_constraint = (
                parsed_budget
            )

            self.budget_source_turn = (
                turn
            )

            self.slot_state.set_slot(
                attribute="budget",
                value=(
                    _budget_slot_value(
                        parsed_budget
                    )
                ),
                source_turn=turn,
                intent_epoch=(
                    self.intent_epoch
                ),
                confidence=1.0,
                strength="hard",
                provenance=(
                    "budget_parser"
                ),
                mode="replace",
            )

        # ----------------------------------
        # 6. PRODUCT EVIDENCE
        # ----------------------------------

        cleaned = (
            _clean_customer_message(
                user_message
            )
        )

        if not cleaned:
            return

        if turn == 1:

            (
                category,
                separator,
                remaining,
            ) = cleaned.partition(
                "."
            )

            self.category_text = (
                category.strip()
            )

            if self.category_text:

                self.slot_state.set_slot(
                    attribute="category",
                    value=(
                        self.category_text
                    ),
                    source_turn=turn,
                    intent_epoch=(
                        self.intent_epoch
                    ),
                    confidence=1.0,
                    strength="hard",
                    provenance="category",
                    mode="replace",
                )

            if (
                separator
                and
                remaining.strip()
            ):

                remaining_text = (
                    remaining.strip()
                )

                self._observe_structured_text(
                    text=remaining_text,
                    turn=turn,
                    pending_attribute=(
                        pending_attribute
                    ),
                    is_override=(
                        is_override
                    ),
                )

                self.evidence.append(
                    Evidence(
                        turn=turn,
                        text=remaining_text,
                    )
                )

            return

        # ----------------------------------
        # V14 STRUCTURED OBSERVATION
        # ----------------------------------

        self._observe_structured_text(
            text=cleaned,
            turn=turn,
            pending_attribute=(
                pending_attribute
            ),
            is_override=(
                is_override
            ),
        )

        # ----------------------------------
        # EXISTING V13 SEARCH EVIDENCE
        # ----------------------------------

        self.evidence.append(
            Evidence(
                turn=turn,
                text=cleaned,
            )
        )

    def active_text(
        self,
    ) -> str:
        """
        Existing V13 retrieval text.

        V14A.1 remains observation-only.
        """

        parts: list[
            str
        ] = []

        if self.category_text:

            parts.append(
                self.category_text
            )

        parts.extend(
            item.text

            for item
            in self.evidence
        )

        return " ".join(
            parts
        )

    def active_slots(
        self,
    ) -> dict[str, str]:
        """
        Compatibility snapshot returning the most
        recent active value per attribute.
        """

        return (
            self.slot_state
            .active_snapshot()
        )

    def active_slot_values(
        self,
    ) -> dict[
        str,
        list[str],
    ]:
        """
        Complete current structured context.

        Multi-valued attributes retain every active
        preference.
        """

        return (
            self.slot_state
            .active_value_snapshot()
        )

    def slot_history(
        self,
        attribute: str,
    ) -> list[
        SlotObservation
    ]:

        return (
            self.slot_state
            .history_for(
                attribute
            )
        )

    def record_recommendations(
        self,
        parent_asins: list[str],
    ) -> None:

        normalized = [
            str(
                parent_asin
            )

            for parent_asin
            in parent_asins
        ]

        self.last_recommendations = (
            normalized
        )

        self.recommended_asins.update(
            normalized
        )

    def record_question(
        self,
        attribute: str | None,
    ) -> None:

        self.last_asked_attribute = (
            attribute
        )

        if attribute:

            self.asked_attributes.add(
                attribute
            )

    def failed_recommendations(
        self,
        epoch: int | None = None,
    ) -> set[str]:

        selected_epoch = (
            self.intent_epoch
            if epoch is None
            else epoch
        )

        return set(
            self
            .failed_recommendations_by_epoch
            .get(
                selected_epoch,
                set(),
            )
        )