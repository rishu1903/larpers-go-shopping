from __future__ import annotations

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
    Remove conversation/simulator boilerplate
    while preserving searchable product evidence.
    """

    text = re.sub(
        r"\s+",
        " ",
        message,
    ).strip()

    lowered = text.lower()

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
            r"^Actually, ignore my earlier preference\.\s*"
            r"What I need is:\s*",
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
    ).strip(" .")


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
    # PRODUCT STATE
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
    #
    # These fields observe explicit customer
    # rejection without changing V12 ranking by
    # themselves. The V13 orchestration layer can
    # use them during controlled shadow ablations.

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
        """
        Attribute an explicit rejection to the
        previous recommendation set.

        The failure message is treated as a runtime
        strategy signal, not searchable product
        evidence.
        """

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

    def update(
        self,
        user_message: str,
        turn: int,
    ) -> None:

        # ----------------------------------
        # 0. OBSERVE RECOMMENDATION FAILURE
        # ----------------------------------

        self._observe_failure(
            user_message=user_message,
            turn=turn,
        )

        # ----------------------------------
        # 1. UPDATE SHOPPING INTENT
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
        # 2. NO-PREFERENCE TRACKING
        # ----------------------------------

        no_pref = (
            _NO_PREFERENCE_RE.search(
                user_message
            )
        )

        if no_pref:

            self.no_preference.add(
                no_pref
                .group(1)
                .lower()
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
        # 3. INTENT OVERRIDE
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

            # A new intent starts a new failure epoch.
            #
            # Historical rejection evidence remains
            # available for diagnostics but does not
            # poison the new intent's miss streak.
            self.intent_epoch += 1

            self.miss_streak = 0

            self.last_recommendations = []

            self.clarification_exhausted = (
                False
            )

            # Preserve category while deleting stale
            # mutable Turn-1 evidence.
            self.evidence = [
                item

                for item
                in self.evidence

                if item.turn != 1
            ]

            # Budget is also mutable evidence.
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

        # ----------------------------------
        # 4. STRUCTURED BUDGET EXTRACTION
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

        # ----------------------------------
        # 5. POSITIVE PRODUCT EVIDENCE
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

            if (
                separator
                and
                remaining.strip()
            ):

                self.evidence.append(
                    Evidence(
                        turn=turn,
                        text=(
                            remaining.strip()
                        ),
                    )
                )

            return

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
        Return category plus currently valid
        textual product evidence.
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

    def record_recommendations(
        self,
        parent_asins: list[str],
    ) -> None:
        """
        Remember shown products and retain the
        most recent recommendation set.

        This lets a later explicit customer rejection
        be attributed to the products that caused it.
        """

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
        """
        Remember structured questions
        already asked.
        """

        if attribute:

            self.asked_attributes.add(
                attribute
            )

    def failed_recommendations(
        self,
        epoch: int | None = None,
    ) -> set[str]:
        """
        Return rejected ASINs for one intent epoch.

        V13A does not yet use this set as a hard
        product filter. It exists so future strategy
        logic can distinguish same-intent negatives
        from products shown before an override.
        """

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