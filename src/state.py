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
    is_override,
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
        "those options are not quite right yet"
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

    def update(
        self,
        user_message: str,
        turn: int,
    ) -> None:

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

        override_triggered = (
            is_override(
                user_message,
            )
        )

        if override_triggered:

            self.override_seen = True

            self.clarification_exhausted = (
                False
            )

            # Remove mutable Turn-1 evidence.
            self.evidence = [
                item

                for item
                in self.evidence

                if item.turn != 1
            ]

            # Budget is also mutable evidence.
            #
            # If the stale preference being
            # discarded originated on Turn 1,
            # remove its structured form too.
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
        #
        # A newly stated budget supersedes the
        # previous budget instead of intersecting
        # indefinitely with stale constraints.

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
        Remember products already shown.
        """

        self.recommended_asins.update(
            str(
                parent_asin
            )

            for parent_asin
            in parent_asins
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