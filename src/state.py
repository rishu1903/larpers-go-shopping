from __future__ import annotations

from dataclasses import dataclass, field
import re


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
    Remove simulator / conversation boilerplate
    while preserving product evidence.
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

    # Persistent product category.
    category_text: str = ""

    # Mutable product constraints.
    evidence: list[Evidence] = field(
        default_factory=list
    )

    # Attributes explicitly declined by
    # the shopper.
    no_preference: set[str] = field(
        default_factory=set
    )

    # Attributes the agent has already asked.
    asked_attributes: set[str] = field(
        default_factory=set
    )

    override_seen: bool = False

    clarification_exhausted: bool = False

    recommended_asins: set[str] = field(
        default_factory=set
    )

    def update(
        self,
        user_message: str,
        turn: int,
    ) -> None:

        # ----------------------------------
        # NO-PREFERENCE TRACKING
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

        # Important V6 distinction:
        #
        # "I don't have an additional
        #  preference for MATERIAL"
        #
        # does NOT mean:
        #
        # "I have no other preferences."
        #
        # We can still ask about color,
        # style, size, etc.
        #
        # Only a failed broad `other`
        # clarification tells us that the
        # shopper has nothing else useful
        # to add.
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
        # INTENT OVERRIDE
        # ----------------------------------

        is_override = (
            "actually"
            in user_message.lower()

            and

            "ignore my earlier preference"
            in user_message.lower()
        )

        if is_override:

            self.override_seen = True

            # New intent may make clarification
            # productive again.
            self.clarification_exhausted = (
                False
            )

            # Remove only mutable Turn-1
            # preference evidence.
            #
            # category_text is stored separately
            # and therefore survives.
            self.evidence = [
                item
                for item
                in self.evidence
                if item.turn != 1
            ]

        # ----------------------------------
        # POSITIVE PRODUCT EVIDENCE
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
                and remaining.strip()
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
        Return current category + active
        product constraints.
        """

        parts: list[str] = []

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
        Track products previously shown.
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
        Remember which structured questions
        have already been asked.
        """

        if attribute:

            self.asked_attributes.add(
                attribute
            )