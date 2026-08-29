from __future__ import annotations

from dataclasses import dataclass, field
import re


_NO_PREFERENCE_RE = re.compile(
    r"i don't have (?:an additional |a )?preference for ([a-z_]+)",
    re.IGNORECASE,
)

_ADDITIONAL_NO_PREFERENCE_RE = re.compile(
    r"i don't have an additional preference for ([a-z_]+)",
    re.IGNORECASE,
)


def _clean_customer_message(message: str) -> str:
    """
    Remove simulator boilerplate while preserving
    product-relevant evidence.
    """

    text = re.sub(
        r"\s+",
        " ",
        message,
    ).strip()

    lowered = text.lower()

    # These replies contain no new searchable
    # product information.
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

    for pattern, replacement in replacements:
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

    # Persistent broad product category.
    category_text: str = ""

    # Active mutable customer evidence.
    evidence: list[Evidence] = field(
        default_factory=list
    )

    # Attributes the customer explicitly
    # said they have no preference for.
    no_preference: set[str] = field(
        default_factory=set
    )

    override_seen: bool = False

    # Once the customer explicitly says that
    # there is no additional preference left
    # to provide, the agent should stop asking
    # the same question repeatedly and switch
    # from exploitation to exploration.
    clarification_exhausted: bool = False

    # Products already shown during this session.
    # This allows later turns to surface new
    # alternatives rather than repeating results.
    recommended_asins: set[str] = field(
        default_factory=set
    )

    def update(
        self,
        user_message: str,
        turn: int,
    ) -> None:

        no_pref = _NO_PREFERENCE_RE.search(
            user_message
        )

        if no_pref:
            self.no_preference.add(
                no_pref.group(1).lower()
            )

        # Important distinction:
        #
        # "I don't have a preference for X"
        #
        # does NOT necessarily mean the customer
        # has nothing else to tell us.
        #
        # But:
        #
        # "I don't have an ADDITIONAL preference"
        #
        # means our broad clarification attempt
        # has run out of useful information.
        if _ADDITIONAL_NO_PREFERENCE_RE.search(
            user_message
        ):
            self.clarification_exhausted = True

        is_override = (
            "actually"
            in user_message.lower()
            and
            "ignore my earlier preference"
            in user_message.lower()
        )

        if is_override:
            self.override_seen = True

            # An override represents a new intent
            # state, so clarification may once again
            # become useful.
            self.clarification_exhausted = False

            # Remove mutable Turn-1 preference
            # evidence while preserving:
            #
            # - the separately stored category
            # - useful evidence learned later
            self.evidence = [
                item
                for item in self.evidence
                if item.turn != 1
            ]

        cleaned = _clean_customer_message(
            user_message
        )

        if not cleaned:
            return

        if turn == 1:
            # Example:
            #
            # "Shirts Polos. Button closure"
            #
            # Broad category:
            #   Shirts Polos
            #
            # Mutable preference:
            #   Button closure

            head, separator, tail = (
                cleaned.partition(".")
            )

            self.category_text = (
                head.strip()
            )

            if (
                separator
                and tail.strip()
            ):
                self.evidence.append(
                    Evidence(
                        turn=turn,
                        text=tail.strip(),
                    )
                )

            return

        self.evidence.append(
            Evidence(
                turn=turn,
                text=cleaned,
            )
        )

    def active_text(self) -> str:
        """
        Return the current category plus
        all active product evidence.
        """

        parts = (
            [self.category_text]
            if self.category_text
            else []
        )

        parts.extend(
            item.text
            for item in self.evidence
        )

        return " ".join(parts)

    def record_recommendations(
        self,
        parent_asins: list[str],
    ) -> None:
        """
        Remember products already shown
        during this shopping session.
        """

        self.recommended_asins.update(
            str(parent_asin)
            for parent_asin
            in parent_asins
        )