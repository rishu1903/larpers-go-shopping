from __future__ import annotations

from dataclasses import dataclass, field
import re


_NO_PREFERENCE_RE = re.compile(
    r"i don't have (?:an additional |a )?preference for ([a-z_]+)",
    re.IGNORECASE,
)


def _clean_customer_message(message: str) -> str:
    """Remove simulator boilerplate while preserving product-relevant evidence."""

    text = re.sub(r"\s+", " ", message).strip()

    # These simulator replies contain no useful product information.
    lowered = text.lower()

    if lowered.startswith("those options are not quite right yet"):
        return ""

    if lowered.startswith("i don't have a preference for"):
        return ""

    if lowered.startswith("i don't have an additional preference for"):
        return ""

    # Strip fixed simulator wording while keeping the useful product evidence.
    replacements = (
        (r"^I'm looking for\s+", ""),
        (r"\bA key requirement is:\s*", " "),
        (r"^For that, what matters is:\s*", ""),
        (
            r"^Actually, ignore my earlier preference\.\s*What I need is:\s*",
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

    # Remove browsing boilerplate.
    text = re.sub(
        r",?\s*but I'm still exploring\.?$",
        "",
        text,
        flags=re.IGNORECASE,
    )

    return re.sub(r"\s+", " ", text).strip(" .")


@dataclass
class Evidence:
    turn: int
    text: str


@dataclass
class SessionState:
    user_profile: dict

    evidence: list[Evidence] = field(default_factory=list)

    no_preference: set[str] = field(default_factory=set)

    override_seen: bool = False

    def update(
        self,
        user_message: str,
        turn: int,
    ) -> None:

        # Detect explicit "no preference" answers.
        no_pref = _NO_PREFERENCE_RE.search(user_message)

        if no_pref:
            self.no_preference.add(
                no_pref.group(1).lower()
            )

        # Detect the official intent-override wording.
        is_override = (
            "actually" in user_message.lower()
            and "ignore my earlier preference"
            in user_message.lower()
        )

        if is_override:
            self.override_seen = True

            # The stale preference originates in Turn 1.
            #
            # Remove Turn 1, but KEEP useful constraints
            # learned during intermediate turns.
            self.evidence = [
                item
                for item in self.evidence
                if item.turn != 1
            ]

        cleaned = _clean_customer_message(user_message)

        if cleaned:
            self.evidence.append(
                Evidence(
                    turn=turn,
                    text=cleaned,
                )
            )

    def active_text(self) -> str:
        """Return all currently valid search evidence."""

        return " ".join(
            item.text
            for item in self.evidence
        )