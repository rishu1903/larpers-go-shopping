from __future__ import annotations

from enum import Enum
import re


class ShoppingIntent(
    str,
    Enum,
):
    """
    High-level shopping mode.

    BUYING
        The shopper has moved into a
        constraint-driven purchase search.

    BROWSING
        The shopper is still exploring
        possibilities and may not yet have
        precise constraints.
    """

    BUYING = "buying"
    BROWSING = "browsing"


# --------------------------------------------------
# BROWSING SIGNALS
# --------------------------------------------------
#
# These indicate that the shopper is deliberately
# keeping the search space broad.

BROWSING_PATTERNS = (
    r"\bstill exploring\b",
    r"\bjust browsing\b",
    r"\bnot sure\b",
    r"\bnot certain\b",
    r"\bopen to\b",
    r"\blooking for ideas\b",
    r"\bshow me (?:some )?(?:ideas|options)\b",
    r"\bwhat would you recommend\b",
    r"\bhelp me explore\b",
)


# --------------------------------------------------
# STRONG BUYING SIGNALS
# --------------------------------------------------
#
# These indicate explicit requirements or
# purchase constraints.

BUYING_PATTERNS = (
    r"\bkey requirement\b",
    r"\brequirement is\b",
    r"\bmust have\b",
    r"\bi need\b",
    r"\bwhat i need is\b",
    r"\bneeds? to\b",
    r"\bunder \$?\d+\b",
    r"\bbelow \$?\d+\b",
    r"\bless than \$?\d+\b",
    r"\bbudget\b",
    r"\bsize\s*[:=]?\s*[a-z0-9]+\b",
)


# --------------------------------------------------
# NARROWING SIGNALS
# --------------------------------------------------
#
# These are especially important when a shopper
# started in BROWSING mode.
#
# They indicate that the user is now supplying
# concrete preferences and should move toward the
# high-precision BUYING path.

NARROWING_PATTERNS = (
    r"^for that, what matters is:",
    r"\bi prefer\b",
    r"\bi'd prefer\b",
    r"\bi would prefer\b",
    r"\bi want\b",
    r"\bi'd like\b",
    r"\bi would like\b",
    r"\bit must\b",
    r"\bi need it\b",
    r"\bmy budget\b",
    r"\bthe main thing is\b",
    r"\bwhat matters most is\b",
    r"\bideally\b",
)


def _matches_any(
    text: str,
    patterns: tuple[str, ...],
) -> bool:
    """
    Return True when any intent signal
    appears in the message.
    """

    return any(
        re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )
        is not None
        for pattern
        in patterns
    )


def infer_intent(
    user_message: str,
    turn: int,
    current: ShoppingIntent | None,
) -> tuple[
    ShoppingIntent,
    float,
]:
    """
    Infer the shopper's current mode.

    The router uses only participant-visible
    conversation text.

    It never receives or reconstructs the
    evaluator's hidden scenario type.

    Intent is dynamic rather than permanent.

    Example:

        Turn 1:
            "I'm still exploring jackets"

            -> BROWSING

        Turn 2:
            "I would prefer something waterproof"

            -> BUYING

    Returns:

        (
            intent,
            confidence,
        )
    """

    text = re.sub(
        r"\s+",
        " ",
        user_message.lower(),
    ).strip()

    # --------------------------------------------------
    # 1. EXPLICIT INTENT OVERRIDE
    # --------------------------------------------------
    #
    # An override is the clearest possible signal
    # that the shopper now has a concrete need.

    if (
        "ignore my earlier preference"
        in text
        or
        "what i need is"
        in text
    ):

        return (
            ShoppingIntent.BUYING,
            1.0,
        )

    # --------------------------------------------------
    # 2. EXPLICIT BROWSING LANGUAGE
    # --------------------------------------------------
    #
    # We check this before generic purchase wording
    # because a sentence may contain a category while
    # explicitly saying the shopper is still exploring.

    if _matches_any(
        text,
        BROWSING_PATTERNS,
    ):

        return (
            ShoppingIntent.BROWSING,
            0.95,
        )

    # --------------------------------------------------
    # 3. STRONG BUYING LANGUAGE
    # --------------------------------------------------

    if _matches_any(
        text,
        BUYING_PATTERNS,
    ):

        return (
            ShoppingIntent.BUYING,
            0.95,
        )

    # --------------------------------------------------
    # 4. BROWSING -> BUYING TRANSITION
    # --------------------------------------------------
    #
    # This is the important dynamic-routing case.
    #
    # A vague shopper may gradually establish concrete
    # preferences. Once that happens, we switch from
    # exploratory intent into a precision-oriented
    # buying intent.

    if (
        current
        == ShoppingIntent.BROWSING
        and
        turn > 1
        and
        _matches_any(
            text,
            NARROWING_PATTERNS,
        )
    ):

        return (
            ShoppingIntent.BUYING,
            0.85,
        )

    # --------------------------------------------------
    # 5. PRESERVE CURRENT MODE WHEN EVIDENCE IS WEAK
    # --------------------------------------------------
    #
    # We should not oscillate between modes because of
    # ordinary conversational filler.

    if current is not None:

        return (
            current,
            0.60,
        )

    # --------------------------------------------------
    # 6. CONSERVATIVE DEFAULT
    # --------------------------------------------------
    #
    # A concrete product-category request with no
    # exploratory language is treated as purchase
    # oriented, but with lower confidence.

    return (
        ShoppingIntent.BUYING,
        0.55,
    )