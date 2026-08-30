from __future__ import annotations

from dataclasses import dataclass
import re
from collections.abc import Iterable


# --------------------------------------------------
# NUMERIC VALUE
# --------------------------------------------------

_NUMBER = (
    r"(\d+(?:,\d{3})*(?:\.\d{1,2})?)"
)


# --------------------------------------------------
# MONEY CONTEXT
# --------------------------------------------------
#
# We deliberately require evidence that a number
# refers to money before interpreting expressions
# such as:
#
#     "up to 80"
#
# as a budget.
#
# Without this guard, ordinary product features
# such as:
#
#     "fits up to 8-inch wrist"
#
# would incorrectly become:
#
#     max budget = $8
#

_MONEY_CONTEXT_RE = re.compile(
    r"(?:"
    r"\$"
    r"|"
    r"\busd\b"
    r"|"
    r"\bdollars?\b"
    r"|"
    r"\bbudget\b"
    r"|"
    r"\bprice\b"
    r"|"
    r"\bcost\b"
    r"|"
    r"\bspend\b"
    r"|"
    r"\bspending\b"
    r")",
    re.IGNORECASE,
)


# --------------------------------------------------
# RANGE PATTERNS
# --------------------------------------------------

_RANGE_PATTERNS = (
    re.compile(
        rf"\bbetween\s+"
        rf"(?:[$]\s*)?{_NUMBER}"
        rf"\s+(?:and|to)\s+"
        rf"(?:[$]\s*)?{_NUMBER}\b",
        re.IGNORECASE,
    ),

    re.compile(
        rf"\bfrom\s+"
        rf"(?:[$]\s*)?{_NUMBER}"
        rf"\s+(?:to|-)\s+"
        rf"(?:[$]\s*)?{_NUMBER}\b",
        re.IGNORECASE,
    ),

    re.compile(
        rf"(?:[$]\s*)?{_NUMBER}"
        rf"\s*-\s*"
        rf"(?:[$]\s*)?{_NUMBER}",
        re.IGNORECASE,
    ),
)


# --------------------------------------------------
# MAXIMUM PRICE PATTERNS
# --------------------------------------------------

_MAX_PATTERNS = (
    re.compile(
        rf"\bunder\s+"
        rf"(?:[$]\s*)?{_NUMBER}",
        re.IGNORECASE,
    ),

    re.compile(
        rf"\bbelow\s+"
        rf"(?:[$]\s*)?{_NUMBER}",
        re.IGNORECASE,
    ),

    re.compile(
        rf"\bless than\s+"
        rf"(?:[$]\s*)?{_NUMBER}",
        re.IGNORECASE,
    ),

    re.compile(
        rf"\bat most\s+"
        rf"(?:[$]\s*)?{_NUMBER}",
        re.IGNORECASE,
    ),

    re.compile(
        rf"\bno more than\s+"
        rf"(?:[$]\s*)?{_NUMBER}",
        re.IGNORECASE,
    ),

    re.compile(
        rf"\bup to\s+"
        rf"(?:[$]\s*)?{_NUMBER}",
        re.IGNORECASE,
    ),

    re.compile(
        rf"\bwithin\s+"
        rf"(?:[$]\s*)?{_NUMBER}",
        re.IGNORECASE,
    ),

    re.compile(
        rf"\bmax(?:imum)?"
        rf"(?:\s+of)?"
        rf"\s+(?:[$]\s*)?"
        rf"{_NUMBER}",
        re.IGNORECASE,
    ),

    # A stated shopping budget is interpreted
    # as the shopper's maximum spend.
    #
    # Examples:
    #
    #   budget $80
    #   budget: 80
    #   budget is 80
    #   budget of $80
    re.compile(
        rf"\bbudget"
        rf"(?:\s+is|\s+of|\s*:)?"
        rf"\s+(?:[$]\s*)?"
        rf"{_NUMBER}",
        re.IGNORECASE,
    ),

    re.compile(
        rf"(?:<=|<)\s*"
        rf"(?:[$]\s*)?"
        rf"{_NUMBER}",
        re.IGNORECASE,
    ),

    re.compile(
        rf"\bcheaper than\s+"
        rf"(?:[$]\s*)?{_NUMBER}",
        re.IGNORECASE,
    ),

    # cap/ceiling/capped at/of $X
    #
    # Deliberately distinct wording from the
    # existing "over"/"above" MIN patterns above --
    # "capped at $50" does not overlap with "over
    # $50" or "above $50", so this does not risk
    # also satisfying a MIN pattern on the same
    # number.
    re.compile(
        rf"\b(?:cap|ceiling|capped)\s+"
        rf"(?:at|of)\s+"
        rf"(?:[$]\s*)?{_NUMBER}",
        re.IGNORECASE,
    ),
)


# --------------------------------------------------
# MINIMUM PRICE PATTERNS
# --------------------------------------------------

_MIN_PATTERNS = (
    re.compile(
        rf"\bover\s+"
        rf"(?:[$]\s*)?{_NUMBER}",
        re.IGNORECASE,
    ),

    re.compile(
        rf"\babove\s+"
        rf"(?:[$]\s*)?{_NUMBER}",
        re.IGNORECASE,
    ),

    re.compile(
        rf"\bmore than\s+"
        rf"(?:[$]\s*)?{_NUMBER}",
        re.IGNORECASE,
    ),

    re.compile(
        rf"\bat least\s+"
        rf"(?:[$]\s*)?{_NUMBER}",
        re.IGNORECASE,
    ),

    re.compile(
        rf"\bminimum"
        rf"(?:\s+of)?"
        rf"\s+(?:[$]\s*)?"
        rf"{_NUMBER}",
        re.IGNORECASE,
    ),

    re.compile(
        rf"(?:>=|>)\s*"
        rf"(?:[$]\s*)?"
        rf"{_NUMBER}",
        re.IGNORECASE,
    ),
)


# --------------------------------------------------
# NEGATION
# --------------------------------------------------
#
# A bound keyword immediately preceded by a negation
# word inverts the meaning of the phrase instead of
# stating it:
#
#     not over $100
#     not under $80
#
# Without this guard, the "over"/"under" patterns
# above would match regardless of the negation and
# silently produce the opposite of what the shopper
# said.

_NEGATION_PREFIX_RE = re.compile(
    r"\b(?:not|n't|never)\s+$",
    re.IGNORECASE,
)


def _is_negated_at(
    text: str,
    start: int,
    window: int = 12,
) -> bool:

    prefix = text[
        max(0, start - window):start
    ]

    return (
        _NEGATION_PREFIX_RE.search(
            prefix
        )
        is not None
    )


@dataclass(
    frozen=True
)
class BudgetConstraint:
    """
    Structured hard price constraint.

    Either price bound may be absent.

    Examples:

        under $100

            min_price = None
            max_price = 100

        over $50

            min_price = 50
            max_price = None

        $50-$100

            min_price = 50
            max_price = 100
    """

    min_price: float | None = None
    max_price: float | None = None

    def matches(
        self,
        price: float,
    ) -> bool:
        """
        Return whether a product price satisfies
        this hard constraint.
        """

        if (
            self.min_price is not None
            and
            price < self.min_price
        ):
            return False

        if (
            self.max_price is not None
            and
            price > self.max_price
        ):
            return False

        return True


# --------------------------------------------------
# EXPLICIT REMOVAL
# --------------------------------------------------
#
# A shopper can explicitly cancel a previously
# stated budget without providing a replacement
# value:
#
#     no budget limit
#     doesn't matter on price
#     any price is fine
#
# parse_budget_constraint() returns the REMOVE_BUDGET
# sentinel for these phrases so the caller can
# distinguish "explicit removal requested" from
# "nothing budget-related said" (both of which would
# otherwise be indistinguishable None results).

class _RemoveBudget:

    __slots__ = ()

    def __repr__(self) -> str:
        return "REMOVE_BUDGET"


REMOVE_BUDGET = _RemoveBudget()


_NO_CONSTRAINT_RE = re.compile(
    r"\bno\s+budget(?:\s+limit)?\b"
    r"|\bno\s+price\s+limit\b"
    r"|\bno\s+limit\s+on\s+(?:budget|price|spending)\b"
    r"|\b(?:any|no)\s+price\s+is\s+fine\b"
    r"|\bprice\s+(?:doesn't|does\s+not)\s+matter\b"
    r"|\b(?:doesn't|does\s+not)\s+matter\s+on\s+price\b"
    r"|\bnot\s+picky\s+about\s+price\b",
    re.IGNORECASE,
)


def _is_explicit_no_constraint(
    text: str,
) -> bool:

    return (
        _NO_CONSTRAINT_RE.search(
            text
        )
        is not None
    )


def _number(
    value: str,
) -> float:
    """
    Convert a captured monetary number
    into a float.
    """

    return float(
        value.replace(
            ",",
            "",
        )
    )


def _has_money_context(
    text: str,
) -> bool:
    """
    Return whether the message contains enough
    evidence that a numeric expression refers
    to money.

    This protects the parser from product
    measurements such as:

        up to 8-inch wrist
        up to 30 metres
        100-hour battery

    while still accepting:

        up to $80
        budget up to 80
        price under 80
        under 80 dollars
    """

    return (
        _MONEY_CONTEXT_RE.search(
            text
        )
        is not None
    )


def coerce_price(
    value: object,
) -> float | None:
    """
    Safely convert catalogue price metadata
    into a numeric value.

    Supported examples:

        39.99
        "39.99"
        "$39.99"

    Missing or malformed prices return None.
    """

    if value is None:
        return None

    if isinstance(
        value,
        bool,
    ):
        return None

    if isinstance(
        value,
        (
            int,
            float,
        ),
    ):
        numeric = float(
            value
        )

        if numeric < 0:
            return None

        return numeric

    text = str(
        value
    ).strip()

    if not text:
        return None

    match = re.search(
        r"(?:[$]\s*)?"
        r"(\d+(?:,\d{3})*(?:\.\d{1,2})?)",
        text,
    )

    if not match:
        return None

    numeric = _number(
        match.group(1)
    )

    if numeric < 0:
        return None

    return numeric


def parse_budget_constraint(
    text: str,
) -> BudgetConstraint | _RemoveBudget | None:
    """
    Extract an explicit hard budget constraint.

    Returns REMOVE_BUDGET when the shopper explicitly
    cancels a budget without providing a replacement
    value (e.g. "no budget limit"), and None when
    nothing budget-related was said at all -- callers
    that only replace an existing constraint on a new
    value should treat any non-None result as
    actionable and branch on `result is REMOVE_BUDGET`
    to distinguish the two.

    A hard price constraint requires monetary
    context.

    Accepted examples:

        under $80
        budget under 80
        price below 100
        spend up to 120
        between $50 and $100

    Rejected examples:

        up to 8-inch wrist
        up to 30 metres
        100-hour battery
        around $100

    Approximate price language is intentionally
    not converted into a strict constraint.
    """

    normalized = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    if not normalized:
        return None

    # --------------------------------------------------
    # REQUIRE MONEY CONTEXT
    # --------------------------------------------------
    #
    # This is the key V8.1 safety guard.

    if not _has_money_context(
        normalized
    ):
        return None

    # --------------------------------------------------
    # DO NOT HARD-FILTER APPROXIMATE BUDGETS
    # --------------------------------------------------

    if re.search(
        r"\b(?:around|about|roughly|approximately)\b",
        normalized,
        flags=re.IGNORECASE,
    ):
        return None

    # --------------------------------------------------
    # RANGE
    # --------------------------------------------------

    for pattern in _RANGE_PATTERNS:

        match = pattern.search(
            normalized
        )

        if not match:
            continue

        first = _number(
            match.group(1)
        )

        second = _number(
            match.group(2)
        )

        return BudgetConstraint(
            min_price=min(
                first,
                second,
            ),
            max_price=max(
                first,
                second,
            ),
        )

    # --------------------------------------------------
    # LOWER / UPPER BOUNDS
    # --------------------------------------------------

    min_price: float | None = None
    max_price: float | None = None

    for pattern in _MIN_PATTERNS:

        match = pattern.search(
            normalized
        )

        if match and not _is_negated_at(
            normalized,
            match.start(),
        ):

            min_price = _number(
                match.group(1)
            )

            break

    for pattern in _MAX_PATTERNS:

        match = pattern.search(
            normalized
        )

        if match and not _is_negated_at(
            normalized,
            match.start(),
        ):

            max_price = _number(
                match.group(1)
            )

            break

    if (
        min_price is None
        and
        max_price is None
    ):

        if _is_explicit_no_constraint(
            normalized
        ):
            return REMOVE_BUDGET

        return None

    # Reject contradictory constraints instead
    # of silently creating an empty result set.
    if (
        min_price is not None
        and
        max_price is not None
        and
        min_price > max_price
    ):
        return None

    return BudgetConstraint(
        min_price=min_price,
        max_price=max_price,
    )


def apply_budget_constraint(
    candidates: Iterable[dict],
    constraint: BudgetConstraint | None,
) -> list[dict]:
    """
    Apply the budget as a genuine eligibility
    filter while preserving candidate order.

    Products without usable price metadata cannot
    be verified against an active hard budget, so
    they are excluded when such a constraint is
    active.
    """

    items = list(
        candidates
    )

    if constraint is None:
        return items

    filtered: list[
        dict
    ] = []

    for candidate in items:

        price = coerce_price(
            candidate.get(
                "price"
            )
        )

        if price is None:
            continue

        if constraint.matches(
            price
        ):

            filtered.append(
                candidate
            )

    return filtered