from __future__ import annotations

from src.product_profile import build_product_profile
from src.shopping_intent import CompiledShoppingIntent


def candidate_violates_observed_exclusion(
    candidate: dict,
    intent: CompiledShoppingIntent | None,
) -> bool:
    """
    Return True only when catalogue evidence explicitly confirms a forbidden
    attribute value.

    Missing catalogue evidence is intentionally neutral. This keeps negative
    constraints conservative: we remove known violations without assuming that
    an undocumented attribute is absent or present.
    """

    if intent is None or not intent.exclusions:
        return False

    profile = build_product_profile(candidate)

    for attribute, signals in intent.exclusions.items():
        observed_values = set(profile.values(attribute))
        if not observed_values:
            continue

        for signal in signals:
            if signal.value in observed_values:
                return True

    return False


def filter_observed_exclusions(
    candidates: list[dict],
    intent: CompiledShoppingIntent | None,
) -> list[dict]:
    """
    Preserve ranking order while removing products that explicitly exhibit a
    forbidden value.
    """

    if intent is None or not intent.exclusions:
        return list(candidates)

    return [
        candidate
        for candidate in candidates
        if not candidate_violates_observed_exclusion(candidate, intent)
    ]
