from __future__ import annotations

import re


# --------------------------------------------------
# SAFE PROFILE PERSONALIZATION
# --------------------------------------------------
#
# The evaluator supplies anonymized aggregate profile
# tags such as:
#
#     material
#     fit
#     comfort
#     style
#     durability
#     performance
#     warmth
#     weather
#
# These describe DIMENSIONS that mattered in prior
# shopping behaviour.
#
# They do NOT tell us the value the shopper wants.
#
# Therefore:
#
#     "material"
#
# may justify asking about material,
#
# but must NEVER be converted into:
#
#     "cotton"
#
# unless the shopper says cotton in the current
# conversation.


DIRECT_ATTRIBUTES = {
    "material",
    "color",
    "size",
    "style",
    "brand",
    "budget",
    "feature",
    "use_case",
}


# Controlled profile tags can support one or more
# clarification dimensions.
#
# Values are relative affinities, not ranking scores.
PROFILE_TAG_TO_ATTRIBUTES: dict[
    str,
    dict[
        str,
        float,
    ],
] = {
    "material": {
        "material": 1.00,
    },

    "fit": {
        "size": 1.00,
        "style": 0.70,
    },

    "style": {
        "style": 1.00,
    },

    "comfort": {
        "feature": 0.80,
        "size": 0.50,
    },

    "durability": {
        "feature": 1.00,
    },

    "performance": {
        "feature": 0.90,
        "use_case": 0.65,
    },

    "warmth": {
        "feature": 1.00,
        "use_case": 0.40,
    },

    "weather": {
        "feature": 0.90,
        "use_case": 0.70,
    },

    # Deliberately carries no assumption.
    "general shopping": {},
}


def _normalize_tag(
    value: object,
) -> str:
    """
    Normalize one controlled profile tag.
    """

    text = re.sub(
        r"[_-]+",
        " ",
        str(
            value
            or ""
        ).lower(),
    )

    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


def preference_tags(
    user_profile: object,
) -> list[str]:
    """
    Extract only the controlled preference tags
    from the aggregate user profile.

    We intentionally do NOT infer shopping
    preferences from:

        average_prior_rating
        purchase_frequency
        rating_style
        summary

    Those fields are not safe evidence for a
    current product preference.
    """

    if not isinstance(
        user_profile,
        dict,
    ):
        return []

    raw_tags = user_profile.get(
        "preference_tags",
        [],
    )

    if not isinstance(
        raw_tags,
        list,
    ):
        return []

    result: list[str] = []

    for raw_tag in raw_tags:

        tag = _normalize_tag(
            raw_tag
        )

        if (
            tag
            and
            tag not in result
        ):

            result.append(
                tag
            )

    return result


def attribute_affinity(
    user_profile: object,
) -> dict[
    str,
    float,
]:
    """
    Convert controlled profile tags into safe
    clarification-dimension affinities.

    Example:

        ["fit", "material"]

    becomes approximately:

        {
            "size": 1.0,
            "style": 0.7,
            "material": 1.0,
        }

    Importantly, no concrete preference VALUE
    is ever invented.
    """

    affinities: dict[
        str,
        float,
    ] = {}

    for tag in preference_tags(
        user_profile
    ):

        # Future-proof direct controlled tags.
        #
        # If a private profile contains:
        #
        #     "color"
        #
        # we can safely infer that COLOR matters
        # without assuming WHICH color.
        if tag in DIRECT_ATTRIBUTES:

            affinities[tag] = max(
                affinities.get(
                    tag,
                    0.0,
                ),
                1.0,
            )

        mapped = (
            PROFILE_TAG_TO_ATTRIBUTES.get(
                tag,
                {},
            )
        )

        for (
            attribute,
            weight,
        ) in mapped.items():

            affinities[
                attribute
            ] = max(
                affinities.get(
                    attribute,
                    0.0,
                ),
                float(
                    weight
                ),
            )

    return affinities


def affinity_for_attribute(
    user_profile: object,
    attribute: str,
) -> float:
    """
    Return profile support for one clarification
    dimension.
    """

    return attribute_affinity(
        user_profile
    ).get(
        attribute,
        0.0,
    )


def supporting_profile_tag(
    user_profile: object,
    attribute: str,
) -> str | None:
    """
    Return the strongest controlled profile tag
    supporting an attribute.

    This is used only for transparent
    customer-facing wording/debugging.
    """

    best_tag: str | None = None
    best_weight = 0.0

    for tag in preference_tags(
        user_profile
    ):

        if (
            tag == attribute
            and
            attribute in DIRECT_ATTRIBUTES
        ):

            weight = 1.0

        else:

            weight = (
                PROFILE_TAG_TO_ATTRIBUTES
                .get(
                    tag,
                    {},
                )
                .get(
                    attribute,
                    0.0,
                )
            )

        if weight > best_weight:

            best_weight = (
                weight
            )

            best_tag = (
                tag
            )

    return best_tag