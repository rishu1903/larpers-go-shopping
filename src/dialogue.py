from __future__ import annotations

from src.profile import (
    supporting_profile_tag,
)

from src.questions import (
    choose_candidate_attribute,
    choose_differentiating_attribute,
)

from src.state import (
    SessionState,
)


QUESTION_TEXT = {
    "material":
        "Do you have a preferred material?",

    "color":
        "Do you have a preferred color?",

    "size":
        (
            "Are there any sizing or fit "
            "requirements I should prioritize?"
        ),

    "style":
        "What style or fit do you prefer?",

    "use_case":
        "What will you mainly use it for?",

    "feature":
        (
            "Is there a specific feature "
            "that matters most?"
        ),

    "other":
        (
            "What else matters most to you "
            "for this purchase?"
        ),
}


PROFILE_QUESTION_TEXT = {
    "material":
        (
            "Material tends to matter in your "
            "shopping preferences. Do you have "
            "a preferred material for this item?"
        ),

    "size":
        (
            "Fit tends to matter in your shopping "
            "preferences. Are there any sizing or "
            "fit requirements I should prioritize?"
        ),

    "style":
        (
            "Style or fit tends to matter in your "
            "shopping preferences. What style or "
            "fit do you prefer here?"
        ),

    "use_case":
        (
            "How an item performs for its intended "
            "use tends to matter in your preferences. "
            "What will you mainly use it for?"
        ),

    "feature":
        (
            "Product qualities tend to matter in "
            "your shopping preferences. Is there a "
            "specific feature I should prioritize?"
        ),
}


def _question_message(
    state: SessionState,
    attribute: str,
    turn: int,
) -> str:
    """
    Create customer-facing clarification text.

    Early broad discovery remains unchanged.

    Later questions may acknowledge an anonymized
    aggregate preference dimension, but never invent
    a concrete preference value.
    """

    if (
        turn <= 3
        or
        attribute == "other"
    ):

        return QUESTION_TEXT[
            attribute
        ]

    profile_tag = (
        supporting_profile_tag(
            state.user_profile,
            attribute,
        )
    )

    if (
        profile_tag is not None
        and
        attribute
        in PROFILE_QUESTION_TEXT
    ):

        return PROFILE_QUESTION_TEXT[
            attribute
        ]

    return QUESTION_TEXT[
        attribute
    ]


def choose_clarification(
    state: SessionState,
    turn: int,
    candidates: list[dict],
    tied_pair: list[dict] | None = None,
) -> tuple[
    str | None,
    str,
]:
    """
    Choose the next conversational action.

    When tied_pair is provided, prefer an attribute
    that differentiates the two tied candidates.
    Falls through to choose_candidate_attribute if
    no asymmetric attribute exists.
    """

    if turn >= 10:

        return (
            None,
            (
                "Here are the closest "
                "matches I found."
            ),
        )

    if state.clarification_exhausted:

        return (
            None,
            (
                "I have enough information, "
                "so I'm showing you some "
                "different relevant alternatives."
            ),
        )

    if tied_pair is not None and len(tied_pair) == 2:
        attribute = choose_differentiating_attribute(
            state=state,
            candidate_a=tied_pair[0],
            candidate_b=tied_pair[1],
            turn=turn,
        )
        if attribute == "other":
            attribute = choose_candidate_attribute(
                state=state,
                candidates=candidates,
                turn=turn,
            )
    else:
        attribute = choose_candidate_attribute(
            state=state,
            candidates=candidates,
            turn=turn,
        )

    return (
        attribute,
        _question_message(
            state=state,
            attribute=attribute,
            turn=turn,
        ),
    )