from __future__ import annotations

from src.questions import (
    choose_candidate_attribute,
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


def choose_clarification(
    state: SessionState,
    turn: int,
    candidates: list[dict],
) -> tuple[
    str | None,
    str,
]:
    """
    Choose the next conversational action.
    """

    # No point asking a question on the
    # final allowed turn.
    if turn >= 10:

        return (
            None,
            (
                "Here are the closest "
                "matches I found."
            ),
        )

    # Once broad clarification is exhausted,
    # use remaining turns for exploration.
    if state.clarification_exhausted:

        return (
            None,
            (
                "I have enough information, "
                "so I'm showing you some "
                "different relevant alternatives."
            ),
        )

    attribute = (
        choose_candidate_attribute(
            state=state,
            candidates=candidates,
            turn=turn,
        )
    )

    return (
        attribute,
        QUESTION_TEXT[
            attribute
        ],
    )