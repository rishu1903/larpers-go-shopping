from __future__ import annotations

from src.state import SessionState


def choose_clarification(
    state: SessionState,
    turn: int,
) -> tuple[str | None, str]:
    """
    Decide whether another clarification
    question is still useful.
    """

    # No benefit asking another question
    # on the final allowed turn.
    if turn >= 10:
        return (
            None,
            "Here are the closest matches I found.",
        )

    # The customer has explicitly indicated
    # that they have no additional preference.
    #
    # Stop repeating the same clarification
    # question and use subsequent turns to
    # explore alternative candidates instead.
    if state.clarification_exhausted:
        return (
            None,
            (
                "I have enough information, "
                "so I'm showing you some "
                "different relevant alternatives."
            ),
        )

    return (
        "other",
        (
            "What else matters most to you "
            "for this purchase?"
        ),
    )