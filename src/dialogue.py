from __future__ import annotations

from src.state import SessionState


def choose_clarification(
    state: SessionState,
    turn: int,
) -> tuple[str | None, str]:

    # There is no point asking another question
    # on the final allowed turn.
    if turn >= 10:
        return (
            None,
            "Here are the closest matches I found.",
        )

    # Version 1 uses the broad `other`
    # clarification mechanism provided
    # by the official evaluator.
    #
    # Later we will replace this with
    # candidate-aware question selection.
    return (
        "other",
        "What else matters most to you for this purchase?",
    )