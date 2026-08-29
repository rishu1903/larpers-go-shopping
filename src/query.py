from __future__ import annotations

from src.state import SessionState


def build_search_query(
    state: SessionState,
) -> str:
    """
    Build the current retrieval query from
    all active conversational evidence.
    """

    return state.active_text()