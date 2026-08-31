from __future__ import annotations

from src.state import SessionState


# ------------------------------------------------------
# CONFIDENCE-GATED SHORTLIST WIDTH
# ------------------------------------------------------
#
# The scorer freezes the target's rank at the FIRST turn
# it appears in our list and ends the session there. A
# mediocre early placement is therefore permanent: the
# remaining turns are discarded.
#
# The metric prices the two sides of that trade very
# differently:
#
#     one extra turn      0.20 / 10  = 0.02
#     rank 2 -> rank 1    0.30 x 0.5 = 0.15
#
# So a turn is roughly seven times cheaper than settling
# for second place.
#
# When our top candidate outscores the runner-up we are
# making a real choice, and committing to it is honest.
# When they score identically we are not choosing at all
# -- `rerank_candidates` breaks the tie on popularity,
# which is a coin flip. Those ties are the majority of
# our non-first-place finishes.
#
# In that situation the useful move is to keep asking
# rather than to guess: one more disclosed constraint
# usually separates the tied block, and the constraint
# costs a fraction of what the coin flip does.
#
# The shortlist is therefore a question, not a verdict.
# We narrow only while a wrong answer is still cheap to
# recover from, and we widen to the full list as soon as
# that stops being true.


# Turn at which we stop narrowing. Deferral only pays
# while turns remain to recover in, and an intent
# override cannot be scored before turn 3 anyway, so
# narrowing past this point buys nothing.
WIDEN_TURN = 3


def shortlist_width(
    ranked: list[dict],
    state: SessionState,
    turn: int,
    top_k: int,
) -> int:
    """
    Number of products to show this turn.

    Returns 1 when we hold a genuinely separated best
    candidate and can still recover from being wrong,
    and `top_k` in every other case.
    """

    # Past the recovery window there is no upside left
    # in holding results back.
    if turn >= WIDEN_TURN:
        return top_k

    # Exploration means the shopper has told us they
    # have nothing further to add. No further constraint
    # is coming, so waiting cannot break a tie.
    if state.clarification_exhausted:
        return top_k

    # Nothing to be confident about.
    if len(ranked) < 2:
        return top_k

    leader = ranked[0].get(
        "relevance"
    )

    if leader is None:
        return top_k

    # A leader that matched nothing is not a leader.
    # This is the weak-retrieval case: we have no
    # candidate worth leading with, the shortlist would
    # be a guess rather than a proposal, and holding the
    # field back risks the hit rate for no gain.
    if leader <= 0.0:
        return top_k

    # Note that a tie between the leader and the runner
    # up is NOT a reason to widen. A tie means the order
    # was settled by the popularity fallback rather than
    # by anything the shopper asked for, so showing the
    # field would lock in a rank we did not actually
    # choose. Those are the turns where one more
    # disclosed constraint is worth the most, and where
    # leading with a proposal and a question beats
    # presenting a shelf.

    return 1
