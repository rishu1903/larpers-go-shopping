from __future__ import annotations

from dataclasses import dataclass
import math

from src.state import SessionState


V12_PRECISION_LEXICAL_LIMIT = 100
V12_PRECISION_SEMANTIC_LIMIT = 100

V12_EXPLORATION_LEXICAL_LIMIT = 500
V12_EXPLORATION_SEMANTIC_LIMIT = 250


# V13 production configuration.
#
# Failure-aware protected recovery is enabled only
# after an explicit recommendation failure during
# exploration.
#
# Selected through the label-free V13 protected
# recovery ablation:
#
#     depth_step = 0.75
#     recovery_slots = 1
#
# This was the smallest tested configuration that:
#
# - rescued two additional V12 residual cases;
# - preserved every V12 continuation rescue;
# - increased cumulative shadow Hit@10;
# - improved session efficiency; and
# - avoided allowing deeper retrieval to replace
#   the proven V12 recommendation path wholesale.
FAILURE_AWARE_ORCHESTRATION_ENABLED = True


# Each observed recommendation failure expands the
# exploration candidate depth by this proportion.
FAILURE_DEPTH_STEP = 0.75


# Preserve almost all of the proven V12 continuation
# while allowing one genuinely new deep-recovery
# candidate into the final Top-K.
#
# With top_k=10:
#
#     9 V12 continuation candidates
#     +
#     1 recovery-only candidate
FAILURE_RECOVERY_SLOTS = 1


# Bound cost growth even if a session repeatedly misses.
MAX_FAILURE_LEVEL = 2


@dataclass(frozen=True)
class RetrievalPlan:
    lexical_limit: int
    semantic_limit: int
    strategy: str
    failure_level: int


def configure_failure_orchestration(
    *,
    enabled: bool,
    depth_step: float | None = None,
    recovery_slots: int | None = None,
) -> None:
    """
    Configure V13 failure-aware routing.

    Production uses:

        enabled=True
        depth_step=0.75
        recovery_slots=1

    The configuration function remains available
    for reproducible offline ablations.
    """

    global FAILURE_AWARE_ORCHESTRATION_ENABLED
    global FAILURE_DEPTH_STEP
    global FAILURE_RECOVERY_SLOTS

    if depth_step is not None:

        if (
            not math.isfinite(
                depth_step
            )
            or
            depth_step < 0.0
        ):
            raise ValueError(
                "depth_step must be a finite "
                "non-negative number"
            )

        FAILURE_DEPTH_STEP = float(
            depth_step
        )

    if recovery_slots is not None:

        if (
            isinstance(
                recovery_slots,
                bool,
            )
            or
            not isinstance(
                recovery_slots,
                int,
            )
            or
            recovery_slots < 0
        ):
            raise ValueError(
                "recovery_slots must be a "
                "non-negative integer"
            )

        FAILURE_RECOVERY_SLOTS = (
            recovery_slots
        )

    FAILURE_AWARE_ORCHESTRATION_ENABLED = (
        bool(
            enabled
        )
    )


def v12_retrieval_plan(
    exploration: bool,
) -> RetrievalPlan:
    """
    Return the frozen V12 retrieval plan.

    V13 explicitly retains this plan so deeper
    recovery can never silently replace the proven
    V12 continuation path.
    """

    if not exploration:

        return RetrievalPlan(
            lexical_limit=(
                V12_PRECISION_LEXICAL_LIMIT
            ),
            semantic_limit=(
                V12_PRECISION_SEMANTIC_LIMIT
            ),
            strategy="precision",
            failure_level=0,
        )

    return RetrievalPlan(
        lexical_limit=(
            V12_EXPLORATION_LEXICAL_LIMIT
        ),
        semantic_limit=(
            V12_EXPLORATION_SEMANTIC_LIMIT
        ),
        strategy="exploration_v12",
        failure_level=0,
    )


def retrieval_plan(
    state: SessionState,
    exploration: bool,
) -> RetrievalPlan:
    """
    Build the runtime retrieval plan.

    Precision
    =========

    Always preserve V12:

        lexical Top 100

    Exploration
    ===========

    Start from V12:

        lexical Top 500
            +
        semantic Top 250

    After explicit recommendation failures,
    V13 progressively expands retrieval depth.

    Failure level 1:

        lexical Top 875
        semantic Top 438

    Failure level 2:

        lexical Top 1250
        semantic Top 625

    The failure level is capped to keep runtime
    and memory growth bounded.
    """

    baseline = (
        v12_retrieval_plan(
            exploration
        )
    )

    if (
        not exploration
        or
        not FAILURE_AWARE_ORCHESTRATION_ENABLED
    ):
        return baseline

    failure_level = min(
        max(
            int(
                state.miss_streak
            ),
            0,
        ),
        MAX_FAILURE_LEVEL,
    )

    if failure_level <= 0:
        return baseline

    factor = (
        1.0
        +
        FAILURE_DEPTH_STEP
        * failure_level
    )

    return RetrievalPlan(
        lexical_limit=int(
            round(
                baseline.lexical_limit
                * factor
            )
        ),
        semantic_limit=int(
            round(
                baseline.semantic_limit
                * factor
            )
        ),
        strategy=(
            "failure_recovery_"
            f"{failure_level}"
        ),
        failure_level=(
            failure_level
        ),
    )


def should_use_protected_recovery(
    state: SessionState,
    exploration: bool,
) -> bool:
    """
    Activate protected recovery only after an
    explicit same-intent recommendation failure.

    Initial buying, browsing and first-exploration
    behaviour remain identical to V12.
    """

    return (
        FAILURE_AWARE_ORCHESTRATION_ENABLED

        and

        exploration

        and

        state.miss_streak > 0

        and

        FAILURE_RECOVERY_SLOTS > 0
    )


def select_protected_recovery(
    baseline_ranked: list[dict],
    expanded_ranked: list[dict],
    top_k: int,
) -> list[dict]:
    """
    Preserve the proven V12 continuation while
    providing a bounded slot for deep recovery.

    With top_k=10 and recovery_slots=1:

        V12 ranks 1-9
            +
        highest-ranked product found only by the
        expanded recovery candidate pool

    Products already present anywhere within the
    V12 candidate universe are not treated as
    recovery candidates.

    If deeper retrieval contributes nothing new,
    the unused recovery slot falls back to V12.
    """

    if top_k <= 0:
        return []

    recovery_slots = min(
        FAILURE_RECOVERY_SLOTS,
        top_k,
    )

    baseline_quota = (
        top_k
        -
        recovery_slots
    )

    baseline_asins = {
        str(
            candidate.get(
                "parent_asin",
                "",
            )
        )

        for candidate
        in baseline_ranked
    }

    selected: list[
        dict
    ] = []

    selected_asins: set[
        str
    ] = set()

    def append_candidates(
        candidates: list[dict],
        limit: int | None = None,
    ) -> None:

        added = 0

        for candidate in candidates:

            asin = str(
                candidate.get(
                    "parent_asin",
                    "",
                )
            )

            if (
                not asin
                or
                asin in selected_asins
            ):
                continue

            selected.append(
                candidate
            )

            selected_asins.add(
                asin
            )

            added += 1

            if (
                len(selected)
                >= top_k
            ):
                break

            if (
                limit is not None
                and
                added >= limit
            ):
                break

    # Preserve the V12 recommendation prefix.
    append_candidates(
        baseline_ranked,
        baseline_quota,
    )

    # Only products genuinely absent from the
    # complete V12 candidate pool qualify for the
    # protected recovery slot.
    recovery_only = [
        candidate

        for candidate
        in expanded_ranked

        if str(
            candidate.get(
                "parent_asin",
                "",
            )
        )
        not in baseline_asins
    ]

    append_candidates(
        recovery_only,
        recovery_slots,
    )

    # If deeper retrieval contributed no suitable
    # new candidate, fill the remaining capacity
    # from V12.
    append_candidates(
        baseline_ranked
    )

    # Final sparse-pool safeguard.
    append_candidates(
        expanded_ranked
    )

    return selected[
        :top_k
    ]