from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from src import override_semantic

from src.intent import (
    ShoppingIntent,
    is_override,
)

from src.state import (
    SessionState,
)


# --------------------------------------------------
# SHARED SESSION SETUP
# --------------------------------------------------
#
# Every case starts from an identical two-turn
# history so that only the override message itself
# varies between cases.
#
#   Turn 1: category + a soon-to-be-stale preference
#   Turn 2: an additional preference that must
#           survive the override
#
# A fully correct override must, after the Turn 3
# message:
#
#   - classify the session as BUYING
#   - set override_seen
#   - remove the Turn-1 preference from active_text
#   - keep the Turn-2 preference in active_text
#   - add the new preference to active_text

_STALE_PREFERENCE = "black style"
_SURVIVING_PREFERENCE = "waterproof"
_NEW_PREFERENCE = "casual white sneakers"


def _run_override_case(
    override_message: str,
) -> SessionState:

    state = SessionState(
        user_profile={},
    )

    state.update(
        "I am looking for running shoes. "
        "black style",
        1,
    )

    state.update(
        "For that, what matters is: waterproof.",
        2,
    )

    state.update(
        override_message,
        3,
    )

    return state


def _override_fully_succeeded(
    state: SessionState,
) -> bool:

    active = state.active_text().lower()

    return (
        state.intent == ShoppingIntent.BUYING
        and state.override_seen
        and _STALE_PREFERENCE not in active
        and _SURVIVING_PREFERENCE in active
        and _NEW_PREFERENCE in active
    )


# --------------------------------------------------
# PARAPHRASE CORPUS
# --------------------------------------------------
#
# One entry per override phrasing. "exact_simulator"
# reproduces evaluator/local_evaluator.py line 85
# verbatim; every other entry is a meaning-preserving
# paraphrase (reverse an earlier preference, state a
# new concrete one) using different wording.
#
# All of these are now caught by the broadened
# REVERSAL_PATTERNS regex list in src/intent.py alone
# (no semantic model required). They stay here as a
# regression benchmark for that regex list.

_PARAPHRASE_CASES = (
    (
        "exact_simulator",
        "Actually, ignore my earlier preference. "
        f"What I need is: {_NEW_PREFERENCE}.",
    ),
    (
        "missing_actually",
        "Ignore my earlier preference. "
        f"What I need is: {_NEW_PREFERENCE}.",
    ),
    (
        "reordered_clauses",
        f"What I need is: {_NEW_PREFERENCE}. "
        "Please ignore my earlier preference, actually.",
    ),
    (
        "changed_my_mind",
        "I changed my mind, ignore my earlier "
        f"preference, what I need is: {_NEW_PREFERENCE}.",
    ),
    (
        "scratch_that",
        "Actually, scratch that - I really want "
        f"{_NEW_PREFERENCE} now.",
    ),
    (
        "forget_what_i_said",
        "Actually, forget what I said before, "
        f"I want {_NEW_PREFERENCE} instead.",
    ),
    (
        "on_second_thought",
        "On second thought, I need "
        f"{_NEW_PREFERENCE} instead of black.",
    ),
    (
        "synonym_disregard",
        "Actually, disregard my earlier preference. "
        f"What I need is: {_NEW_PREFERENCE}.",
    ),
    (
        "never_mind",
        "Never mind my earlier answer, "
        f"what I need is: {_NEW_PREFERENCE}.",
    ),
    (
        "ignore_what_i_said",
        "Actually, ignore what I said earlier. "
        f"I need {_NEW_PREFERENCE} instead.",
    ),
)


# --------------------------------------------------
# SEMANTIC-ONLY CORPUS
# --------------------------------------------------
#
# These share no trigger word with REVERSAL_PATTERNS
# or "what i need is", by design: they only pass
# through the semantic fallback stage
# (src/override_semantic.py), never the free regex
# stage. Used to validate the stage-1-then-stage-2
# wiring with a stub detector, and (opt-in) real
# model accuracy.

_SEMANTIC_ONLY_CASES = (
    (
        "toss_the_idea",
        "Let's toss the earlier idea, get me "
        f"{_NEW_PREFERENCE} instead.",
    ),
    (
        "withdraw",
        "I'd like to withdraw my earlier preference; "
        f"{_NEW_PREFERENCE} works better.",
    ),
    (
        "not_what_i_want_anymore",
        "That's not what I want anymore, I'd rather "
        f"have {_NEW_PREFERENCE}.",
    ),
)


# --------------------------------------------------
# ORDINARY (NON-OVERRIDE) NEGATIVE CONTROLS
# --------------------------------------------------
#
# Ordinary narrowing/buying/browsing language that
# must NEVER be treated as an override, regardless of
# how the regex list or semantic threshold evolve.

_NEGATIVE_CONTROLS = (
    "I prefer something waterproof.",
    "I'm still exploring options.",
    "under $80 please.",
    "My budget is around 100 dollars.",
    "I would like something in leather.",
)


class _StubDetector:
    """
    Deterministic stand-in for
    OverrideSemanticDetector, so tests never need the
    real (optional, heavy) dependency or a real model
    download.
    """

    def __init__(
        self,
        positive_texts: tuple[str, ...] = (),
        raise_on_call: bool = False,
    ) -> None:

        self._positive_texts = set(
            positive_texts,
        )

        self._raise_on_call = raise_on_call

    def is_override(
        self,
        text: str,
    ) -> bool:

        if self._raise_on_call:

            raise RuntimeError(
                "stub detector failure",
            )

        return text in self._positive_texts


class OverrideRobustnessTest(
    unittest.TestCase,
):

    def setUp(
        self,
    ) -> None:

        # Isolate the module-level singleton cache so
        # tests never leak a stub (or a real load
        # attempt) into each other or into unrelated
        # test files that exercise SessionState.update().
        self._saved_detector = (
            override_semantic._detector
        )

        self._saved_load_attempted = (
            override_semantic._load_attempted
        )

        override_semantic._detector = None
        override_semantic._load_attempted = True

    def tearDown(
        self,
    ) -> None:

        override_semantic._detector = (
            self._saved_detector
        )

        override_semantic._load_attempted = (
            self._saved_load_attempted
        )

    # ----------------------------------------------
    # REGEX-STAGE REGRESSION BENCHMARK
    # ----------------------------------------------
    #
    # With the singleton forced to None in setUp, the
    # semantic stage is a guaranteed no-op here, so
    # this class only exercises the free regex stage
    # in src/intent.py's REVERSAL_PATTERNS.

    def test_exact_simulator_phrasing_still_triggers_full_override(
        self,
    ) -> None:

        state = _run_override_case(
            "Actually, ignore my earlier preference. "
            f"What I need is: {_NEW_PREFERENCE}.",
        )

        self.assertTrue(
            _override_fully_succeeded(
                state,
            ),
        )

    def test_paraphrase_corpus_full_override_rate(
        self,
    ) -> None:

        successes = sum(
            _override_fully_succeeded(
                _run_override_case(
                    message,
                ),
            )
            for _, message in _PARAPHRASE_CASES
        )

        rate = successes / len(
            _PARAPHRASE_CASES,
        )

        # The broadened regex list now covers every
        # known paraphrase in this corpus without
        # needing the semantic fallback at all. If
        # this drops, a regex pattern regressed.
        self.assertEqual(
            rate,
            1.0,
        )

    def test_ordinary_narrowing_language_is_not_an_override(
        self,
    ) -> None:

        for message in _NEGATIVE_CONTROLS:

            with self.subTest(
                message=message,
            ):

                self.assertFalse(
                    is_override(
                        message,
                    ),
                )

    # ----------------------------------------------
    # SEMANTIC FALLBACK WIRING (STUBBED)
    # ----------------------------------------------
    #
    # These validate the stage-1-then-stage-2 control
    # flow and the fail-closed exception handling in
    # src/override_semantic.py, using a stub so they
    # never require the real dependency or a model
    # download.

    def test_semantic_fallback_catches_paraphrase_regex_missed(
        self,
    ) -> None:

        _, message = _SEMANTIC_ONLY_CASES[0]

        with patch.object(
            override_semantic,
            "get_detector",
            return_value=_StubDetector(
                positive_texts=(
                    message,
                ),
            ),
        ):

            state = _run_override_case(
                message,
            )

            self.assertTrue(
                _override_fully_succeeded(
                    state,
                ),
            )

    def test_semantic_fallback_not_needed_when_regex_already_matched(
        self,
    ) -> None:

        # The stub would reject everything, but the
        # regex stage should already have matched, so
        # the stub's answer must never matter here.

        with patch.object(
            override_semantic,
            "get_detector",
            return_value=_StubDetector(),
        ):

            state = _run_override_case(
                "Actually, scratch that - I really want "
                f"{_NEW_PREFERENCE} now.",
            )

            self.assertTrue(
                _override_fully_succeeded(
                    state,
                ),
            )

    def test_semantic_detector_exception_falls_back_to_false(
        self,
    ) -> None:

        with patch.object(
            override_semantic,
            "get_detector",
            return_value=_StubDetector(
                raise_on_call=True,
            ),
        ):

            self.assertFalse(
                override_semantic.semantic_override(
                    "anything at all",
                ),
            )

    def test_missing_dependency_makes_get_detector_return_none(
        self,
    ) -> None:

        override_semantic._load_attempted = False

        with patch.object(
            override_semantic,
            "OverrideSemanticDetector",
            side_effect=ImportError(
                "sentence-transformers not installed",
            ),
        ):

            self.assertIsNone(
                override_semantic.get_detector(),
            )

    def test_semantic_only_phrasing_is_not_an_override_without_a_detector(
        self,
    ) -> None:

        # With no detector available at all (the
        # default state after setUp), phrasing outside
        # REVERSAL_PATTERNS must not be treated as an
        # override -- it should fail closed, not
        # silently guess yes.

        _, message = _SEMANTIC_ONLY_CASES[0]

        state = _run_override_case(
            message,
        )

        self.assertFalse(
            _override_fully_succeeded(
                state,
            ),
        )


# --------------------------------------------------
# OPT-IN REAL MODEL VALIDATION
# --------------------------------------------------
#
# Loads the actual OverrideSemanticDetector (real
# dependency, real downloaded/cached model). Kept out
# of the default `python -m unittest -v` run since it
# is slow and requires an optional dependency; set
# RUN_REAL_OVERRIDE_MODEL_TESTS=1 to include it.

@unittest.skipUnless(
    os.environ.get(
        "RUN_REAL_OVERRIDE_MODEL_TESTS",
    )
    == "1",
    "set RUN_REAL_OVERRIDE_MODEL_TESTS=1 to test "
    "the real embedding model",
)
class RealOverrideSemanticModelTest(
    unittest.TestCase,
):

    @classmethod
    def setUpClass(
        cls,
    ) -> None:

        override_semantic._detector = None
        override_semantic._load_attempted = False

        cls.detector = (
            override_semantic.get_detector()
        )

        if cls.detector is None:

            raise unittest.SkipTest(
                "OverrideSemanticDetector could not "
                "be loaded (missing dependency or "
                "cached model files)",
            )

    def test_semantic_only_paraphrases_score_above_threshold(
        self,
    ) -> None:

        for label, message in _SEMANTIC_ONLY_CASES:

            with self.subTest(
                case=label,
            ):

                self.assertTrue(
                    self.detector.is_override(
                        message,
                    ),
                )

    def test_negative_controls_score_below_threshold(
        self,
    ) -> None:

        for message in _NEGATIVE_CONTROLS:

            with self.subTest(
                message=message,
            ):

                self.assertFalse(
                    self.detector.is_override(
                        message,
                    ),
                )

    def test_semantic_only_paraphrase_triggers_full_override_end_to_end(
        self,
    ) -> None:

        override_semantic._detector = self.detector
        override_semantic._load_attempted = True

        for _, message in _SEMANTIC_ONLY_CASES:

            with self.subTest(
                message=message,
            ):

                state = _run_override_case(
                    message,
                )

                self.assertTrue(
                    _override_fully_succeeded(
                        state,
                    ),
                )


if __name__ == "__main__":
    unittest.main()
