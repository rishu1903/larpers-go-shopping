from __future__ import annotations

import numpy as np


# --------------------------------------------------
# OVERRIDE EXEMPLARS
# --------------------------------------------------
#
# Natural-language sentences expressing the same
# underlying concept as src/intent.py's
# REVERSAL_PATTERNS, but as full sentences rather
# than fixed phrases.
#
# This is the fallback path for reversals that don't
# contain any of the literal regex phrases (e.g. a
# private-set paraphrase). Coverage depends on how
# conceptually diverse this list is, not on matching
# it verbatim: a pretrained sentence embedding model
# places semantically similar sentences near each
# other even without shared vocabulary.
#
# This only detects an ANNOUNCED reversal (a cue
# phrase signaling "discard what I said before"). It
# cannot detect an implicit contradiction between
# turns with no such cue.

OVERRIDE_EXEMPLARS = (
    "Ignore my earlier preference.",
    "Please disregard what I said before.",
    "Forget what I said earlier.",
    "Scratch that, I want something different.",
    "Never mind my previous answer.",
    "I changed my mind about what I want.",
    "On second thought, let's go with something else.",
    "Actually, I don't want that anymore.",
    "Let's toss the earlier idea and go with something new.",
    "I take back what I said earlier.",
    "Here's what I actually need instead.",
    "Hold on, I want to change my answer.",
    "Let's forget the previous requirement.",
    "That's not what I want anymore, I'd rather have something else.",
    "I'd like to withdraw my earlier preference.",
    "Skip what I mentioned before, here's the real requirement.",
)


# --------------------------------------------------
# NON-OVERRIDE EXEMPLARS
# --------------------------------------------------
#
# Plain cosine similarity against OVERRIDE_EXEMPLARS
# alone does not work: empirically (see
# scripts/tune_override_threshold.py), ordinary
# shopping-preference sentences -- especially the
# evaluator's own "I don't have a preference for X" /
# "I don't have an additional preference for X"
# boilerplate -- score AS HIGH OR HIGHER than genuine
# overrides, because a general-purpose sentence
# embedding mostly captures shared topic ("this is
# about a shopping preference"), not the specific
# pragmatic act (asserting vs. declining vs.
# reversing a preference).
#
# The fix is to score by MARGIN against a contrasting
# negative set, not by an absolute threshold against
# positives alone: classify as override only when a
# message reads more like a reversal than like an
# ordinary preference statement or a decline-to-answer.

NON_OVERRIDE_EXEMPLARS = (
    "I don't have a preference for that.",
    "I don't have an additional preference for other.",
    "I don't have a preference for material.",
    "I prefer something waterproof.",
    "I would like something in leather.",
    "I'm still exploring options.",
    "I am still exploring jackets.",
    "My budget is around 100 dollars.",
    "Keep the price under $80.",
    "I need something comfortable.",
    "For that, what matters is: waterproof.",
    "A key requirement is: leather.",
    "Those options are not quite right yet.",
)


_DEFAULT_MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

# Tuned in scripts/tune_override_threshold.py: the
# margin (similarity to OVERRIDE_EXEMPLARS minus
# similarity to NON_OVERRIDE_EXEMPLARS).
#
# The cases that actually depend on this threshold are
# the ones with NO stage-1 regex match (this stage
# never runs otherwise): those scored 0.1067-0.1533.
# Every known negative scored below -0.09. 0.07 sits
# well inside that ~0.2 gap, biased toward the higher
# (more conservative) end, since a false positive
# silently discards evidence while a false negative
# just falls back to "not an override".
_DEFAULT_THRESHOLD = 0.07


def _encode(
    model,
    texts: list[str],
):

    return np.asarray(
        model.encode(
            texts,
            normalize_embeddings=True,
        ),
        dtype=np.float32,
    )


class OverrideSemanticDetector:
    """
    Local, offline semantic fallback for override
    detection.

    Loaded from a locally cached model rather than
    fetched from the Hugging Face Hub at runtime, so
    this never requires network access at inference
    time (see docs/submission_rules.md: official
    scoring may disable network access).

    Scores by MARGIN between two exemplar sets rather
    than by absolute similarity to positives alone --
    see the NON_OVERRIDE_EXEMPLARS comment above for
    why an absolute threshold does not work here.
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL_NAME,
        threshold: float = _DEFAULT_THRESHOLD,
        positive_exemplars: tuple[
            str, ...
        ] = OVERRIDE_EXEMPLARS,
        negative_exemplars: tuple[
            str, ...
        ] = NON_OVERRIDE_EXEMPLARS,
    ) -> None:

        # Imported here, not at module scope, so that
        # importing this module never requires the
        # (heavy, optional) sentence-transformers
        # dependency to be installed. Only constructing
        # a detector does.
        from sentence_transformers import (
            SentenceTransformer,
        )

        self.threshold = threshold

        self.model = SentenceTransformer(
            model_name,
        )

        self.positive_embeddings = _encode(
            self.model,
            list(
                positive_exemplars,
            ),
        )

        self.negative_embeddings = _encode(
            self.model,
            list(
                negative_exemplars,
            ),
        )

    def score(
        self,
        text: str,
    ) -> float:
        """
        Return how much more this text resembles a
        reversal than an ordinary preference
        statement: the highest similarity to any
        override exemplar, minus the highest
        similarity to any non-override exemplar.

        Positive means "more reversal-like",
        negative means "more ordinary-preference-like".
        """

        query = _encode(
            self.model,
            [
                text,
            ],
        )[0]

        positive_similarity = float(
            np.max(
                self.positive_embeddings
                @ query
            )
        )

        negative_similarity = float(
            np.max(
                self.negative_embeddings
                @ query
            )
        )

        return (
            positive_similarity
            - negative_similarity
        )

    def is_override(
        self,
        text: str,
    ) -> bool:

        return (
            self.score(
                text,
            )
            >= self.threshold
        )


# --------------------------------------------------
# LAZY SINGLETON
# --------------------------------------------------
#
# Constructing OverrideSemanticDetector loads a real
# model (slow, and only possible if the optional
# dependency + cached weights are present). Load it
# at most once, and cache a failure as None so a
# missing dependency doesn't retry (and re-raise) on
# every single turn.

_detector: OverrideSemanticDetector | None = None
_load_attempted = False


def get_detector() -> OverrideSemanticDetector | None:
    """
    Return the shared detector instance, or None if
    it could not be loaded (missing dependency,
    missing cached model files, or any other load
    error).

    Tests should monkeypatch this function directly
    to inject a stub detector rather than loading the
    real model.
    """

    global _detector
    global _load_attempted

    if _load_attempted:
        return _detector

    _load_attempted = True

    try:
        _detector = OverrideSemanticDetector()

    except Exception:

        _detector = None

    return _detector


def semantic_override(
    text: str,
) -> bool:
    """
    Return whether `text` semantically matches an
    override exemplar closely enough to be treated as
    a reversal.

    Fails closed: any missing dependency, missing
    model, or inference error results in False rather
    than raising, so this is always safe to call as a
    best-effort fallback.
    """

    detector = get_detector()

    if detector is None:
        return False

    try:
        return detector.is_override(
            text,
        )

    except Exception:

        return False
