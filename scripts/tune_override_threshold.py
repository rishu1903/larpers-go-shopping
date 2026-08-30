"""
Empirically choose the override-detection semantic
similarity threshold.

Loads the real OverrideSemanticDetector (bypassing its
configured threshold) and scores:

  - POSITIVES: paraphrases that should be treated as an
    override, pulled from
    tests/test_override_robustness.py
  - NEGATIVES: ordinary narrowing/buying/browsing/
    no-preference sentences that must NOT be treated as
    an override, pulled from the existing test suite

Prints every score and recommends the smallest threshold
that separates every known positive from every known
negative (with a small safety margin toward the
negatives, since a false positive silently discards
evidence while a false negative just falls back to
"not an override").

Run from the repository root:

    python -m scripts.tune_override_threshold
"""

from __future__ import annotations

from src.override_semantic import OverrideSemanticDetector

from tests.test_override_robustness import (
    _NEW_PREFERENCE,
    _PARAPHRASE_CASES,
    _SEMANTIC_ONLY_CASES,
)


POSITIVES = tuple(
    message for _, message in (
        *_PARAPHRASE_CASES,
        *_SEMANTIC_ONLY_CASES,
    )
)


NEGATIVES = (
    # Existing negative controls.
    "I prefer something waterproof.",
    "I'm still exploring options.",
    "under $80 please.",
    "My budget is around 100 dollars.",
    "I would like something in leather.",
    # The actual false positive found by the real
    # model at threshold 0.5: "no preference"
    # boilerplate the evaluator itself emits.
    "I don't have a preference for material.",
    "I don't have an additional preference for other.",
    "I don't have an additional preference for color.",
    # Other simulator/dialogue phrasing that must not
    # be mistaken for a reversal.
    "I am still exploring jackets",
    "Those options are not quite right yet.",
    "I would prefer something waterproof",
    "I'm looking for Shoes. A key requirement is: leather.",
    "For that, what matters is: fits up to 8-inch wrist circumference.",
    "Keep the price under $80.",
    f"I need {_NEW_PREFERENCE}.",
    f"I'd like {_NEW_PREFERENCE}.",
)


def main() -> None:

    detector = OverrideSemanticDetector(
        threshold=0.0,
    )

    positive_scores = [
        (message, detector.score(message))
        for message in POSITIVES
    ]

    negative_scores = [
        (message, detector.score(message))
        for message in NEGATIVES
    ]

    print("POSITIVES")
    for message, score in sorted(
        positive_scores,
        key=lambda item: item[1],
    ):
        print(f"  {score:.4f}  {message}")

    print()
    print("NEGATIVES")
    for message, score in sorted(
        negative_scores,
        key=lambda item: item[1],
        reverse=True,
    ):
        print(f"  {score:.4f}  {message}")

    min_positive = min(score for _, score in positive_scores)
    max_negative = max(score for _, score in negative_scores)

    print()
    print(f"min positive score: {min_positive:.4f}")
    print(f"max negative score: {max_negative:.4f}")

    if max_negative >= min_positive:
        print(
            "NO CLEAN SEPARATION -- positives and "
            "negatives overlap. Add more exemplars or "
            "reconsider this approach for the "
            "overlapping cases."
        )
        return

    # Bias toward the negative side: a missed override
    # just falls back to "not an override" (same as
    # today), but a false positive silently discards
    # evidence, which is worse.
    threshold = max_negative + 0.4 * (
        min_positive - max_negative
    )

    print(f"recommended threshold: {threshold:.4f}")


if __name__ == "__main__":
    main()
