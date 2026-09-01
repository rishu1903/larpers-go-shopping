"""Domain-synonym expansion for retrieval query text.

Both the lexical (BM25) and semantic routes match query text against
catalogue vocabulary. A shopper's paraphrase ("something that resists
moisture", "suitable for wet conditions") often shares no words with
the catalogue's own adjective-style listing text ("Waterproof",
"Water-Resistant"). This module appends a small set of canonical
catalogue-style terms to the query whenever a generic trigger phrase
for that concept is present, purely additively -- the original text is
never altered or removed, only extended.

Seeded with one evidence-based cluster: moisture-resistance concepts
(waterproof/water-resistant) were found to be the single largest
concentrated source of residual retrieval misses in the concept
robustness and end-to-end shadow benchmarks (6 of 14 miss_both cases).
Trigger phrases are short, generic domain fragments -- not copied
verbatim from any specific benchmark case -- so the expansion
generalizes to phrasing the benchmark itself doesn't generate.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SynonymGroup:
    """A cluster of interchangeable domain vocabulary.

    ``trigger_phrases`` are the (broad, generic) phrases that activate
    the group. ``canonical_terms`` are the short, catalogue-realistic
    terms appended to the query when triggered.
    """

    trigger_phrases: frozenset[str]
    canonical_terms: frozenset[str]


_SYNONYM_GROUPS: tuple[SynonymGroup, ...] = (
    SynonymGroup(
        trigger_phrases=frozenset(
            {
                "waterproof",
                "water resistant",
                "water-resistant",
                "water repellent",
                "water-repellent",
                "weatherproof",
                "rainproof",
                "resists moisture",
                "moisture resistant",
                "moisture-resistant",
                "repels water",
                "resist water",
                "block water",
                "water getting in",
                "keeps water out",
                "heavy rain",
                "wet weather",
                "wet conditions",
                "rainy conditions",
                "damp conditions",
                "protects against rain",
            }
        ),
        canonical_terms=frozenset(
            {
                "waterproof",
                "water resistant",
                "water repellent",
                "weatherproof",
                "rainproof",
                "moisture resistant",
            }
        ),
    ),
)


def expand_query_text(text: str) -> str:
    """Return ``text`` with matched synonym-group terms appended.

    Purely additive and idempotent: the original text is always
    preserved verbatim, and only canonical terms not already present
    (case-insensitively) are appended.
    """

    lowered = text.lower()

    additions: list[str] = []

    for group in _SYNONYM_GROUPS:

        if not any(
            phrase in lowered
            for phrase in group.trigger_phrases
        ):
            continue

        for term in sorted(group.canonical_terms):

            if (
                term not in lowered
                and term not in additions
            ):
                additions.append(term)

    if not additions:
        return text

    return text + " " + " ".join(additions)
