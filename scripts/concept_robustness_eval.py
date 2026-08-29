from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import re
from typing import Iterable

from starter.agent import Agent
from src.retrieval import retrieve_candidates
from src.state import Evidence, SessionState


# ==================================================
# CONCEPT SPECIFICATION
# ==================================================


@dataclass(frozen=True)
class ConceptSpec:
    name: str

    # Patterns that establish that a catalogue
    # product explicitly possesses the concept.
    positive_patterns: tuple[str, ...]

    # Patterns forbidden from the generated query.
    # This ensures that we really are testing
    # paraphrase/generalisation rather than exact
    # keyword retrieval.
    leakage_patterns: tuple[str, ...]

    paraphrases: tuple[str, ...]

    # These patterns are matched ONLY against
    # specific category path components after
    # generic Amazon ancestors such as:
    #
    #     Clothing, Shoes & Jewelry
    #     Men
    #     Women
    #
    # have been removed.
    allowed_category_patterns: tuple[str, ...]

    excluded_category_patterns: tuple[str, ...] = ()


# ==================================================
# CATEGORY DOMAIN PATTERNS
# ==================================================


FOOTWEAR = (
    r"\bshoes?\b",
    r"\bboots?\b",
    r"\bsneakers?\b",
    r"\bslippers?\b",
    r"\bsandals?\b",
    r"\bclogs?\b",
    r"\bloafers?\b",
    r"\boxfords?\b",
    r"\bmules?\b",
    r"\bfootwear\b",
)


OUTERWEAR = (
    r"\bjackets?\b",
    r"\bcoats?\b",
    r"\bvests?\b",
    r"\banoraks?\b",
    r"\braincoats?\b",
    r"\bwindbreakers?\b",
    r"\bparkas?\b",
    r"\bfleece\b",
)


APPAREL = (
    r"\bshirts?\b",
    r"\bt[- ]?shirts?\b",
    r"\btees?\b",
    r"\btops?\b",
    r"\bblouses?\b",
    r"\bpants?\b",
    r"\btrousers?\b",
    r"\bleggings?\b",
    r"\bshorts?\b",
    r"\bdresses?\b",
    r"\bskirts?\b",
    r"\bsweaters?\b",
    r"\bsweatshirts?\b",
    r"\bhoodies?\b",
    r"\bpolos?\b",
    r"\bjeans?\b",
    r"\bjumpsuits?\b",
    r"\brompers?\b",
    r"\bbras?\b",
    r"\bpanties\b",
    r"\bunderwear\b",
    r"\bsocks?\b",
)


BAGS = (
    r"\bbackpacks?\b",
    r"\bhandbags?\b",
    r"\bshoulder bags?\b",
    r"\btote bags?\b",
    r"\bduffels?\b",
    r"\btravel bags?\b",
    r"\bpurses?\b",
)


WATCHES = (
    r"\bwatches?\b",
    r"\bwrist watches?\b",
)


HEADWEAR = (
    r"\bhats?\b",
    r"\bcaps?\b",
    r"\bbeanies?\b",
    r"\bheadbands?\b",
)


HANDWEAR = (
    r"\bgloves?\b",
    r"\bmittens?\b",
)


SCARVES = (
    r"\bscarves?\b",
    r"\bwraps?\b",
    r"\bpashminas?\b",
)


BELTS = (
    r"\bbelts?\b",
)


BRACELETS = (
    r"\bbracelets?\b",
)


# ==================================================
# CONCEPT DEFINITIONS
# ==================================================


CONCEPTS = (
    ConceptSpec(
        name="waterproof",

        positive_patterns=(
            r"\bwaterproof\b",
        ),

        leakage_patterns=(
            r"\bwaterproof\b",
        ),

        paraphrases=(
            (
                "something that can handle "
                "heavy rain without water "
                "getting in"
            ),
            (
                "something designed to keep "
                "water from getting through"
            ),
            (
                "something suitable for very "
                "wet conditions without "
                "moisture getting inside"
            ),
        ),

        allowed_category_patterns=(
            FOOTWEAR
            + OUTERWEAR
            + BAGS
        ),
    ),

    ConceptSpec(
        name="water_resistant",

        positive_patterns=(
            r"\bwater[- ]resistant\b",
            r"\bwater resistance\b",
        ),

        leakage_patterns=(
            r"\bwater[- ]resistant\b",
            r"\bwater resistance\b",
        ),

        paraphrases=(
            (
                "something that can handle "
                "splashes and light rain"
            ),
            (
                "something that resists "
                "moisture during everyday use"
            ),
            (
                "something suitable for "
                "occasional wet conditions"
            ),
        ),

        allowed_category_patterns=(
            FOOTWEAR
            + OUTERWEAR
            + BAGS
            + WATCHES
            + HEADWEAR
        ),
    ),

    ConceptSpec(
        name="non_slip",

        positive_patterns=(
            r"\bnon[- ]?slip\b",
            r"\bnonslip\b",
            r"\bslip[- ]resistant\b",
            r"\banti[- ]slip\b",
        ),

        leakage_patterns=(
            r"\bnon[- ]?slip\b",
            r"\bnonslip\b",
            r"\bslip[- ]resistant\b",
            r"\banti[- ]slip\b",
        ),

        paraphrases=(
            (
                "something with secure "
                "traction on slick surfaces"
            ),
            (
                "something that grips well "
                "on smooth or wet floors"
            ),
            (
                "something designed to "
                "reduce sliding underfoot"
            ),
        ),

        allowed_category_patterns=(
            FOOTWEAR
        ),
    ),

    ConceptSpec(
        name="breathable",

        positive_patterns=(
            r"\bbreathable\b",
        ),

        leakage_patterns=(
            r"\bbreathable\b",
        ),

        paraphrases=(
            (
                "something that allows "
                "airflow while being worn"
            ),
            (
                "something that helps reduce "
                "heat buildup during wear"
            ),
            (
                "something ventilated enough "
                "for warm conditions"
            ),
        ),

        allowed_category_patterns=(
            FOOTWEAR
            + OUTERWEAR
            + APPAREL
        ),
    ),

    ConceptSpec(
        name="lightweight",

        positive_patterns=(
            r"\blightweight\b",
            r"\blight[- ]weight\b",
        ),

        leakage_patterns=(
            r"\blightweight\b",
            r"\blight[- ]weight\b",
        ),

        paraphrases=(
            (
                "something that does not feel "
                "heavy during use"
            ),
            (
                "something easy to wear or "
                "carry for long periods"
            ),
            (
                "something with minimal weight "
                "for everyday use"
            ),
        ),

        allowed_category_patterns=(
            FOOTWEAR
            + OUTERWEAR
            + APPAREL
            + BAGS
        ),
    ),

    ConceptSpec(
        name="insulated",

        positive_patterns=(
            r"\binsulated\b",
            r"\binsulation\b",
        ),

        leakage_patterns=(
            r"\binsulated\b",
            r"\binsulation\b",
        ),

        paraphrases=(
            (
                "something that helps retain "
                "warmth in cold conditions"
            ),
            (
                "something designed to reduce "
                "heat loss in the cold"
            ),
            (
                "something that provides extra "
                "warmth for low temperatures"
            ),
        ),

        allowed_category_patterns=(
            OUTERWEAR
            + (
                r"\bboots?\b",
            )
            + HANDWEAR
            + HEADWEAR
            + SCARVES
        ),
    ),

    ConceptSpec(
        name="stretch",

        positive_patterns=(
            r"\bstretch\b",
            r"\bstretchy\b",
        ),

        leakage_patterns=(
            r"\bstretch\b",
            r"\bstretchy\b",
        ),

        paraphrases=(
            (
                "something that moves "
                "comfortably with the body"
            ),
            (
                "something with flexibility "
                "when I move"
            ),
            (
                "something that gives a little "
                "instead of feeling rigid"
            ),
        ),

        allowed_category_patterns=(
            OUTERWEAR
            + APPAREL
            + HANDWEAR
        ),
    ),

    ConceptSpec(
        name="adjustable",

        positive_patterns=(
            r"\badjustable\b",
        ),

        leakage_patterns=(
            r"\badjustable\b",
        ),

        paraphrases=(
            (
                "something whose fit can be "
                "changed as needed"
            ),
            (
                "something I can tighten or "
                "loosen for a better fit"
            ),
            (
                "something with a fit that "
                "can be fine-tuned"
            ),
        ),

        allowed_category_patterns=(
            FOOTWEAR
            + BAGS
            + HEADWEAR
            + BELTS
            + BRACELETS
            + WATCHES
        ),
    ),

    ConceptSpec(
        name="pockets",

        positive_patterns=(
            r"\bpockets?\b",
        ),

        leakage_patterns=(
            r"\bpockets?\b",
        ),

        paraphrases=(
            (
                "something with built-in "
                "storage for small essentials"
            ),
            (
                "something with places to "
                "keep a phone or other "
                "small items"
            ),
            (
                "something that lets me carry "
                "small belongings without a "
                "separate pouch"
            ),
        ),

        allowed_category_patterns=(
            OUTERWEAR
            + APPAREL
            + BAGS
        ),
    ),

    ConceptSpec(
        name="hooded",

        positive_patterns=(
            r"\bhooded\b",
            r"\bhood\b",
        ),

        leakage_patterns=(
            r"\bhood(?:ed|ie|ies)?\b",
        ),

        paraphrases=(
            (
                "something with built-in "
                "head coverage"
            ),
            (
                "something that can cover my "
                "head when the weather turns bad"
            ),
            (
                "something with an attached "
                "covering for the head"
            ),
        ),

        allowed_category_patterns=(
            OUTERWEAR
            + (
                r"\bhoodies?\b",
                r"\bsweatshirts?\b",
            )
        ),
    ),

    ConceptSpec(
        name="hiking",

        positive_patterns=(
            r"\bhiking\b",
        ),

        leakage_patterns=(
            r"\bhiking\b",
        ),

        paraphrases=(
            (
                "something suitable for "
                "trails and uneven terrain"
            ),
            (
                "something intended for long "
                "walks on outdoor trails"
            ),
            (
                "something appropriate for "
                "rough paths and trail use"
            ),
        ),

        allowed_category_patterns=(
            FOOTWEAR
            + OUTERWEAR
            + APPAREL
            + BAGS
            + HEADWEAR
        ),
    ),

    ConceptSpec(
        name="running",

        positive_patterns=(
            r"\brunning\b",
        ),

        leakage_patterns=(
            r"\brunning\b",
        ),

        paraphrases=(
            (
                "something suitable for "
                "jogging and training"
            ),
            (
                "something intended for "
                "repeated fast-paced exercise"
            ),
            (
                "something appropriate for "
                "regular jogs and workouts"
            ),
        ),

        allowed_category_patterns=(
            FOOTWEAR
            + OUTERWEAR
            + (
                r"\bshirts?\b",
                r"\bt[- ]?shirts?\b",
                r"\btees?\b",
                r"\btops?\b",
                r"\bshorts?\b",
                r"\bpants?\b",
                r"\bleggings?\b",
                r"\bsocks?\b",
            )
            + HEADWEAR
            + BELTS
        ),

        excluded_category_patterns=(
            r"\bcycling\b",
            r"\blingerie\b",
            r"\bcorsets?\b",
            r"\bbustiers?\b",
        ),
    ),

    ConceptSpec(
        name="winter",

        positive_patterns=(
            r"\bwinter\b",
        ),

        leakage_patterns=(
            r"\bwinter\b",
        ),

        paraphrases=(
            (
                "something suitable for "
                "cold-weather use"
            ),
            (
                "something intended for low "
                "temperatures and chilly "
                "conditions"
            ),
            (
                "something appropriate when "
                "the weather gets very cold"
            ),
        ),

        allowed_category_patterns=(
            OUTERWEAR
            + (
                r"\bboots?\b",
            )
            + HANDWEAR
            + HEADWEAR
            + SCARVES
        ),

        excluded_category_patterns=(
            r"\bsandals?\b",
            r"\bflip[- ]?flops?\b",
        ),
    ),
)


# ==================================================
# GENERIC AMAZON CATEGORY COMPONENTS
# ==================================================


GENERIC_CATEGORY_COMPONENTS = {
    "clothing shoes and jewelry",
    "clothing",
    "shoes",
    "jewelry",
    "men",
    "women",
    "boys",
    "girls",
    "baby",
    "novelty",
    "novelty and more",
}


def _flatten(
    value: object,
) -> str:

    if value is None:
        return ""

    if isinstance(
        value,
        dict,
    ):

        return " ".join(
            f"{key} {_flatten(item)}"
            for key, item
            in value.items()
        )

    if isinstance(
        value,
        list,
    ):

        return " ".join(
            _flatten(item)
            for item
            in value
        )

    return str(value)


def _display_text(
    value: object,
) -> str:

    return re.sub(
        r"\s+",
        " ",
        str(
            value
            or ""
        ),
    ).strip()


def _normalized_category(
    value: object,
) -> str:

    text = (
        _display_text(
            value
        )
        .lower()
        .replace(
            "&",
            " and ",
        )
    )

    text = re.sub(
        r"[^a-z0-9]+",
        " ",
        text,
    )

    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


def product_matching_text(
    product: dict,
) -> str:

    return " ".join(
        _flatten(
            product.get(
                field
            )
        )
        for field
        in (
            "title",
            "features",
            "details",
            "description",
            "store",
        )
    )


def raw_category_parts(
    product: dict,
) -> list[str]:

    raw = product.get(
        "categories",
        [],
    )

    if isinstance(
        raw,
        list,
    ):
        values = raw

    elif raw:
        values = [
            raw
        ]

    else:
        values = []

    return [
        _display_text(
            value
        )
        for value
        in values
        if _display_text(
            value
        )
    ]


def specific_category_parts(
    product: dict,
) -> list[str]:
    """
    Remove generic Amazon ancestors.

    Crucially, if a product only has:

        Clothing, Shoes & Jewelry
        Women

    this function returns [].

    We do NOT fall back to those generic values.
    """

    return [
        part
        for part
        in raw_category_parts(
            product
        )
        if (
            _normalized_category(
                part
            )
            not in
            GENERIC_CATEGORY_COMPONENTS
        )
    ]


def category_key_and_label(
    product: dict,
) -> tuple[
    tuple[str, ...],
    str,
]:

    parts = specific_category_parts(
        product
    )

    if not parts:
        return (
            (),
            "",
        )

    selected = parts[
        -2:
    ]

    key = tuple(
        _normalized_category(
            item
        )
        for item
        in selected
    )

    label = " ".join(
        selected
    )

    return (
        key,
        label,
    )


def _category_pattern_matches(
    parts: Iterable[str],
    pattern: str,
) -> bool:

    return any(
        re.search(
            pattern,
            _normalized_category(
                part
            ),
            flags=re.IGNORECASE,
        )
        is not None
        for part
        in parts
    )


def category_allowed(
    spec: ConceptSpec,
    category_parts: Iterable[str],
) -> bool:
    """
    Validate the semantic domain of a concept.

    Matching happens component-by-component,
    never against the concatenated root path.

    Therefore:

        "Clothing, Shoes & Jewelry"

    can never satisfy a footwear rule merely
    because the word "Shoes" occurs in the
    Amazon root category.
    """

    parts = list(
        category_parts
    )

    if not parts:
        return False

    allowed = any(
        _category_pattern_matches(
            parts,
            pattern,
        )
        for pattern
        in spec.allowed_category_patterns
    )

    if not allowed:
        return False

    excluded = any(
        _category_pattern_matches(
            parts,
            pattern,
        )
        for pattern
        in spec.excluded_category_patterns
    )

    return not excluded


def matches_concept(
    spec: ConceptSpec,
    product: dict,
) -> bool:

    text = product_matching_text(
        product
    )

    return any(
        re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )
        is not None
        for pattern
        in spec.positive_patterns
    )


def query_has_leakage(
    spec: ConceptSpec,
    query: str,
) -> bool:

    return any(
        re.search(
            pattern,
            query,
            flags=re.IGNORECASE,
        )
        is not None
        for pattern
        in spec.leakage_patterns
    )


def _stable_key_id(
    key: tuple[str, ...],
) -> str:

    raw = (
        " > ".join(
            key
        )
        .encode(
            "utf-8"
        )
    )

    return (
        hashlib.sha1(
            raw
        )
        .hexdigest()[
            :10
        ]
    )


# ==================================================
# CASE GENERATION
# ==================================================


def build_concept_cases(
    products: Iterable[dict],
    specs: tuple[
        ConceptSpec,
        ...
    ] = CONCEPTS,
    min_positives: int = 2,
    min_category_size: int = 10,
    min_negatives: int = 5,
) -> list[dict]:

    members: dict[
        tuple[str, ...],
        set[str],
    ] = defaultdict(
        set
    )

    labels: dict[
        tuple[str, ...],
        str,
    ] = {}

    category_domains: dict[
        tuple[str, ...],
        set[str],
    ] = defaultdict(
        set
    )

    positives: dict[
        tuple[
            str,
            tuple[str, ...],
        ],
        set[str],
    ] = defaultdict(
        set
    )

    for product in products:

        asin = _display_text(
            product.get(
                "parent_asin"
            )
        )

        parts = specific_category_parts(
            product
        )

        (
            key,
            label,
        ) = category_key_and_label(
            product
        )

        if (
            not asin
            or
            not parts
            or
            not key
            or
            not label
        ):
            continue

        members[
            key
        ].add(
            asin
        )

        labels[
            key
        ] = label

        category_domains[
            key
        ].update(
            parts
        )

        for spec in specs:

            if not category_allowed(
                spec,
                parts,
            ):
                continue

            if not matches_concept(
                spec,
                product,
            ):
                continue

            positives[
                (
                    spec.name,
                    key,
                )
            ].add(
                asin
            )

    cases: list[
        dict
    ] = []

    for spec in specs:

        for key in sorted(
            members
        ):

            category_parts = sorted(
                category_domains[
                    key
                ]
            )

            # Recheck the entire grouped category
            # before using it as a benchmark case.
            if not category_allowed(
                spec,
                category_parts,
            ):
                continue

            relevant = positives.get(
                (
                    spec.name,
                    key,
                ),
                set(),
            )

            category_size = len(
                members[
                    key
                ]
            )

            relevant_count = len(
                relevant
            )

            negative_count = (
                category_size
                - relevant_count
            )

            if (
                relevant_count
                < min_positives
            ):
                continue

            if (
                category_size
                < min_category_size
            ):
                continue

            if (
                negative_count
                < min_negatives
            ):
                continue

            category_label = (
                labels[
                    key
                ]
            )

            for (
                paraphrase_index,
                paraphrase,
            ) in enumerate(
                spec.paraphrases,
                start=1,
            ):

                query = (
                    f"{category_label}; "
                    f"{paraphrase}"
                )

                if query_has_leakage(
                    spec,
                    query,
                ):
                    continue

                cases.append(
                    {
                        "case_id": (
                            f"concept_"
                            f"{spec.name}_"
                            f"{_stable_key_id(key)}_"
                            f"p{paraphrase_index}"
                        ),

                        "concept":
                            spec.name,

                        "category":
                            category_label,

                        "category_key":
                            list(
                                key
                            ),

                        "category_domain_parts":
                            category_parts,

                        "paraphrase_index":
                            paraphrase_index,

                        "paraphrase":
                            paraphrase,

                        "query":
                            query,

                        "relevant_asins":
                            sorted(
                                relevant
                            ),

                        "relevant_count":
                            relevant_count,

                        "category_size":
                            category_size,

                        "positive_rate":
                            round(
                                relevant_count
                                / category_size,
                                6,
                            ),
                    }
                )

    return cases


def select_cases(
    cases: list[dict],
    max_cases: int,
    seed: int,
) -> list[dict]:

    buckets: dict[
        str,
        list[dict],
    ] = defaultdict(
        list
    )

    for case in cases:

        buckets[
            case[
                "concept"
            ]
        ].append(
            case
        )

    rng = random.Random(
        seed
    )

    for bucket in buckets.values():

        bucket.sort(
            key=lambda item: (
                tuple(
                    item[
                        "category_key"
                    ]
                ),
                item[
                    "paraphrase_index"
                ],
            )
        )

        rng.shuffle(
            bucket
        )

    concepts = sorted(
        buckets
    )

    positions = {
        concept: 0
        for concept
        in concepts
    }

    selected: list[
        dict
    ] = []

    while (
        len(
            selected
        )
        < max_cases
    ):

        added = False

        for concept in concepts:

            if (
                len(
                    selected
                )
                >= max_cases
            ):
                break

            position = positions[
                concept
            ]

            bucket = buckets[
                concept
            ]

            if (
                position
                >= len(
                    bucket
                )
            ):
                continue

            selected.append(
                bucket[
                    position
                ]
            )

            positions[
                concept
            ] += 1

            added = True

        if not added:
            break

    return selected


def load_catalog(
    path: Path,
) -> list[dict]:

    with path.open(
        encoding="utf-8"
    ) as handle:

        return [
            json.loads(
                line
            )
            for line
            in handle
            if line.strip()
        ]


# ==================================================
# RETRIEVAL METRICS
# ==================================================


def _route_case_metrics(
    ranked_asins: list[str],
    relevant_asins: set[str],
    cutoffs: list[int],
) -> dict:

    relevant_ranks = [
        index
        for index, asin
        in enumerate(
            ranked_asins,
            start=1,
        )
        if asin
        in relevant_asins
    ]

    result: dict[
        str,
        object,
    ] = {
        "first_relevant_rank":
            (
                relevant_ranks[
                    0
                ]
                if relevant_ranks
                else None
            ),

        "relevant_ranks":
            relevant_ranks,
    }

    for cutoff in cutoffs:

        hits = len(
            set(
                ranked_asins[
                    :cutoff
                ]
            )
            &
            relevant_asins
        )

        result[
            f"hits_at_{cutoff}"
        ] = hits

        result[
            f"hit_at_{cutoff}"
        ] = (
            hits > 0
        )

        result[
            f"precision_at_{cutoff}"
        ] = round(
            hits
            / cutoff,
            6,
        )

        result[
            f"recall_at_{cutoff}"
        ] = round(
            hits
            / len(
                relevant_asins
            ),
            6,
        )

    return result


def evaluate_case(
    agent: Agent,
    case: dict,
    depth: int,
) -> dict:

    cutoffs = sorted(
        {
            cutoff
            for cutoff
            in (
                10,
                50,
                depth,
            )
            if cutoff
            <= depth
        }
    )

    state = SessionState(
        user_profile={}
    )

    state.category_text = (
        case[
            "category"
        ]
    )

    state.evidence = [
        Evidence(
            turn=2,
            text=(
                case[
                    "paraphrase"
                ]
            ),
        )
    ]

    lexical_candidates = (
        retrieve_candidates(
            connection=(
                agent.connection
            ),

            state=state,

            rating_numbers=(
                agent._rating_numbers
            ),

            asin_to_rowid=(
                agent._asin_to_rowid
            ),

            semantic=None,

            exploration=False,
        )
    )

    lexical_asins = [
        item[
            "parent_asin"
        ]
        for item
        in lexical_candidates[
            :depth
        ]
    ]

    semantic_asins = [
        asin
        for asin, _
        in agent.semantic.search(
            state.active_text(),
            top_n=depth,
        )
    ]

    relevant = set(
        case[
            "relevant_asins"
        ]
    )

    lexical = _route_case_metrics(
        lexical_asins,
        relevant,
        cutoffs,
    )

    semantic = _route_case_metrics(
        semantic_asins,
        relevant,
        cutoffs,
    )

    hybrid: dict[
        str,
        object,
    ] = {}

    for cutoff in cutoffs:

        union = (
            set(
                lexical_asins[
                    :cutoff
                ]
            )
            |
            set(
                semantic_asins[
                    :cutoff
                ]
            )
        )

        hits = len(
            union
            &
            relevant
        )

        hybrid[
            f"candidate_count_at_{cutoff}"
        ] = len(
            union
        )

        hybrid[
            f"hits_at_{cutoff}"
        ] = hits

        hybrid[
            f"hit_at_{cutoff}"
        ] = (
            hits > 0
        )

        hybrid[
            f"candidate_recall_at_{cutoff}"
        ] = round(
            hits
            / len(
                relevant
            ),
            6,
        )

    return {
        "case_id":
            case[
                "case_id"
            ],

        "concept":
            case[
                "concept"
            ],

        "category":
            case[
                "category"
            ],

        "category_domain_parts":
            case[
                "category_domain_parts"
            ],

        "paraphrase_index":
            case[
                "paraphrase_index"
            ],

        "paraphrase":
            case[
                "paraphrase"
            ],

        "query":
            case[
                "query"
            ],

        "relevant_count":
            case[
                "relevant_count"
            ],

        "positive_rate":
            case[
                "positive_rate"
            ],

        "relevant_asin_sample":
            case[
                "relevant_asins"
            ][
                :10
            ],

        "category_size":
            case[
                "category_size"
            ],

        "lexical":
            lexical,

        "semantic":
            semantic,

        "hybrid":
            hybrid,
    }


def _summarize_route(
    results: list[dict],
    route: str,
    cutoffs: list[int],
    depth: int,
) -> dict:

    if not results:
        return {
            "sample_count": 0
        }

    summary: dict[
        str,
        object,
    ] = {
        "sample_count":
            len(
                results
            )
    }

    for cutoff in cutoffs:

        summary[
            f"hit_rate_at_{cutoff}"
        ] = round(
            sum(
                result[
                    route
                ][
                    f"hit_at_{cutoff}"
                ]
                for result
                in results
            )
            / len(
                results
            ),
            6,
        )

        summary[
            f"macro_precision_at_{cutoff}"
        ] = round(
            sum(
                result[
                    route
                ][
                    f"precision_at_{cutoff}"
                ]
                for result
                in results
            )
            / len(
                results
            ),
            6,
        )

        summary[
            f"macro_recall_at_{cutoff}"
        ] = round(
            sum(
                result[
                    route
                ][
                    f"recall_at_{cutoff}"
                ]
                for result
                in results
            )
            / len(
                results
            ),
            6,
        )

    summary[
        f"mrr_at_{depth}"
    ] = round(
        sum(
            (
                1.0
                /
                result[
                    route
                ][
                    "first_relevant_rank"
                ]
            )
            if (
                result[
                    route
                ][
                    "first_relevant_rank"
                ]
                is not None
            )
            else 0.0
            for result
            in results
        )
        / len(
            results
        ),
        6,
    )

    return summary


def _summarize_hybrid(
    results: list[dict],
    cutoffs: list[int],
) -> dict:

    if not results:
        return {
            "sample_count": 0
        }

    summary: dict[
        str,
        object,
    ] = {
        "sample_count":
            len(
                results
            )
    }

    for cutoff in cutoffs:

        summary[
            f"hit_rate_at_{cutoff}"
        ] = round(
            sum(
                result[
                    "hybrid"
                ][
                    f"hit_at_{cutoff}"
                ]
                for result
                in results
            )
            / len(
                results
            ),
            6,
        )

        summary[
            f"macro_candidate_recall_at_{cutoff}"
        ] = round(
            sum(
                result[
                    "hybrid"
                ][
                    f"candidate_recall_at_{cutoff}"
                ]
                for result
                in results
            )
            / len(
                results
            ),
            6,
        )

        summary[
            f"mean_candidate_count_at_{cutoff}"
        ] = round(
            sum(
                result[
                    "hybrid"
                ][
                    f"candidate_count_at_{cutoff}"
                ]
                for result
                in results
            )
            / len(
                results
            ),
            3,
        )

    return summary


def summarize_results(
    results: list[dict],
    depth: int,
) -> dict:

    cutoffs = sorted(
        {
            cutoff
            for cutoff
            in (
                10,
                50,
                depth,
            )
            if cutoff
            <= depth
        }
    )

    lexical = _summarize_route(
        results,
        "lexical",
        cutoffs,
        depth,
    )

    semantic = _summarize_route(
        results,
        "semantic",
        cutoffs,
        depth,
    )

    hybrid = _summarize_hybrid(
        results,
        cutoffs,
    )

    semantic_rescues = 0
    lexical_only = 0
    both = 0
    neither = 0

    for result in results:

        lexical_hit = result[
            "lexical"
        ][
            f"hit_at_{depth}"
        ]

        semantic_hit = result[
            "semantic"
        ][
            f"hit_at_{depth}"
        ]

        if (
            lexical_hit
            and semantic_hit
        ):
            both += 1

        elif lexical_hit:
            lexical_only += 1

        elif semantic_hit:
            semantic_rescues += 1

        else:
            neither += 1

    grouped: dict[
        str,
        list[dict],
    ] = defaultdict(
        list
    )

    for result in results:

        grouped[
            result[
                "concept"
            ]
        ].append(
            result
        )

    by_concept = {}

    for (
        concept,
        group,
    ) in sorted(
        grouped.items()
    ):

        by_concept[
            concept
        ] = {
            "lexical":
                _summarize_route(
                    group,
                    "lexical",
                    cutoffs,
                    depth,
                ),

            "semantic":
                _summarize_route(
                    group,
                    "semantic",
                    cutoffs,
                    depth,
                ),

            "hybrid":
                _summarize_hybrid(
                    group,
                    cutoffs,
                ),
        }

    return {
        "sample_count":
            len(
                results
            ),

        "depth":
            depth,

        "lexical":
            lexical,

        "semantic":
            semantic,

        "hybrid":
            hybrid,

        "complementarity": {
            "semantic_rescue_cases":
                semantic_rescues,

            "lexical_only_cases":
                lexical_only,

            "both_routes_hit":
                both,

            "neither_route_hit":
                neither,
        },

        "by_concept":
            by_concept,
    }


# ==================================================
# CLI
# ==================================================


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Catalogue-derived, domain-gated "
            "concept robustness benchmark."
        )
    )

    parser.add_argument(
        "--catalog",
        default="data/catalog.jsonl",
    )

    parser.add_argument(
        "--cases",
        type=int,
        default=120,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
    )

    parser.add_argument(
        "--depth",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--min-positives",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--min-category-size",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--min-negatives",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--output",
        default=(
            "experiments/"
            "v10_2_concept_robustness.json"
        ),
    )

    args = parser.parse_args()

    if args.cases <= 0:
        raise ValueError(
            "--cases must be greater than 0"
        )

    if not (
        10
        <= args.depth
        <= 100
    ):
        raise ValueError(
            "--depth must be between 10 and 100"
        )

    catalog_path = Path(
        args.catalog
    )

    products = load_catalog(
        catalog_path
    )

    available = build_concept_cases(
        products,

        min_positives=(
            args.min_positives
        ),

        min_category_size=(
            args.min_category_size
        ),

        min_negatives=(
            args.min_negatives
        ),
    )

    selected = select_cases(
        available,
        args.cases,
        args.seed,
    )

    if not selected:
        raise RuntimeError(
            (
                "No valid concept robustness "
                "cases could be generated."
            )
        )

    available_by_concept: dict[
        str,
        int,
    ] = defaultdict(
        int
    )

    selected_by_concept: dict[
        str,
        int,
    ] = defaultdict(
        int
    )

    for case in available:

        available_by_concept[
            case[
                "concept"
            ]
        ] += 1

    for case in selected:

        selected_by_concept[
            case[
                "concept"
            ]
        ] += 1

    print(
        (
            f"Built {len(available)} valid "
            f"domain-gated cases; "
            f"selected {len(selected)}."
        )
    )

    print(
        "Building agent indexes..."
    )

    agent = Agent(
        catalog_path=(
            catalog_path
        )
    )

    results: list[
        dict
    ] = []

    for (
        index,
        case,
    ) in enumerate(
        selected,
        start=1,
    ):

        results.append(
            evaluate_case(
                agent,
                case,
                args.depth,
            )
        )

        if (
            index % 20 == 0
            or
            index
            == len(
                selected
            )
        ):

            print(
                (
                    f"Evaluated "
                    f"{index}/"
                    f"{len(selected)}"
                )
            )

    summary = summarize_results(
        results,
        args.depth,
    )

    report = {
        "benchmark":
            (
                "catalogue-derived "
                "domain-gated concept "
                "robustness evaluation"
            ),

        "uses_public_labels":
            False,

        "methodology": {
            "exact_source_asin_is_only_positive":
                False,

            "same_category_positive_set":
                True,

            "generic_category_ancestors_removed":
                True,

            "category_matching_is_component_scoped":
                True,

            "concept_domain_gating":
                True,

            "held_out_keyword_leakage_rejected":
                True,

            "multiple_paraphrases_per_concept":
                True,
        },

        "config": {
            "catalog":
                str(
                    catalog_path
                ),

            "requested_cases":
                args.cases,

            "selected_cases":
                len(
                    selected
                ),

            "available_cases":
                len(
                    available
                ),

            "seed":
                args.seed,

            "depth":
                args.depth,

            "min_positives":
                args.min_positives,

            "min_category_size":
                args.min_category_size,

            "min_negatives":
                args.min_negatives,

            "available_cases_by_concept":
                dict(
                    sorted(
                        available_by_concept.items()
                    )
                ),

            "selected_cases_by_concept":
                dict(
                    sorted(
                        selected_by_concept.items()
                    )
                ),
        },

        "summary":
            summary,

        "cases":
            results,
    }

    output = Path(
        args.output
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "Concept robustness summary"
    )
    print(
        "=========================="
    )

    print(
        json.dumps(
            {
                "sample_count":
                    summary[
                        "sample_count"
                    ],

                "lexical":
                    summary[
                        "lexical"
                    ],

                "semantic":
                    summary[
                        "semantic"
                    ],

                "hybrid":
                    summary[
                        "hybrid"
                    ],

                "complementarity":
                    summary[
                        "complementarity"
                    ],
            },
            indent=2,
        )
    )

    print()
    print(
        (
            f"Saved report to: "
            f"{output}"
        )
    )


if __name__ == "__main__":
    main()