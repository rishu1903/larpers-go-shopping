from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


CATALOG_PATH = Path("data/catalog.jsonl")

ASSET_DIR = Path("assets")

PIPELINE_PATH = (
    ASSET_DIR
    / "semantic_pipeline.joblib"
)

EMBEDDINGS_PATH = (
    ASSET_DIR
    / "catalog_lsa.npy"
)

ASINS_PATH = (
    ASSET_DIR
    / "semantic_asins.json"
)


def _text(
    value: object,
) -> str:
    """
    Flatten catalogue values into searchable text.
    """

    if value is None:
        return ""

    if isinstance(
        value,
        dict,
    ):
        return " ".join(
            f"{key} {item}"
            for key, item
            in value.items()
        )

    if isinstance(
        value,
        list,
    ):
        return " ".join(
            str(item)
            for item
            in value
        )

    return str(value)


def product_text(
    product: dict,
) -> str:
    """
    Build one semantic document per product.

    We deliberately exclude:
    - parent_asin
    - rating_number
    - average_rating
    - price

    because the semantic representation should
    describe what the product IS, not how popular
    it is or how much it costs.
    """

    fields = (
        product.get(
            "title"
        ),
        product.get(
            "categories"
        ),
        product.get(
            "features"
        ),
        product.get(
            "details"
        ),
        product.get(
            "store"
        ),
        product.get(
            "description"
        ),
    )

    return " ".join(
        _text(value)
        for value
        in fields
    )


def main() -> None:

    if not CATALOG_PATH.exists():
        raise FileNotFoundError(
            f"Expected catalogue at "
            f"{CATALOG_PATH}. "
            f"Run this script from "
            f"the repository root."
        )

    texts: list[str] = []
    asins: list[str] = []

    print(
        "Reading catalogue..."
    )

    with CATALOG_PATH.open(
        encoding="utf-8"
    ) as handle:

        for line in handle:

            product = json.loads(
                line
            )

            asins.append(
                str(
                    product[
                        "parent_asin"
                    ]
                )
            )

            texts.append(
                product_text(
                    product
                )
            )

    if len(texts) != 50_000:
        raise ValueError(
            "Expected 50,000 products "
            f"but found "
            f"{len(texts):,}."
        )

    # --------------------------------------------------
    # STAGE 1:
    # SPARSE TF-IDF PRODUCT REPRESENTATION
    # --------------------------------------------------

    print(
        "Building TF-IDF representation..."
    )

    vectorizer = (
        TfidfVectorizer(
            max_features=20_000,
            ngram_range=(
                1,
                2,
            ),
            min_df=2,
            max_df=0.98,
            sublinear_tf=True,
            dtype=np.float32,
        )
    )

    sparse_matrix = (
        vectorizer
        .fit_transform(
            texts
        )
    )

    # --------------------------------------------------
    # STAGE 2:
    # LATENT SEMANTIC ANALYSIS
    # --------------------------------------------------
    #
    # Truncated SVD learns latent relationships
    # between product words and phrases.
    #
    # The original TF-IDF vectors contain
    # 20,000 dimensions.
    #
    # We compress that into:
    #
    #     96 dense dimensions
    #
    # giving us a lightweight semantic space.

    print(
        "Learning 96-dimensional "
        "latent semantic space..."
    )

    svd = TruncatedSVD(
        n_components=96,
        n_iter=5,
        random_state=42,
    )

    embeddings = (
        svd
        .fit_transform(
            sparse_matrix
        )
        .astype(
            np.float32
        )
    )

    # Cosine similarity becomes a simple
    # dot product after normalization.
    embeddings = (
        normalize(
            embeddings
        )
        .astype(
            np.float16
        )
    )

    # --------------------------------------------------
    # WRITE ASSETS
    # --------------------------------------------------

    ASSET_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Writing semantic assets..."
    )

    joblib.dump(
        (
            vectorizer,
            svd,
        ),
        PIPELINE_PATH,
        compress=3,
    )

    np.save(
        EMBEDDINGS_PATH,
        embeddings,
    )

    # Keep a separate ASIN mapping rather than
    # assuming that future catalogue row order
    # will always be identical.
    with ASINS_PATH.open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            asins,
            handle,
            separators=(
                ",",
                ":",
            ),
        )

    print()
    print(
        "Semantic index ready:"
    )
    print(
        f"  Products:   "
        f"{len(asins):,}"
    )
    print(
        f"  Dimensions: "
        f"{embeddings.shape[1]}"
    )
    print(
        f"  Vocabulary: "
        f"{len(vectorizer.vocabulary_):,}"
    )
    print(
        f"  Pipeline:   "
        f"{PIPELINE_PATH}"
    )
    print(
        f"  Embeddings: "
        f"{EMBEDDINGS_PATH}"
    )
    print(
        f"  ASIN map:   "
        f"{ASINS_PATH}"
    )


if __name__ == "__main__":
    main()