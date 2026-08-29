from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import (
    normalize,
)


class SemanticRetriever:
    """
    Lightweight dense retrieval over a
    catalogue-trained latent semantic space.

    The expensive representation learning is
    performed offline by:

        scripts/build_semantic_index.py

    Runtime search only performs:

        query
          ↓
        TF-IDF transform
          ↓
        SVD transform
          ↓
        cosine similarity against 50k vectors
    """

    def __init__(
        self,
        pipeline_path: str | Path = (
            "assets/"
            "semantic_pipeline.joblib"
        ),
        embeddings_path: str | Path = (
            "assets/"
            "catalog_lsa.npy"
        ),
        asins_path: str | Path = (
            "assets/"
            "semantic_asins.json"
        ),
    ) -> None:

        self.vectorizer, self.svd = (
            joblib.load(
                pipeline_path
            )
        )

        # Stored on disk as float16 to save
        # space. Convert to float32 once at
        # startup for faster similarity math.
        self.embeddings = (
            np.load(
                embeddings_path
            )
            .astype(
                np.float32,
                copy=False,
            )
        )

        with Path(
            asins_path
        ).open(
            encoding="utf-8"
        ) as handle:

            self.asins: list[str] = (
                json.load(
                    handle
                )
            )

        if (
            self.embeddings.shape[0]
            != len(
                self.asins
            )
        ):
            raise ValueError(
                "semantic embeddings and "
                "ASIN mapping are misaligned"
            )

    def search(
        self,
        query: str,
        top_n: int = 100,
    ) -> list[
        tuple[
            str,
            float,
        ]
    ]:
        """
        Return:

            [
                (parent_asin, similarity),
                ...
            ]

        ordered from most semantically similar
        to least similar.
        """

        if not query.strip():
            return []

        sparse_query = (
            self.vectorizer
            .transform(
                [
                    query
                ]
            )
        )

        dense_query = (
            self.svd
            .transform(
                sparse_query
            )
            .astype(
                np.float32
            )
        )

        dense_query = (
            normalize(
                dense_query
            )
            .astype(
                np.float32,
                copy=False,
            )[0]
        )

        # Because both catalogue vectors and
        # query vector are normalized:
        #
        # cosine_similarity =
        # embeddings @ query

        scores = (
            self.embeddings
            @ dense_query
        )

        n = min(
            top_n,
            scores.shape[0],
        )

        if n <= 0:
            return []

        # argpartition avoids sorting all
        # 50,000 products when only the
        # highest-scoring subset is needed.
        indexes = np.argpartition(
            scores,
            -n,
        )[-n:]

        indexes = indexes[
            np.argsort(
                scores[
                    indexes
                ]
            )[::-1]
        ]

        return [
            (
                self.asins[
                    int(index)
                ],
                float(
                    scores[
                        index
                    ]
                ),
            )
            for index
            in indexes
        ]