"""Semantic similarity backend.

Uses sentence-transformers when installed (true semantic embeddings) and falls
back to a hybrid word + character n-gram TF-IDF model otherwise, so the tool
always works offline. The active backend is recorded in the report.
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from ..logging_setup import get_logger
from ..textutil import SKLEARN_STOPWORDS, canonical_text

log = get_logger("embeddings")


class SemanticModel:
    def __init__(self, use_transformers: bool = True, model_name: str = "all-MiniLM-L6-v2"):
        self.backend = "tfidf"
        self._st_model = None
        self._word: TfidfVectorizer | None = None
        self._bow: TfidfVectorizer | None = None
        self._char: TfidfVectorizer | None = None
        if use_transformers:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore

                self._st_model = SentenceTransformer(model_name)
                self.backend = f"sentence-transformers:{model_name}"
                log.info("Semantic backend: %s", self.backend)
            except Exception as exc:
                log.info("sentence-transformers unavailable (%s) – using TF-IDF similarity", type(exc).__name__)

    # ------------------------------------------------------------------
    def fit(self, corpus: list[str]) -> SemanticModel:
        if self._st_model is not None:
            return self
        corpus = [canonical_text(c) for c in corpus if c and c.strip()] or ["empty"]
        token_pattern = r"(?u)\b[a-zA-Z0-9][a-zA-Z0-9]+\b"
        self._word = TfidfVectorizer(ngram_range=(1, 2), stop_words=SKLEARN_STOPWORDS, sublinear_tf=True,
                                     token_pattern=token_pattern)
        # Binary bag-of-words without IDF: on a topical site the topic words are frequent and
        # IDF would drown them out – this block keeps them visible.
        self._bow = TfidfVectorizer(ngram_range=(1, 1), stop_words=SKLEARN_STOPWORDS, use_idf=False, binary=True,
                                    token_pattern=token_pattern)
        self._char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
        try:
            self._word.fit(corpus)
            self._bow.fit(corpus)
        except ValueError:  # corpus contains only stopwords
            self._word = self._bow = None
        self._char.fit(corpus)
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        texts = [t or "" for t in texts]
        if not texts:
            return np.zeros((0, 1))
        if self._st_model is not None:
            vectors = self._st_model.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
            return np.asarray(vectors, dtype=np.float32)
        if self._char is None:
            self.fit(texts)
        texts = [canonical_text(t) for t in texts]
        # Concatenating sqrt-weighted, L2-normalised blocks makes the cosine a weighted
        # average of the per-block cosines.
        blocks = []
        if self._word is not None and self._bow is not None:
            blocks.append((self._word, 0.4))
            blocks.append((self._bow, 0.35))
            blocks.append((self._char, 0.25))
        else:
            blocks.append((self._char, 1.0))
        parts = [normalize(vec.transform(texts)).toarray() * np.sqrt(w) for vec, w in blocks]
        return np.hstack(parts).astype(np.float32)

    @staticmethod
    def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if a.size == 0 or b.size == 0:
            return np.zeros((a.shape[0], b.shape[0]))
        return np.clip(a @ b.T, 0.0, 1.0)

    def similarity(self, a: list[str], b: list[str]) -> np.ndarray:
        return self.cosine_matrix(self.encode(a), self.encode(b))
