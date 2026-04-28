from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from ..core.base import BaseAnalyzer, AnalysisResult, EmbeddingCache

CONTENT_POS = {"NOUN", "VERB", "ADJ", "ADV"}


class ComplexityAnalyzer(BaseAnalyzer):
    """
    Vocabulary richness (TTR), lexical density, and discourse coherence
    (cosine similarity between adjacent sentence embeddings from the shared cache).
    """

    def __init__(self, nlp) -> None:
        self._nlp = nlp

    @property
    def name(self) -> str:
        return "complexity"

    def analyze(self, df: pd.DataFrame, cache: EmbeddingCache) -> AnalysisResult:
        out = df.copy()

        word_counts, ttrs, densities = [], [], []
        for sent in out["sentence"].tolist():
            doc = self._nlp(sent)
            tokens = [t for t in doc if not t.is_space and not t.is_punct]
            words = [t.text.lower() for t in tokens]
            content = [t for t in tokens if t.pos_ in CONTENT_POS]
            n = len(words)
            word_counts.append(n)
            ttrs.append(len(set(words)) / n if n > 0 else 0.0)
            densities.append(len(content) / n if n > 0 else 0.0)

        out["word_count"] = word_counts
        out["type_token_ratio"] = ttrs
        out["lexical_density"] = densities

        if cache.embeddings is not None and len(cache.embeddings) > 1:
            embs = cache.embeddings
            sims = cosine_similarity(embs[:-1], embs[1:]).diagonal()
            out["coherence_to_prev"] = np.concatenate([[np.nan], sims])
        else:
            out["coherence_to_prev"] = np.nan

        return AnalysisResult(
            name=self.name,
            per_sentence=out,
            global_metrics=self._compute_global(out),
        )

    def _compute_global(self, df: pd.DataFrame) -> dict[str, float]:
        metrics: dict[str, float] = {
            "coherence_mean": float(df["coherence_to_prev"].dropna().mean()),
        }
        if "speaker" in df.columns:
            for speaker, group in df.groupby("speaker"):
                metrics[f"{speaker}_ttr_mean"] = float(group["type_token_ratio"].mean())
                metrics[f"{speaker}_wc_mean"] = float(group["word_count"].mean())
                metrics[f"{speaker}_density_mean"] = float(group["lexical_density"].mean())
        return metrics
