from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd

from ..core.base import BaseAnalyzer, AnalysisResult, EmbeddingCache
from ..topics.topic_modeler import TopicModeler

if TYPE_CHECKING:
    from ..modelling.valence_arousal import ValenceArousalModel


class AffectAnalyzer(BaseAnalyzer):
    """
    Scores each sentence on valence and arousal via XLM-RoBERTa batch_score().
    Optionally assigns LDA topic clusters.
    """

    def __init__(
        self,
        model: ValenceArousalModel,
        use_topics: bool = False,
        num_topics: int = 5,
        max_features: int = 1000,
        topic_keywords_top_n: int = 3,
    ) -> None:
        self._model = model
        self.use_topics = use_topics
        self.num_topics = num_topics
        self.max_features = max_features
        self.topic_keywords_top_n = topic_keywords_top_n
        self._topic_modeler: Optional[TopicModeler] = TopicModeler() if use_topics else None

    @property
    def name(self) -> str:
        return "affect"

    def analyze(self, df: pd.DataFrame, cache: EmbeddingCache) -> AnalysisResult:
        sentences = df["sentence"].tolist()

        scores = self._model.batch_score(sentences)
        if hasattr(scores, "numpy"):
            scores = scores.numpy()

        out = df.copy()
        out["valence"] = scores[:, 0]
        out["arousal"] = scores[:, 1]

        if self.use_topics and self._topic_modeler is not None:
            out["topic"] = self._topic_modeler.fit_global_lda(
                out["sentence"], self.num_topics, self.max_features
            )
            labels = self._topic_modeler.make_topic_labels(self.topic_keywords_top_n)
            out["topic_label"] = out["topic"].map(labels)

        return AnalysisResult(
            name=self.name,
            per_sentence=out,
            global_metrics=self._compute_global(out),
        )

    def _compute_global(self, df: pd.DataFrame) -> dict[str, float]:
        metrics: dict[str, float] = {}
        if "speaker" in df.columns and "duration" in df.columns:
            for speaker, group in df.groupby("speaker"):
                w = group["duration"]
                total = w.sum()
                if total > 0:
                    metrics[f"{speaker}_valence_mean"] = float((group["valence"] * w).sum() / total)
                    metrics[f"{speaker}_arousal_mean"] = float((group["arousal"] * w).sum() / total)
        metrics.setdefault("valence_mean", float(df["valence"].mean()))
        metrics.setdefault("arousal_mean", float(df["arousal"].mean()))
        return metrics
