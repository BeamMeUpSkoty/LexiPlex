from __future__ import annotations

import re

import pandas as pd

from ..core.base import BaseAnalyzer, AnalysisResult, EmbeddingCache

HEDGING: frozenset[str] = frozenset({
    "maybe", "perhaps", "i think", "sort of", "kind of",
    "probably", "possibly", "i suppose", "i guess",
})
CERTAINTY: frozenset[str] = frozenset({
    "definitely", "absolutely", "certainly", "always", "never",
    "without doubt", "for sure",
})
NEGATION_TOKENS: frozenset[str] = frozenset({
    "not", "no", "never", "don't", "doesn't", "didn't",
    "can't", "cannot", "won't", "isn't", "aren't", "wasn't",
})
_SELF_REF = re.compile(r"\b(i|me|my|myself|i'm|i've|i'll|i'd)\b", re.IGNORECASE)


def _phrase_rate(text: str, phrases: frozenset[str]) -> float:
    lower = text.lower()
    words = lower.split()
    if not words:
        return 0.0
    return sum(1 for p in phrases if p in lower) / len(words)


def _negation_density(text: str) -> float:
    words = text.lower().split()
    if not words:
        return 0.0
    return sum(1 for w in words if w in NEGATION_TOKENS) / len(words)


def _self_ref_rate(text: str) -> float:
    words = text.split()
    if not words:
        return 0.0
    return len(_SELF_REF.findall(text)) / len(words)


class ClinicalMarkerAnalyzer(BaseAnalyzer):
    """
    Rule-based clinical language markers. No model required.
    Detects hedging, certainty, self-reference, negation, and questions.
    """

    @property
    def name(self) -> str:
        return "clinical"

    def analyze(self, df: pd.DataFrame, cache: EmbeddingCache) -> AnalysisResult:
        out = df.copy()
        sentences = out["sentence"].tolist()
        out["hedging_rate"] = [_phrase_rate(s, HEDGING) for s in sentences]
        out["certainty_rate"] = [_phrase_rate(s, CERTAINTY) for s in sentences]
        out["self_ref_rate"] = [_self_ref_rate(s) for s in sentences]
        out["negation_density"] = [_negation_density(s) for s in sentences]
        out["is_question"] = [int(s.strip().endswith("?")) for s in sentences]
        return AnalysisResult(
            name=self.name,
            per_sentence=out,
            global_metrics=self._compute_global(out),
        )

    def _compute_global(self, df: pd.DataFrame) -> dict[str, float]:
        cols = ("hedging_rate", "certainty_rate", "self_ref_rate", "negation_density", "is_question")
        metrics: dict[str, float] = {}
        if "speaker" in df.columns:
            for speaker, group in df.groupby("speaker"):
                for col in cols:
                    metrics[f"{speaker}_{col}"] = float(group[col].mean())
        else:
            for col in cols:
                metrics[col] = float(df[col].mean())
        return metrics
