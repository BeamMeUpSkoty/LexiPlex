from __future__ import annotations

import pandas as pd

from .base import BaseAnalyzer, AnalysisResult, EmbeddingCache


class AnalyzerRegistry:
    """Collects BaseAnalyzer instances and runs them all against a DataFrame."""

    def __init__(self) -> None:
        self._analyzers: dict[str, BaseAnalyzer] = {}

    def register(self, analyzer: BaseAnalyzer) -> None:
        if analyzer.name in self._analyzers:
            raise ValueError(f"Analyzer '{analyzer.name}' already registered.")
        self._analyzers[analyzer.name] = analyzer

    def run_all(self, df: pd.DataFrame, cache: EmbeddingCache) -> dict[str, AnalysisResult]:
        return {name: a.analyze(df, cache) for name, a in self._analyzers.items()}
