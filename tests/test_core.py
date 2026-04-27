import pytest
import numpy as np
import pandas as pd
from affect_analyzer.core.base import BaseAnalyzer, AnalysisResult, EmbeddingCache
from affect_analyzer.core.registry import AnalyzerRegistry


def test_embedding_cache_defaults_to_none():
    cache = EmbeddingCache()
    assert cache.embeddings is None
    assert cache.sentences is None


def test_analysis_result_stores_fields():
    df = pd.DataFrame({"sentence": ["hello world"]})
    result = AnalysisResult(name="test", per_sentence=df, global_metrics={"score": 0.5})
    assert result.name == "test"
    assert result.global_metrics["score"] == 0.5
    assert result.metadata == {}


def test_base_analyzer_is_abstract():
    with pytest.raises(TypeError):
        BaseAnalyzer()


def test_partial_analyzer_missing_name_is_still_abstract():
    class NoName(BaseAnalyzer):
        def analyze(self, df, cache):
            return AnalysisResult(name="x", per_sentence=df.copy(), global_metrics={})

    with pytest.raises(TypeError):
        NoName()


def test_concrete_analyzer_works_when_complete():
    class GoodAnalyzer(BaseAnalyzer):
        @property
        def name(self) -> str:
            return "good"

        def analyze(self, df, cache):
            return AnalysisResult(name="good", per_sentence=df.copy(), global_metrics={})

    a = GoodAnalyzer()
    assert a.name == "good"


class _DoubleAnalyzer(BaseAnalyzer):
    @property
    def name(self) -> str:
        return "double"

    def analyze(self, df, cache):
        return AnalysisResult(
            name="double",
            per_sentence=df.copy(),
            global_metrics={"count": float(len(df))},
        )


def test_registry_run_all_returns_result_per_analyzer():
    registry = AnalyzerRegistry()
    registry.register(_DoubleAnalyzer())
    df = pd.DataFrame({"sentence": ["hello", "world"]})
    cache = EmbeddingCache()
    results = registry.run_all(df, cache)
    assert "double" in results
    assert results["double"].global_metrics["count"] == 2.0


def test_registry_duplicate_name_raises():
    registry = AnalyzerRegistry()
    registry.register(_DoubleAnalyzer())
    with pytest.raises(ValueError, match="already registered"):
        registry.register(_DoubleAnalyzer())
