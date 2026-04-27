import pytest
import numpy as np
import pandas as pd
from affect_analyzer.core.base import BaseAnalyzer, AnalysisResult, EmbeddingCache


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


def test_concrete_analyzer_works_when_complete():
    class GoodAnalyzer(BaseAnalyzer):
        @property
        def name(self) -> str:
            return "good"

        def analyze(self, df, cache):
            return AnalysisResult(name="good", per_sentence=df.copy(), global_metrics={})

    a = GoodAnalyzer()
    assert a.name == "good"
