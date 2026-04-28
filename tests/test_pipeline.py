# tests/test_pipeline.py
import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock

from affect_analyzer.core.base import BaseAnalyzer, AnalysisResult, EmbeddingCache
from affect_analyzer.core.registry import AnalyzerRegistry
from affect_analyzer.pipeline import LexiPlexPipeline

DATA_CSV = "data/mock_therapy_session.csv"


class _PassthroughAnalyzer(BaseAnalyzer):
    @property
    def name(self) -> str:
        return "passthrough"

    def analyze(self, df, cache):
        return AnalysisResult(
            name="passthrough",
            per_sentence=df.copy(),
            global_metrics={"rows": float(len(df))},
        )


def _fake_model():
    model = MagicMock()
    model.embed_sentences.side_effect = lambda sentences, **kw: np.zeros((len(sentences), 768))
    return model


def test_pipeline_run_returns_result_dict():
    registry = AnalyzerRegistry()
    registry.register(_PassthroughAnalyzer())
    pipeline = LexiPlexPipeline(registry=registry, model=_fake_model(), language="en")
    results = pipeline.run(DATA_CSV)
    assert "passthrough" in results
    assert results["passthrough"].global_metrics["rows"] > 0


def test_pipeline_adds_duration_and_turn_columns():
    registry = AnalyzerRegistry()
    captured = {}

    class _CapturingAnalyzer(BaseAnalyzer):
        @property
        def name(self):
            return "capture"

        def analyze(self, df, cache):
            captured["df"] = df
            return AnalysisResult(name="capture", per_sentence=df.copy(), global_metrics={})

    registry.register(_CapturingAnalyzer())
    pipeline = LexiPlexPipeline(registry=registry, model=_fake_model(), language="en")
    pipeline.run(DATA_CSV)

    df = captured["df"]
    assert "duration" in df.columns
    assert "turn_id" in df.columns
    assert "is_turn_start" in df.columns
    assert df["turn_id"].min() >= 0, "turn_id must be 0-based (no negative values)"
    assert df["is_turn_start"].iloc[0] == False, "first sentence is never a turn start"
    assert df["duration"].min() >= 0, "duration must be non-negative"


def test_pipeline_cache_is_shared_and_embed_called_once():
    """Embeddings must be computed once and the same cache object passed to all analyzers."""
    registry = AnalyzerRegistry()
    caches_received: list = []

    class _CacheCapturingA(BaseAnalyzer):
        @property
        def name(self): return "cache_a"
        def analyze(self, df, cache):
            caches_received.append(cache)
            return AnalysisResult(name="cache_a", per_sentence=df.copy(), global_metrics={})

    class _CacheCapturingB(BaseAnalyzer):
        @property
        def name(self): return "cache_b"
        def analyze(self, df, cache):
            caches_received.append(cache)
            return AnalysisResult(name="cache_b", per_sentence=df.copy(), global_metrics={})

    registry.register(_CacheCapturingA())
    registry.register(_CacheCapturingB())
    model = _fake_model()
    pipeline = LexiPlexPipeline(registry=registry, model=model, language="en")
    pipeline.run(DATA_CSV)

    assert len(caches_received) == 2
    assert caches_received[0].embeddings is not None
    assert caches_received[0].embeddings.shape[1] == 768
    # Same cache object passed to both analyzers
    assert caches_received[0] is caches_received[1]
    # embed_sentences called exactly once regardless of analyzer count
    assert model.embed_sentences.call_count == 1
