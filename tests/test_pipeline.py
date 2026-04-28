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


def test_pipeline_cache_has_embeddings():
    registry = AnalyzerRegistry()
    captured_cache = {}

    class _CacheCapturingAnalyzer(BaseAnalyzer):
        @property
        def name(self):
            return "cache_capture"

        def analyze(self, df, cache):
            captured_cache["cache"] = cache
            return AnalysisResult(name="cache_capture", per_sentence=df.copy(), global_metrics={})

    registry.register(_CacheCapturingAnalyzer())
    pipeline = LexiPlexPipeline(registry=registry, model=_fake_model(), language="en")
    pipeline.run(DATA_CSV)

    assert captured_cache["cache"].embeddings is not None
    assert captured_cache["cache"].embeddings.shape[1] == 768
