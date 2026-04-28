import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock

from affect_analyzer.core.base import EmbeddingCache


def _make_df():
    return pd.DataFrame({
        "sentence": ["I feel anxious.", "Things are okay.", "I'm worried.", "Not bad today."],
        "speaker": ["Client", "Therapist", "Client", "Therapist"],
        "start": [0.0, 5.0, 10.0, 15.0],
        "end": [4.5, 9.0, 14.0, 20.0],
        "duration": [4.5, 4.0, 4.0, 5.0],
        "turn_id": [0, 1, 2, 3],
        "is_turn_start": [True, True, True, True],
    })


def _make_cache():
    return EmbeddingCache(
        embeddings=np.random.randn(4, 768),
        sentences=_make_df()["sentence"].tolist(),
    )


def _make_mock_model(n=4):
    model = MagicMock()
    model.batch_score.return_value = np.array([
        [-0.5, 0.3], [0.2, -0.1], [-0.6, 0.4], [0.1, -0.2]
    ])
    return model


# ---- AffectAnalyzer ----

def test_affect_analyzer_adds_valence_arousal_columns():
    from affect_analyzer.analyzers.affect import AffectAnalyzer
    result = AffectAnalyzer(model=_make_mock_model()).analyze(_make_df(), _make_cache())
    assert "valence" in result.per_sentence.columns
    assert "arousal" in result.per_sentence.columns
    assert result.name == "affect"


def test_affect_analyzer_uses_batch_score_exactly_once():
    from affect_analyzer.analyzers.affect import AffectAnalyzer
    model = _make_mock_model()
    AffectAnalyzer(model=model).analyze(_make_df(), _make_cache())
    model.batch_score.assert_called_once()
    args = model.batch_score.call_args[0][0]
    assert len(args) == 4


def test_affect_analyzer_global_metrics_per_speaker():
    from affect_analyzer.analyzers.affect import AffectAnalyzer
    result = AffectAnalyzer(model=_make_mock_model()).analyze(_make_df(), _make_cache())
    assert "Client_valence_mean" in result.global_metrics
    assert "Therapist_valence_mean" in result.global_metrics
