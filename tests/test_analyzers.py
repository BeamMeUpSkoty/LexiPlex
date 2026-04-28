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


# ---- ComplexityAnalyzer ----

def test_complexity_analyzer_adds_expected_columns():
    from affect_analyzer.analyzers.complexity import ComplexityAnalyzer
    import spacy
    nlp = spacy.load("en_core_web_sm")
    result = ComplexityAnalyzer(nlp=nlp).analyze(_make_df(), _make_cache())
    for col in ("word_count", "type_token_ratio", "lexical_density", "coherence_to_prev"):
        assert col in result.per_sentence.columns, f"Missing column: {col}"
    assert result.name == "complexity"


def test_complexity_first_sentence_coherence_is_nan():
    from affect_analyzer.analyzers.complexity import ComplexityAnalyzer
    import spacy
    nlp = spacy.load("en_core_web_sm")
    result = ComplexityAnalyzer(nlp=nlp).analyze(_make_df(), _make_cache())
    first = result.per_sentence["coherence_to_prev"].iloc[0]
    assert pd.isna(first)


def test_complexity_global_metrics_per_speaker():
    from affect_analyzer.analyzers.complexity import ComplexityAnalyzer
    import spacy
    nlp = spacy.load("en_core_web_sm")
    result = ComplexityAnalyzer(nlp=nlp).analyze(_make_df(), _make_cache())
    assert "Client_ttr_mean" in result.global_metrics
    assert "Therapist_ttr_mean" in result.global_metrics
    assert "coherence_mean" in result.global_metrics


# ---- ClinicalMarkerAnalyzer ----

def test_clinical_analyzer_adds_expected_columns():
    from affect_analyzer.analyzers.clinical import ClinicalMarkerAnalyzer
    result = ClinicalMarkerAnalyzer().analyze(_make_df(), EmbeddingCache())
    for col in ("hedging_rate", "certainty_rate", "self_ref_rate", "negation_density", "is_question"):
        assert col in result.per_sentence.columns, f"Missing: {col}"
    assert result.name == "clinical"


def test_clinical_detects_hedging_and_not_certainty():
    from affect_analyzer.analyzers.clinical import ClinicalMarkerAnalyzer
    df = pd.DataFrame({
        "sentence": ["Maybe I'll go.", "I definitely will.", "Perhaps not."],
        "speaker": ["Client", "Client", "Client"],
        "start": [0.0, 5.0, 10.0], "end": [4.0, 9.0, 13.0],
        "duration": [4.0, 4.0, 3.0], "turn_id": [0, 1, 2],
        "is_turn_start": [True, False, False],
    })
    result = ClinicalMarkerAnalyzer().analyze(df, EmbeddingCache())
    hedge = result.per_sentence["hedging_rate"]
    assert hedge.iloc[0] > 0    # "Maybe"
    assert hedge.iloc[1] == 0   # "I definitely will" — no hedging
    assert hedge.iloc[2] > 0    # "Perhaps"


def test_clinical_detects_questions():
    from affect_analyzer.analyzers.clinical import ClinicalMarkerAnalyzer
    df = pd.DataFrame({
        "sentence": ["How are you?", "I am fine.", "Really?"],
        "speaker": ["Therapist", "Client", "Therapist"],
        "start": [0.0, 5.0, 10.0], "end": [3.0, 8.0, 11.0],
        "duration": [3.0, 3.0, 1.0], "turn_id": [0, 1, 2],
        "is_turn_start": [True, True, True],
    })
    result = ClinicalMarkerAnalyzer().analyze(df, EmbeddingCache())
    is_q = result.per_sentence["is_question"]
    assert is_q.iloc[0] == 1
    assert is_q.iloc[1] == 0
    assert is_q.iloc[2] == 1


# ---- DynamicsAnalyzer ----

def test_dynamics_analyzer_returns_per_speaker_metadata():
    from affect_analyzer.analyzers.dynamics import DynamicsAnalyzer
    result = DynamicsAnalyzer().analyze(_make_df(), _make_cache())
    assert result.name == "dynamics"
    assert "per_speaker" in result.metadata
    per_speaker = result.metadata["per_speaker"]
    assert "Client" in per_speaker.index
    assert "Therapist" in per_speaker.index


def test_dynamics_analyzer_latencies_in_metadata():
    from affect_analyzer.analyzers.dynamics import DynamicsAnalyzer
    result = DynamicsAnalyzer().analyze(_make_df(), _make_cache())
    assert "latencies" in result.metadata
    assert "silence_total" in result.metadata
    assert isinstance(result.metadata["latencies"], list)


def test_dynamics_dominance_sums_to_100():
    from affect_analyzer.analyzers.dynamics import DynamicsAnalyzer
    result = DynamicsAnalyzer().analyze(_make_df(), _make_cache())
    total = result.metadata["per_speaker"]["dominance_pct"].sum()
    assert abs(total - 100.0) < 0.1
