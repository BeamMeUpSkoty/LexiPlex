import numpy as np
import spacy
from unittest.mock import MagicMock

from affect_analyzer import AnalyzerRegistry, LexiPlexPipeline
from affect_analyzer.analyzers.affect import AffectAnalyzer
from affect_analyzer.analyzers.complexity import ComplexityAnalyzer
from affect_analyzer.analyzers.clinical import ClinicalMarkerAnalyzer
from affect_analyzer.analyzers.dynamics import DynamicsAnalyzer

DATA_CSV = "data/mock_therapy_session.csv"


def _mock_model():
    model = MagicMock()
    model.embed_sentences.side_effect = lambda s, **kw: np.zeros((len(s), 768))
    model.batch_score.side_effect = lambda s, **kw: np.zeros((len(s), 2))
    return model


def test_all_four_analyzers_run_and_return_results():
    model = _mock_model()
    nlp = spacy.load("en_core_web_sm")
    registry = AnalyzerRegistry()
    registry.register(AffectAnalyzer(model=model))
    registry.register(ComplexityAnalyzer(nlp=nlp))
    registry.register(ClinicalMarkerAnalyzer())
    registry.register(DynamicsAnalyzer())

    pipeline = LexiPlexPipeline(registry=registry, model=model, language="en")
    results = pipeline.run(DATA_CSV)

    assert set(results.keys()) == {"affect", "complexity", "clinical", "dynamics"}
    for name, result in results.items():
        assert len(result.per_sentence) > 0, f"{name} per_sentence is empty"


def test_affect_columns_present_in_integration():
    model = _mock_model()
    nlp = spacy.load("en_core_web_sm")
    registry = AnalyzerRegistry()
    registry.register(AffectAnalyzer(model=model))
    registry.register(ComplexityAnalyzer(nlp=nlp))
    registry.register(ClinicalMarkerAnalyzer())
    registry.register(DynamicsAnalyzer())

    pipeline = LexiPlexPipeline(registry=registry, model=model, language="en")
    results = pipeline.run(DATA_CSV)

    affect_df = results["affect"].per_sentence
    for col in ("valence", "arousal", "speaker", "turn_id", "is_turn_start"):
        assert col in affect_df.columns, f"Missing column: {col}"
