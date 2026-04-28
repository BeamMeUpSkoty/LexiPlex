from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import pandas as pd

from .core.base import EmbeddingCache, AnalysisResult
from .core.registry import AnalyzerRegistry
from .data_types.transcript import TranscriptFile
from .preprocessing.language import LanguageProcessor

if TYPE_CHECKING:
    from .modelling.valence_arousal import ValenceArousalModel

logger = logging.getLogger(__name__)


def _add_turn_info(df: pd.DataFrame) -> pd.DataFrame:
    """Add turn_id (0-based) and is_turn_start based on consecutive speaker changes."""
    if "speaker" not in df.columns:
        df["turn_id"] = 0
        df["is_turn_start"] = False
        return df
    df = df.copy()
    first_speaker = df["speaker"].iloc[0]
    # NaN at row 0 from shift() means NaN != value is True, so cumsum starts at 1;
    # subtracting 1 gives 0-based IDs and row 0 correctly gets turn_id=0.
    df["turn_id"] = (df["speaker"] != df["speaker"].shift()).cumsum() - 1
    # Use fill_value=first_speaker so row 0 is False (no change at the start).
    df["is_turn_start"] = df["speaker"] != df["speaker"].shift(fill_value=first_speaker)
    return df


class LexiPlexPipeline:
    """Loads a transcript, embeds sentences once, and dispatches to all registered analyzers."""

    def __init__(
        self,
        registry: AnalyzerRegistry,
        model: Optional[ValenceArousalModel] = None,
        language: str = "en",
    ) -> None:
        self.registry = registry
        self.processor = LanguageProcessor(language)
        if model is not None:
            self.model = model
        else:
            from .modelling.valence_arousal import ValenceArousalModel
            self.model = ValenceArousalModel()

    def run(self, path: str) -> dict[str, AnalysisResult]:
        tf = TranscriptFile.open_file(path, language=self.processor.language)
        df = self.processor.preprocess_transcript(tf)

        if "duration" not in df.columns:
            df["duration"] = df["end"] - df["start"]
        df = _add_turn_info(df)

        logger.info("Loaded %d sentences from %s", len(df), path)

        cache = EmbeddingCache(
            embeddings=self.model.embed_sentences(df["sentence"].tolist()),
            sentences=df["sentence"].tolist(),
        )

        return self.registry.run_all(df, cache)
