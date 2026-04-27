from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class EmbeddingCache:
    """Sentence embeddings computed once by the pipeline, shared across all analyzers."""
    embeddings: Optional[np.ndarray] = None
    sentences: Optional[list[str]] = None


@dataclass
class AnalysisResult:
    """Standardised output from any analyzer."""
    name: str
    per_sentence: pd.DataFrame
    global_metrics: dict[str, float]
    metadata: dict = field(default_factory=dict)


class BaseAnalyzer(ABC):
    """Interface all analyzers must implement."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def analyze(self, df: pd.DataFrame, cache: EmbeddingCache) -> AnalysisResult: ...
