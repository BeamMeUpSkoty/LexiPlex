from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.base import BaseAnalyzer, AnalysisResult, EmbeddingCache


class DynamicsAnalyzer(BaseAnalyzer):
    """
    Conversational structure from utterance timestamps.
    Requires 'speaker', 'start', 'end', 'turn_id', 'is_turn_start' columns.
    """

    @property
    def name(self) -> str:
        return "dynamics"

    def analyze(self, df: pd.DataFrame, cache: EmbeddingCache) -> AnalysisResult:
        out = df.copy()

        if "speaker" not in df.columns or "turn_id" not in df.columns:
            return AnalysisResult(
                name=self.name,
                per_sentence=out,
                global_metrics={},
                metadata={"error": "requires speaker and turn_id columns"},
            )

        turns = (
            df.groupby("turn_id")
            .agg(speaker=("speaker", "first"), start=("start", "first"), end=("end", "last"))
            .reset_index()
        )
        turns["duration"] = turns["end"] - turns["start"]

        total_duration = turns["duration"].sum()
        per_speaker = (
            turns.groupby("speaker")
            .agg(
                turns=("turn_id", "count"),
                total_duration=("duration", "sum"),
                mean_utterance_length=("duration", "mean"),
            )
            .reset_index()
        )
        per_speaker["dominance_pct"] = (
            per_speaker["total_duration"] / total_duration * 100
        ).round(2)
        per_speaker = per_speaker.set_index("speaker")

        sorted_turns = turns.sort_values("start")
        latencies = [
            max(0.0, float(v))
            for v in (sorted_turns["start"].iloc[1:].values - sorted_turns["end"].iloc[:-1].values)
        ]

        global_metrics = {
            "silence_total": float(sum(latencies)),
            "latency_mean": float(np.mean(latencies)) if latencies else 0.0,
            "turn_count": float(len(turns)),
        }

        return AnalysisResult(
            name=self.name,
            per_sentence=out,
            global_metrics=global_metrics,
            metadata={
                "per_speaker": per_speaker,
                "latencies": latencies,
                "silence_total": float(sum(latencies)),
            },
        )
