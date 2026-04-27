import numpy as np
import pandas as pd
from typing import List, Dict

from affect_analyzer.modelling.model import ValenceArousalModel
from affect_analyzer.features.extractor import FeatureExtractor


class SentimentFeatureExporter:
    """
    Wraps your ValenceArousalModel and FeatureExtractor to
    compute sentence‐ and word‐level affect features, then
    dump everything to a CSV.
    """

    def __init__(self,
                 model_dir: str = "xlm-roberta-base",
                 device: str = "cpu"):
        # 1) Load your existing model
        self.model = ValenceArousalModel(model_dir, embed_device=device)
        self.device = device

        # 2) Prepare a FeatureExtractor just for sentence‐level valence/arousal
        self.fe = FeatureExtractor(duration_col=None, input_col="sentence")
        self.fe.register_feature(
            "valence",
            lambda s: float(
                self.model.batch_score([s], batch_size=1, device=self.device)[0, 0]
            )
        )
        self.fe.register_feature(
            "arousal",
            lambda s: float(
                self.model.batch_score([s], batch_size=1, device=self.device)[0, 1]
            )
        )

    def word_level_stats(self, sentence: str) -> Dict[str, float]:
        """
        Compute word‐level valence/arousal stats for a single sentence.
        """
        words = sentence.split()
        if not words:
            return {
                "val_mean": np.nan, "val_min": np.nan, "val_max": np.nan,
                "val_std": np.nan,  "val_range": np.nan,
                "aro_mean": np.nan, "aro_min": np.nan, "aro_max": np.nan,
                "aro_std": np.nan,  "aro_range": np.nan
            }

        # batch_score directly on words
        scores = self.model.batch_score(words, batch_size=len(words),
                                        device=self.device).numpy()
        vals = scores[:, 0]
        aros = scores[:, 1]

        return {
            "val_mean":  float(np.mean(vals)),
            "val_min":   float(np.min(vals)),
            "val_max":   float(np.max(vals)),
            "val_std":   float(np.std(vals)),
            "val_range": float(np.ptp(vals)),
            "aro_mean":  float(np.mean(aros)),
            "aro_min":   float(np.min(aros)),
            "aro_max":   float(np.max(aros)),
            "aro_std":   float(np.std(aros)),
            "aro_range": float(np.ptp(aros)),
        }

    def export(
        self,
        sentences: List[str],
        output_csv: str = "sentiment_features.csv"
    ) -> None:
        """
        Compute all features for each sentence and save to CSV.

        Columns:
          - sentence
          - valence
          - arousal
          - val_mean, val_min, val_max, val_std, val_range
          - aro_mean, aro_min, aro_max, aro_std, aro_range
        """
        # 1) Sentence‐level valence/arousal
        df = pd.DataFrame({"sentence": sentences})
        df_sent = self.fe.extract_sentence_features(df)

        # 2) Word‐level statistics
        stats = df_sent["sentence"].apply(self.word_level_stats)
        df_stats = pd.DataFrame(stats.tolist())

        # 3) Combine & save
        out = pd.concat([df_sent, df_stats], axis=1)
        out.to_csv(output_csv, index=False)
        print(f"Saved {len(out)} rows of features to {output_csv}")
