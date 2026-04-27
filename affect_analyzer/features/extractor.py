import pandas as pd
from typing import Callable, Dict, Any

class FeatureExtractor:
    """
    Modular feature extraction for utterance/sentence-level DataFrames,
    supporting both per-item and global (duration-weighted) aggregation.

    Allows registering arbitrary text-based feature functions.
    """
    def __init__(self,
                 duration_col: str = 'duration',
                 input_col: str = 'sentence'):
        """
        Parameters
        ----------
        duration_col : str
            Column name representing duration or weight for global aggregation.
        input_col : str
            Column name of text input to feature functions (e.g., 'sentence' or 'text').
        """
        self.duration_col = duration_col
        self.input_col = input_col
        self._features: Dict[str, Callable[[Any], float]] = {}

    def register_feature(self, name: str, func: Callable[[Any], float]) -> None:
        """
        Register a feature function.

        Parameters
        ----------
        name : str
            Feature name; will be used as a new column in the DataFrame.
        func : Callable[[Any], float]
            Function mapping an input (e.g., text or tokens) to a numeric feature.
        """
        if name in self._features:
            raise KeyError(f"Feature '{name}' already registered.")
        self._features[name] = func

    def extract_sentence_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all registered feature functions to each row of the DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain column `self.input_col` for input to feature funcs.

        Returns
        -------
        pd.DataFrame
            Copy of df with new feature columns.
        """
        df_feat = df.copy()
        for name, func in self._features.items():
            df_feat[name] = df_feat[self.input_col].apply(func)
        return df_feat

    def compute_global_features(self, df_feat: pd.DataFrame) -> Dict[str, float]:
        """
        Compute global (duration-weighted) feature averages.

        Parameters
        ----------
        df_feat : pd.DataFrame
            DataFrame returned by `extract_sentence_features`, containing
            numeric feature columns and `self.duration_col`.

        Returns
        -------
        Dict[str, float]
            Mapping feature names to their global weighted averages.
        """
        if self.duration_col not in df_feat:
            raise KeyError(f"Duration column '{self.duration_col}' not in DataFrame.")
        total_weight = df_feat[self.duration_col].sum()
        if total_weight == 0:
            raise ValueError("Total duration is zero, cannot compute weighted global features.")

        global_feats = {}
        for name in self._features:
            weighted_sum = (df_feat[name] * df_feat[self.duration_col]).sum()
            global_feats[f"{name}_global"] = weighted_sum / total_weight
        return global_feats

    def extract(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Convenience method: extracts sentence-level features and computes globals.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame of utterances or sentences.

        Returns
        -------
        Dict with keys:
          - 'per_sentence': DataFrame with feature columns
          - 'global': dict of global feature values
        """
        df_sent = self.extract_sentence_features(df)
        global_feats = self.compute_global_features(df_sent)
        return {'per_sentence': df_sent, 'global': global_feats}

# Example usage:
# fe = FeatureExtractor(duration_col='duration', input_col='sentence')
# fe.register_feature('length', lambda s: len(s.split()))
# fe.register_feature('exclamations', lambda s: s.count('!'))
# df_features = fe.extract_sentence_features(df_sent)
# globals = fe.compute_global_features(df_features)
# Or combined:
# result = fe.extract(df_sent)
