import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from typing import Dict


class TopicModeler:
    """
    Encapsulates three topic/dynamic‐segmenting strategies:
      - Global LDA
      - Sliding‐window topic drift
      - JS‐divergence information gain
    """

    def __init__(self, stop_words: str = 'english'):
        self.stop_words = stop_words
        self.vectorizer: CountVectorizer = None  # set on fit_global
        self.lda: LatentDirichletAllocation = None

    def fit_global_lda(self,
                       sentences: pd.Series,
                       num_topics: int,
                       max_features: int) -> pd.Series:
        """
        Fits LDA on all sentences and returns a topic‐assignment Series.
        Also caches vectorizer & LDA model for later labeling.
        """
        vec = CountVectorizer(
            max_features=max_features,
            stop_words=self.stop_words
        )
        dtm = vec.fit_transform(sentences)
        lda = LatentDirichletAllocation(
            n_components=num_topics,
            random_state=0
        )
        doc_topics = lda.fit_transform(dtm)
        topics = pd.Series(doc_topics.argmax(axis=1), index=sentences.index)

        # cache for later
        self.vectorizer = vec
        self.lda = lda
        return topics

    def sliding_window_drift(self,
                             df: pd.DataFrame,
                             window_size: int,
                             window_step: int) -> pd.DataFrame:
        """
        Annotates each row with window_id, dominant_topic, and window_entropy.
        Requires that df['topic'] and vectorizer/lda are already set.
        """
        # Get posterior per sentence
        dtm = self.vectorizer.transform(df['sentence'])
        post = self.lda.transform(dtm)

        windows = []
        for start in range(0, len(post) - window_size + 1, window_step):
            sub = post[start : start + window_size]
            mean_dist = sub.mean(axis=0)
            entropy = -np.sum(mean_dist * np.log2(mean_dist + 1e-12))
            dom = int(mean_dist.argmax())
            windows.append({
                'start_idx': start,
                'end_idx': start + window_size,
                'entropy': entropy,
                'dominant_topic': dom
            })

        win_df = pd.DataFrame(windows)

        # Assign window_id to each sentence
        df['window_id'] = -1
        for i, row in win_df.iterrows():
            df.loc[row.start_idx : row.end_idx - 1, 'window_id'] = i

        return df, win_df

    def compute_js_divergence(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes Jensen–Shannon divergence of each sentence vs. global unigram.
        Adds a 'js_divergence' column.
        """
        vec = CountVectorizer(stop_words=self.stop_words)
        dtm = vec.fit_transform(df['sentence'])
        freqs = np.asarray(dtm.sum(axis=0)).ravel()
        p_global = freqs / freqs.sum()

        def _js(sent: str) -> float:
            v = vec.transform([sent]).toarray().ravel()
            if v.sum() == 0:
                return 0.0
            p_sent = v / v.sum()
            return float(jensenshannon(p_sent, p_global, base=2))

        df = df.copy()
        df['js_divergence'] = df['sentence'].apply(_js)
        return df

    def make_topic_labels(self, top_n: int) -> Dict[int, str]:
        """
        Returns mapping topic_index → comma‐joined 'word(weight)' labels.
        """
        names = self.vectorizer.get_feature_names_out()
        labels: Dict[int, str] = {}
        for idx, comp in enumerate(self.lda.components_):
            top = np.argsort(comp)[::-1][:top_n]
            kws = [f"{names[i]}({comp[i]:.3f})" for i in top]
            labels[idx] = ", ".join(kws)
        return labels
