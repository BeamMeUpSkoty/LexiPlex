import argparse
import warnings
from typing import Any, Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from affect_analyzer.data_types.transcript import TranscriptFile
from affect_analyzer.preprocessing.language import LanguageProcessor
from affect_analyzer.features.extractor import FeatureExtractor
from affect_analyzer.topics.topic_modeler import TopicModeler


class ConversationAffectAnalyzer:
    """
    Orchestrates affect and topic analysis on a transcript, with per‐speaker metrics
    and Russell‐circumplex plots.

    Pipeline:
      1. Sentence‐level preprocessing
      2. Optional salience prefilter
      3. Topic modeling (global LDA, sliding‐window, or JS divergence)
      4. Valence/arousal scoring
      5. Compute global, per‐topic, and per‐speaker metrics
      6. Emit circumplex plots per speaker
    """

    def __init__(
        self,
        model_dir: Optional[str] = None,
        language: str = "en",
        device: str = "cpu",
        chunksize: int = 10000,
        model: Optional[Any] = None,
    ):
        # Load or inject valence/arousal model
        if model is not None:
            self.model = model
        else:
            try:
                from affect_analyzer.modelling.model import ValenceArousalModel
                self.model = ValenceArousalModel(
                    model_dir or "xlm-roberta-base", embed_device=device
                )
            except Exception as e:
                warnings.warn(
                    f"Valence/Arousal model load failed: {e}", RuntimeWarning
                )

                class DummyModel:
                    def batch_score(self, texts, batch_size=1, device="cpu"):
                        return np.zeros((len(texts), 2))

                    def compute_salience(self, sentences):
                        sims = np.zeros(len(sentences))
                        mask = np.ones(len(sentences), dtype=bool)
                        return sims, mask

                self.model = DummyModel()

        self.processor = LanguageProcessor(language)
        self.topic_modeler = TopicModeler()
        self.feature_extractor = FeatureExtractor(
            duration_col="duration", input_col="sentence"
        )
        self.device = device
        self.chunksize = chunksize

    def analyze(
        self,
        path: str,
        topic_method: str = "global",
        num_topics: int = 5,
        max_features: int = 1000,
        topic_keywords_top_n: int = 3,
        salience_top_percent: Optional[float] = None,
        window_size: int = 10,
        window_step: int = 5,
    ) -> Union[
        Tuple[Dict[str, float], pd.DataFrame, Dict[int, Dict[str, float]]],
        Tuple[Dict[str, float], pd.DataFrame, Dict[int, Dict[str, float]], pd.DataFrame],
    ]:
        """
        Execute the full pipeline.

        Returns:
          - global_metrics: overall valence/arousal
          - per_sentence: DataFrame of sentence‐level features
          - metrics_by_topic: dict topic→metrics (if topics used)
          - window_info: sliding‐window stats (for 'sliding' only)
        """
        # 1. Load & preprocess to sentences
        tf = TranscriptFile.open_file(
            path, language=self.processor.language, chunksize=self.chunksize
        )
        df = self.processor.preprocess_transcript(tf)
        if "duration" not in df:
            df["duration"] = df["end"] - df["start"]

        # 2. Optional salience filter
        if salience_top_percent is not None:
            sims, _ = self.model.compute_salience(df["sentence"].tolist())
            df["salience"] = sims
            k = max(1, int(len(df) * salience_top_percent / 100))
            top_idxs = np.argsort(sims)[-k:]
            df = df.iloc[top_idxs].reset_index(drop=True)

        # 3. Topic modeling
        window_info = None
        if topic_method == "global":
            df["topic"] = self.topic_modeler.fit_global_lda(
                df["sentence"], num_topics, max_features
            )
        elif topic_method == "sliding":
            df["topic"] = self.topic_modeler.fit_global_lda(
                df["sentence"], num_topics, max_features
            )
            df, window_info = self.topic_modeler.sliding_window_drift(
                df, window_size, window_step
            )
        elif topic_method == "js":
            df = self.topic_modeler.compute_js_divergence(df)
        else:
            raise ValueError(f"Unknown topic_method: {topic_method}")

        # 4. Register valence & arousal features
        fe = self.feature_extractor
        fe._features.clear()
        fe.register_feature(
            "valence",
            lambda txt: float(
                self.model.batch_score(
                    [txt], batch_size=1, device=self.device
                )[0, 0]
            ),
        )
        fe.register_feature(
            "arousal",
            lambda txt: float(
                self.model.batch_score(
                    [txt], batch_size=1, device=self.device
                )[0, 1]
            ),
        )

        # 5. Extract features
        extracted = fe.extract(df)
        global_metrics = extracted["global"]
        per_sentence = extracted["per_sentence"]

        # 6. Per-topic metrics & labels
        metrics_by_topic: Dict[int, Dict[str, float]] = {}
        if topic_method in ("global", "sliding"):
            labels = self.topic_modeler.make_topic_labels(topic_keywords_top_n)
            per_sentence["topic_label"] = per_sentence["topic"].map(labels)
            for topic, group in per_sentence.groupby("topic"):
                metrics_by_topic[int(topic)] = fe.compute_global_features(group)

        # 7. Return results
        if topic_method == "sliding":
            return global_metrics, per_sentence, metrics_by_topic, window_info

        return global_metrics, per_sentence, metrics_by_topic

    def plot_circumplex(
        self,
        per_sentence: pd.DataFrame,
        topic_labels: Optional[Dict[int, str]] = None,
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> None:
        """
        Scatter valence/arousal on the unit circle, color‐coded by topic.
        """
        fig, ax = plt.subplots()
        circle = plt.Circle((0, 0), 1, fill=False, linestyle="-")
        ax.add_artist(circle)
        ax.axhline(0, linestyle="--", color="gray")
        ax.axvline(0, linestyle="--", color="gray")
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_xlabel("Valence")
        ax.set_ylabel("Arousal")
        ax.set_title("Russell Circumplex of Affect")

        if "topic" in per_sentence:
            topics = np.sort(per_sentence["topic"].unique())
            cmap = plt.get_cmap("tab10", len(topics))
            for idx, topic in enumerate(topics):
                subset = per_sentence[per_sentence["topic"] == topic]
                label = (
                    topic_labels.get(topic) if topic_labels else f"Topic {topic}"
                )
                ax.scatter(
                    subset["valence"],
                    subset["arousal"],
                    color=cmap(idx),
                    label=label,
                    edgecolors="k",
                    s=50,
                )
            ax.legend(title="Topics", bbox_to_anchor=(1.05, 1), loc="upper left")
        else:
            ax.scatter(
                per_sentence["valence"],
                per_sentence["arousal"],
                color="blue",
                edgecolors="k",
                s=50,
            )

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches="tight")
        if show:
            plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze transcript for affect & topics."
    )
    parser.add_argument(
        "-i", "--input", required=True, help="Path to transcript CSV."
    )
    parser.add_argument(
        "-l", "--language", default="en", help="spaCy language code (e.g. 'en','de')."
    )
    parser.add_argument(
        "-m", "--model_dir", default=None, help="Valence/arousal model path or HF ID."
    )
    parser.add_argument(
        "--device", default="cpu", help="Inference device ('cpu' or 'cuda')."
    )
    parser.add_argument(
        "--chunksize", type=int, default=10000, help="CSV chunk size."
    )
    parser.add_argument(
        "--topic_method",
        choices=["global", "sliding", "js"],
        default="global",
        help="Topic strategy: global LDA, sliding-window drift, or JS divergence.",
    )
    parser.add_argument(
        "--num_topics", type=int, default=5, help="Number of LDA topics."
    )
    parser.add_argument(
        "--max_features", type=int, default=1000, help="Max vocabulary size for LDA."
    )
    parser.add_argument(
        "--topic_keywords_top_n",
        type=int,
        default=3,
        help="Top-N keywords (with weights) for topic labels.",
    )
    parser.add_argument(
        "--salience_top_percent",
        type=float,
        default=None,
        help="Pre-filter to top X% salient sentences.",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=10,
        help="Window length (in sentences) for sliding-window drift.",
    )
    parser.add_argument(
        "--window_step",
        type=int,
        default=5,
        help="Window step size (in sentences) for sliding-window drift.",
    )
    parser.add_argument(
        "--plot_circumplex",
        action="store_true",
        help="Plot Russell circumplex for valence/arousal.",
    )

    args = parser.parse_args()
    ca = ConversationAffectAnalyzer(
        model_dir=args.model_dir,
        language=args.language,
        device=args.device,
        chunksize=args.chunksize,
    )
    result = ca.analyze(
        args.input,
        topic_method=args.topic_method,
        num_topics=args.num_topics,
        max_features=args.max_features,
        topic_keywords_top_n=args.topic_keywords_top_n,
        salience_top_percent=args.salience_top_percent,
        window_size=args.window_size,
        window_step=args.window_step,
    )

    # Unpack
    if args.topic_method == "sliding":
        global_metrics, per_sentence, metrics_by_topic, window_info = result
    else:
        global_metrics, per_sentence, metrics_by_topic = result

    # Print global & per-topic metrics
    print("Global metrics:")
    for k, v in global_metrics.items():
        print(f"  {k}: {v:.4f}")
    print()

    if args.topic_method in ("global", "sliding"):
        print("Metrics by topic:")
        for t, m in metrics_by_topic.items():
            label = per_sentence.loc[per_sentence["topic"] == t, "topic_label"].iloc[0]
            print(f" Topic {t} [{label}]:")
            for kk, vv in m.items():
                print(f"   {kk}: {vv:.4f}")
        print()

    if args.topic_method == "sliding":
        print("Sliding-window info:")
        print(window_info)
        print()

    if args.plot_circumplex:
        # build topic_labels dict for legend
        if args.topic_method in ("global", "sliding"):
            labels = ca.topic_modeler.make_topic_labels(args.topic_keywords_top_n)
        else:
            labels = None
            ca.plot_circumplex(per_sentence, topic_labels=labels)
    else:
        print("Per-sentence (head):")
        print(per_sentence.head())
