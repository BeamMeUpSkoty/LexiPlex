import logging
from .data_types.transcript import TranscriptFile
from .preprocessing.language import LanguageProcessor
from .modeling.valence_arousal import ValenceArousalModel
from .topics.topic_modeler import TopicModeler
from .features.extractor import FeatureExtractor
from .plotting.circumplex import plot_circumplex
import pandas as pd

logger = logging.getLogger(__name__)

class ConversationPipeline:
    def __init__(
        self,
        transcript_path: str,
        output_dir: str,
        model_name: str = None,
        num_topics: int = 5,
        **kwargs
    ):
        self.transcript_path = transcript_path
        self.output_dir = output_dir
        self.processor = LanguageProcessor()
        self.model = ValenceArousalModel(model_name)
        self.topic_modeler = TopicModeler(num_topics=num_topics)
        self.fe = FeatureExtractor()

    def run(self):
        # 1. Load and split into sentences
        df = TranscriptFile.load(self.transcript_path).to_sentence_df()
        logger.info("Loaded transcript with %d sentences", len(df))

        # 2. Preprocess text
        df['clean'] = self.processor.clean_series(df['sentence'])
        logger.info("Preprocessed text")

        # 3. Topic modeling
        df['topic'] = self.topic_modeler.assign_topics(df['clean'])
        logger.info("Assigned topics")

        # 4. Compute valence/arousal
        scores = self.model.score_batch(df['clean'].tolist())
        df[['valence', 'arousal']] = scores
        logger.info("Computed valence/arousal scores")

        # 5. Plot per speaker circumplex
        for speaker, group in df.groupby('speaker'):
            fig = plot_circumplex(group, 'speaker')
            fig.savefig(f"{self.output_dir}/{speaker}_circumplex.png")
            logger.debug("Saved circumplex for %s", speaker)

        # 6. Plot per topic circumplex
        fig = plot_circumplex(df, 'topic')
        fig.savefig(f"{self.output_dir}/topic_circumplex.png")
        logger.info("Saved topic circumplex")

        # 7. Export CSV with features
        out_csv = f"{self.output_dir}/conversation_features.csv"
        df.to_csv(out_csv, index=False)
        logger.info("Exported features to %s", out_csv)

        return df


if __name__ == '__main__':  # for ad-hoc runs
    import argparse

    parser = argparse.ArgumentParser(description="Run Conversation Affect Analysis")
    parser.add_argument('transcript', help='Path to transcript file')
    parser.add_argument('output', help='Directory to write outputs')
    parser.add_argument('--model', help='HuggingFace model name', default=None)
    parser.add_argument('--topics', type=int, help='Number of topics', default=5)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    pipeline = ConversationPipeline(
        transcript_path=args.transcript,
        output_dir=args.output,
        model_name=args.model,
        num_topics=args.topics
    )
    pipeline.run()
