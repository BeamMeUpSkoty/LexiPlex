import pandas as pd
import importlib
import subprocess
import sys
import warnings
import spacy
from typing import List, Iterable, Optional
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

def _ensure_spacy_model(model_name: str):
    """
    Ensure that the specified spaCy model is installed; if not, download it.
    """
    try:
        importlib.import_module(model_name)
    except ModuleNotFoundError:
        warnings.warn(f"spaCy model '{model_name}' not found. Downloading...", RuntimeWarning)
        subprocess.run([sys.executable, "-m", "spacy", "download", model_name], check=True)

class LanguageProcessor:
    """
    Efficient processor for text in various languages.
    Ensures spaCy models are installed, caches pipelines, provides text cleanup,
    sentence splitting, tokenization, and topic identification via LDA.
    """
    _nlp_cache = {}

    def __init__(
        self,
        language: str = 'en',
        disable: Optional[List[str]] = None
    ):
        """
        Initialize a processor for the specified language.

        Parameters
        ----------
        language : str
            Two-letter language code (e.g., 'en', 'de').
        disable : List[str], optional
            Pipeline components to disable (e.g., ['ner']).
            Default disables 'ner' and parser, using sentencizer.
        """
        self.language = language
        self.disable = disable or ['ner']
        self.nlp = self._load_pipeline(language, self.disable)
        # placeholders for topic modeling
        self._topic_vectorizer = None
        self._topic_model = None

    @classmethod
    def _load_pipeline(cls, language: str, disable: List[str]):
        model_name = cls.get_spacy_model_name(language)
        _ensure_spacy_model(model_name)
        pipes_to_disable = set(disable) | {'parser'}
        if language not in cls._nlp_cache:
            nlp = spacy.load(model_name, disable=list(pipes_to_disable))
            # add sentencizer if parser disabled
            if 'parser' in pipes_to_disable and 'sentencizer' not in nlp.pipe_names:
                nlp.add_pipe('sentencizer')
            cls._nlp_cache[language] = nlp
        return cls._nlp_cache[language]

    @staticmethod
    def get_spacy_model_name(language: str) -> str:
        mapping = {
            'en': 'en_core_web_sm',
            'de': 'de_core_news_sm',
            'fr': 'fr_core_news_sm',
            'es': 'es_core_news_sm',
        }
        return mapping.get(language, f"{language}_core_news_sm")

    def clean_text(self, text: str) -> str:
        return text.strip()

    def split_sentences(self, text: str) -> List[str]:
        doc = self.nlp(text)
        return [sent.text.strip() for sent in doc.sents]

    def tokenize(self, text: str) -> List[str]:
        doc = self.nlp(text)
        return [token.text for token in doc if not token.is_space]

    def preprocess_text(self, text: str) -> List[List[str]]:
        cleaned = self.clean_text(text)
        sentences = self.split_sentences(cleaned)
        return [self.tokenize(sent) for sent in sentences]

    def preprocess_corpus(self, texts: Iterable[str]) -> List[List[List[str]]]:
        return [self.preprocess_text(t) for t in texts]

    def preprocess_transcript(
        self,
        transcript: 'TranscriptFile',
        assign_topics: bool = False,
        num_topics: int = 5,
        max_features: int = 1000
    ) -> pd.DataFrame:
        """
        Convert transcript to sentence-level DataFrame with tokens,
        optionally assign topic labels.

        Parameters
        ----------
        transcript : TranscriptFile
            Instance with read_chunks() yielding DataFrames.
        assign_topics : bool
            If True, fit LDA and assign topic to each sentence.
        num_topics : int
            Number of LDA topics.
        max_features : int
            Max vocabulary size for topic vectorizer.

        Returns
        -------
        pd.DataFrame
            Columns: original metadata, 'sentence', 'tokens', optional 'topic'.
        """
        records = []
        for chunk in transcript.read_chunks():
            chunk['text'] = chunk['text'].apply(self.clean_text)
            for _, row in chunk.iterrows():
                for sent in self.split_sentences(row['text']):
                    tokens = self.tokenize(sent)
                    rec = row.to_dict()
                    rec['sentence'] = sent
                    rec['tokens'] = tokens
                    records.append(rec)
        df = pd.DataFrame.from_records(records)
        # Optional topic modeling
        if assign_topics:
            sentences = df['sentence'].tolist()
            self.fit_topic_model(sentences, num_topics, max_features)
            df['topic'] = self.assign_topics(sentences)
        return df

    def fit_topic_model(
        self,
        sentences: List[str],
        num_topics: int = 5,
        max_features: int = 1000
    ) -> None:
        """
        Fit an LDA topic model on given sentences.

        Parameters
        ----------
        sentences : list of str
        num_topics : int
        max_features : int
        """
        vectorizer = CountVectorizer(max_features=max_features, token_pattern=r"\b\w+\b")
        term_matrix = vectorizer.fit_transform(sentences)
        lda = LatentDirichletAllocation(n_components=num_topics, random_state=0)
        lda.fit(term_matrix)
        self._topic_vectorizer = vectorizer
        self._topic_model = lda

    def assign_topics(self, sentences: List[str]) -> List[int]:
        """
        Assign most probable topic to each sentence.
        """
        if self._topic_model is None or self._topic_vectorizer is None:
            raise ValueError("Topic model not fitted. Call fit_topic_model first.")
        term_matrix = self._topic_vectorizer.transform(sentences)
        topic_dist = self._topic_model.transform(term_matrix)
        return topic_dist.argmax(axis=1).tolist()

    def get_topic_keywords(self, topic_index: int, top_n: int = 10) -> List[str]:
        """
        Retrieve top keywords for a given topic.
        """
        if self._topic_model is None or self._topic_vectorizer is None:
            raise ValueError("Topic model not fitted. Call fit_topic_model first.")
        feature_names = self._topic_vectorizer.get_feature_names_out()
        top_ids = self._topic_model.components_[topic_index].argsort()[::-1][:top_n]
        return [feature_names[i] for i in top_ids]

