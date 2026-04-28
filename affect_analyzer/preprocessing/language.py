import pandas as pd
import importlib
import subprocess
import sys
import warnings
import spacy
from typing import List, Iterable, Optional

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
    sentence splitting, and tokenization.
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
        transcript: 'TranscriptFile'
    ) -> pd.DataFrame:
        """
        Convert transcript to sentence-level DataFrame with tokens.

        Parameters
        ----------
        transcript : TranscriptFile
            Instance with read_chunks() yielding DataFrames.

        Returns
        -------
        pd.DataFrame
            Columns: original metadata, 'sentence', 'tokens'.
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
        return pd.DataFrame.from_records(records)

