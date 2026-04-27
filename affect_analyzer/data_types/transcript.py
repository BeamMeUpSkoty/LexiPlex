import pandas as pd

class TranscriptFile:
    """
    Transcript stored in CSV format with utterance-level annotations.

    Expected CSV schema: 'start', 'end', 'text', optional 'label', optional 'speaker'.
    """

    @staticmethod
    def open_file(
        path: str,
        language: str,
        chunksize: int = 10_000,
        encoding: str = 'utf-8'
    ) -> 'TranscriptFile':
        """
        Factory method to instantiate a TranscriptFile.

        Parameters
        ----------
        path : str
            Path to the CSV file.
        language : str
            Two-letter language code.
        chunksize : int, optional
            Number of rows to read per chunk. Default 10000.
        encoding : str, optional
            File encoding. Default 'utf-8'.

        Returns
        -------
        TranscriptFile
        """
        return TranscriptFile(path, language, chunksize, encoding)

    def __init__(
        self,
        path: str,
        language: str,
        chunksize: int = 10_000,
        encoding: str = 'utf-8'
    ):
        """
        Initialize TranscriptFile reader.

        Parameters
        ----------
        path : str
            Path to the CSV file.
        language : str
            Two-letter language code.
        chunksize : int, optional
            Number of rows to read per chunk. Default 10000.
        encoding : str, optional
            File encoding. Default 'utf-8'.
        """
        self.path = path
        self.language = language
        self.source_type = 'transcript'
        self.chunksize = chunksize
        self.encoding = encoding
        # Define expected dtypes
        self._dtypes = {
            'start': 'float32',
            'end': 'float32',
            'text': 'string',
            'label': 'category',
            'speaker': 'category',
        }
        # Determine which expected columns are actually present
        self.available_cols = list(pd.read_csv(self.path, nrows=0).columns)
        self.usecols = [c for c in self._dtypes if c in self.available_cols]
        # Subset dtypes to present columns
        self.dtypes = {c: self._dtypes[c] for c in self.usecols}

    def read_chunks(self) -> pd.io.parsers.TextFileReader:
        """
        Lazily read CSV in chunks using only available columns.

        Returns
        -------
        Iterator of DataFrame chunks
        """
        return pd.read_csv(
            self.path,
            dtype=self.dtypes,
            usecols=self.usecols,
            chunksize=self.chunksize,
            encoding=self.encoding
        )

    def iter_utterances(self):
        """
        Iterate over individual utterances row by row.

        Yields
        ------
        dict
            Records containing the available columns.
        """
        for chunk in self.read_chunks():
            for rec in chunk.to_dict(orient='records'):
                yield rec

    def to_dataframe(self) -> pd.DataFrame:
        """
        Load entire transcript into a single DataFrame.

        Returns
        -------
        pd.DataFrame
        """
        return pd.concat(self.read_chunks(), ignore_index=True)

    def __len__(self) -> int:
        """
        Count total number of utterances (rows) in the CSV.

        Returns
        -------
        int
        """
        return sum(len(chunk) for chunk in self.read_chunks())

    def __repr__(self) -> str:
        total = None
        try:
            total = len(self)
        except Exception:
            total = 'unknown'
        return (
            f"<TranscriptFile path={self.path!r}, language={self.language!r}, "
            f"chunksize={self.chunksize}, total_utterances={total}>"
        )

    def make_alternating_speaker_df(self) -> pd.DataFrame:
        """
        Merge contiguous utterances by the same speaker, then alternate rows.

        Returns
        -------
        pd.DataFrame
            Columns: ['start', 'end', 'duration', 'speaker']
        """
        df = self.to_dataframe()
        if 'speaker' not in df.columns:
            raise ValueError("'speaker' column not found in transcript.")
        # Mark groups of contiguous identical speakers
        df['group'] = (df['speaker'] != df['speaker'].shift()).cumsum()
        turns = df.groupby('group').agg(
            start=('start', 'first'),
            end=('end', 'last'),
            speaker=('speaker', 'first')
        )
        turns['duration'] = turns['end'] - turns['start']
        return turns.reset_index(drop=True)[['start', 'end', 'duration', 'speaker']]
