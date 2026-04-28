# LexiPlex

> **Alpha / Experimental** — APIs may change without notice. Not yet production-ready.

LexiPlex is a Python library for multi-dimensional language analysis of conversational transcripts. Given a timestamped CSV of a dialogue — a therapy session, a clinical interview, a customer service call — it runs four analyzers in a shared pipeline and produces sentence-level features alongside a five-chapter Streamlit demonstrator.

---

## Quick Start

### Installation

```bash
pip install -e .
```

spaCy language models are downloaded automatically on first run. XLM-RoBERTa model weights should be placed under `models/XLM-RoBERTa-base-MSE/`. If weights are absent the pipeline falls back to random scores (useful for development).

### Streamlit Demo

```bash
streamlit run app/streamlit_app.py
```

Loads `data/mock_therapy_session.csv` by default. Navigate the five chapters in the sidebar.

### CLI

```bash
lexiplex data/mock_therapy_session.csv output/ --language en
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `-m`, `--model` | `models/XLM-RoBERTa-base-MSE` | Model path or HuggingFace ID |
| `-l`, `--language` | `en` | spaCy language code (`en`, `de`, `fr`, `es`) |
| `-t`, `--topics` | off | Enable LDA topic assignment |
| `-v`, `--verbose` | off | DEBUG logging |

### Python API

```python
from affect_analyzer import AnalyzerRegistry, LexiPlexPipeline
from affect_analyzer.analyzers.affect import AffectAnalyzer
from affect_analyzer.analyzers.complexity import ComplexityAnalyzer
from affect_analyzer.analyzers.clinical import ClinicalMarkerAnalyzer
from affect_analyzer.analyzers.dynamics import DynamicsAnalyzer
from affect_analyzer.modelling.valence_arousal import ValenceArousalModel
import spacy

model = ValenceArousalModel("models/XLM-RoBERTa-base-MSE")
nlp = spacy.load("en_core_web_sm")

registry = AnalyzerRegistry()
registry.register(AffectAnalyzer(model=model))
registry.register(ComplexityAnalyzer(nlp=nlp))
registry.register(ClinicalMarkerAnalyzer())
registry.register(DynamicsAnalyzer())

pipeline = LexiPlexPipeline(registry=registry, model=model, language="en")
results = pipeline.run("data/mock_therapy_session.csv")

# results is a dict keyed by analyzer name
affect_df = results["affect"].per_sentence
print(affect_df[["sentence", "speaker", "valence", "arousal"]].head())
print(results["affect"].global_metrics)
```

---

## Input Format

LexiPlex expects a UTF-8 CSV with the following columns:

| Column    | Type    | Required | Description                                    |
|-----------|---------|----------|------------------------------------------------|
| `start`   | float   | Yes      | Utterance start time in seconds                |
| `end`     | float   | Yes      | Utterance end time in seconds                  |
| `text`    | string  | Yes      | Spoken text of the utterance                   |
| `speaker` | string  | No       | Speaker label — enables per-speaker breakdowns |
| `label`   | string  | No       | Optional annotation label                      |

**Example** (from `data/mock_therapy_session.csv`):

```
start,end,text,speaker
0.0,3.3,"Hello, how are you feeling today?",Therapist
4.66,8.7,I've been having trouble sleeping and feeling anxious.,Client
9.43,13.56,Can you tell me more about what's been keeping you up at night?,Therapist
```

---

## The Four Analyzers

Each analyzer implements `BaseAnalyzer` and produces an `AnalysisResult` with `per_sentence` (DataFrame) and `global_metrics` (dict).

### AffectAnalyzer

Scores each sentence on **valence** (pleasant ↔ unpleasant) and **arousal** (activated ↔ calm) using XLM-RoBERTa `batch_score()` — a single batched forward pass for the whole transcript.

Columns added: `valence`, `arousal`, optionally `topic`, `topic_label`

Global metrics: duration-weighted `{speaker}_valence_mean`, `{speaker}_arousal_mean` per speaker.

### ComplexityAnalyzer

Measures vocabulary richness, sentence length, and discourse coherence. Coherence reuses sentence embeddings already computed by the pipeline — no extra inference.

Columns added: `word_count`, `type_token_ratio`, `lexical_density`, `coherence_to_prev`

Global metrics: `coherence_mean`, `{speaker}_ttr_mean`, `{speaker}_wc_mean`, `{speaker}_density_mean`

### ClinicalMarkerAnalyzer

Rule-based detection of clinical language markers. No model required — fast, interpretable, works offline.

Columns added: `hedging_rate`, `certainty_rate`, `self_ref_rate`, `negation_density`, `is_question`

Global metrics: all rates per speaker.

**Health context:** High hedging + low certainty = avoidant language. High self-reference + negation are established markers of depressive cognition (Beck, 1979).

### DynamicsAnalyzer

Conversational structure from timestamps alone. Operates at the utterance/turn level.

`per_sentence` columns added: `turn_id`, `is_turn_start`

`metadata["per_speaker"]`: DataFrame with `turns`, `total_duration`, `dominance_pct`, `mean_utterance_length`

`metadata["latencies"]`: list of inter-turn gaps (seconds)

Global metrics: `silence_total`, `latency_mean`, `turn_count`

---

## Extending with a Custom Analyzer

Subclass `BaseAnalyzer`, implement `name` and `analyze`, register it:

```python
from affect_analyzer.core.base import BaseAnalyzer, AnalysisResult, EmbeddingCache
import pandas as pd

class WordCountAnalyzer(BaseAnalyzer):
    @property
    def name(self) -> str:
        return "word_count"

    def analyze(self, df: pd.DataFrame, cache: EmbeddingCache) -> AnalysisResult:
        out = df.copy()
        out["wc"] = out["sentence"].str.split().str.len()
        return AnalysisResult(
            name=self.name,
            per_sentence=out,
            global_metrics={"wc_mean": float(out["wc"].mean())},
        )

registry.register(WordCountAnalyzer())
```

The new analyzer is automatically included in every `pipeline.run()` call and in the live scorecards if you extend the Streamlit app.

---

## Architecture

```
affect_analyzer/
├── core/
│   ├── base.py           BaseAnalyzer, AnalysisResult, EmbeddingCache
│   └── registry.py       AnalyzerRegistry
├── pipeline.py           LexiPlexPipeline — load → embed once → dispatch
├── analyzers/
│   ├── affect.py         AffectAnalyzer
│   ├── complexity.py     ComplexityAnalyzer
│   ├── clinical.py       ClinicalMarkerAnalyzer
│   └── dynamics.py       DynamicsAnalyzer
├── data_types/
│   └── transcript.py     TranscriptFile — chunked CSV ingestion
├── preprocessing/
│   └── language.py       LanguageProcessor — spaCy sentence splitting & tokenization
├── modelling/
│   └── valence_arousal.py  ValenceArousalModel — XLM-RoBERTa regression + embeddings
├── topics/
│   └── topic_modeler.py  TopicModeler — LDA, sliding-window drift, JS divergence
├── plotting/
│   └── circumplex.py     plot_circumplex — matplotlib Russell circumplex
└── cli.py                lexiplex CLI entry point

app/
└── streamlit_app.py      Five-chapter guided walkthrough with temporal affect playback
```

**Pipeline flow:** `LexiPlexPipeline.run()` loads the transcript, sentence-splits it, computes sentence embeddings **once** into a shared `EmbeddingCache`, then calls `registry.run_all(df, cache)`. Every registered analyzer receives the same DataFrame and the same cache — `ComplexityAnalyzer` uses the cached embeddings for coherence scoring without triggering a second model pass.

---

## The Model & Affect Theory

**Russell's circumplex model of affect** (1980) proposes that all emotional states can be described by two independent dimensions:

- **Valence** — pleasantness (−1 = maximally unpleasant, +1 = maximally pleasant)
- **Arousal** — activation (−1 = maximally calm, +1 = maximally activated)

```
             High Arousal
                  ↑
   Tense/Anxious  |  Excited/Elated
                  |
Negative ←────────┼────────→ Positive
(Unpleasant)      |         (Pleasant)
                  |
   Depressed/Sad  |  Calm/Relaxed
                  ↓
             Low Arousal
```

The bundled model (`models/XLM-RoBERTa-base-MSE/`) is an XLM-RoBERTa-base checkpoint fine-tuned with MSE loss to regress (valence, arousal) from text. Outputs are clamped to [−1, 1] via `Hardtanh`. The multilingual backbone supports English, German, French, Spanish, and other languages covered by XLM-RoBERTa's pretraining.

See `papers/Acircumplexmodelofaffect.pdf` for the theoretical foundation.

---

## Project Status

LexiPlex is an **alpha research prototype** with 26 automated tests covering the core library, all four analyzers, and the full pipeline integration.

- **APIs may change.** Class interfaces and CLI flags are not yet stable.
- **Model weights not in the repository.** Place XLM-RoBERTa weights under `models/XLM-RoBERTa-base-MSE/`. Without them the pipeline uses random scores.
- **Transcripts only.** LexiPlex operates on text. It assumes ASR has already been run if your source is audio.
- **Language support** depends on spaCy model availability. Only `en`, `de`, `fr`, and `es` are explicitly mapped; other languages fall back to `{lang}_core_news_sm`.
