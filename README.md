# LexiPlex

> **Alpha / Experimental** — APIs may change without notice. Not yet production-ready.

LexiPlex is a Python toolkit for affect analysis of conversational transcripts. Given a timestamped CSV of a dialogue — a therapy session, a clinical interview, a customer service call — it scores each sentence on **valence** (pleasant ↔ unpleasant) and **arousal** (activated ↔ deactivated), assigns topic labels, and plots the results on a Russell circumplex. The output is both human-readable (circumplex PNGs) and machine-readable (a feature CSV), making it useful for researchers, developers, and clinicians alike.

---

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

spaCy language models are downloaded automatically on first run. The XLM-RoBERTa model weights are bundled under `models/XLM-RoBERTa-base-MSE/` — no separate download required.

### CLI

```bash
python conversation_affect_analyzer.py \
  --input data/mock_therapy_session.csv \
  --topic_method global \
  --num_topics 5 \
  --plot_circumplex
```

This will print per-sentence valence/arousal scores and topic assignments to stdout, and display a circumplex plot. To save plots and the feature CSV programmatically, use `ConversationPipeline` directly (see Architecture Overview).

### Python API

```python
from conversation_affect_analyzer import ConversationAffectAnalyzer

ca = ConversationAffectAnalyzer(
    model_dir="models/XLM-RoBERTa-base-MSE",
    language="en",
    device="cpu",
)

global_metrics, per_sentence, metrics_by_topic = ca.analyze(
    "data/mock_therapy_session.csv",
    topic_method="global",
    num_topics=5,
)

print(global_metrics)
print(per_sentence[["sentence", "speaker", "valence", "arousal", "topic"]].head())
```

---

## Input Format

LexiPlex expects a UTF-8 CSV with the following columns:

| Column    | Type    | Required | Description                                      |
|-----------|---------|----------|--------------------------------------------------|
| `start`   | float   | Yes      | Utterance start time in seconds                  |
| `end`     | float   | Yes      | Utterance end time in seconds                    |
| `text`    | string  | Yes      | Spoken text of the utterance                     |
| `speaker` | string  | No       | Speaker label — enables per-speaker circumplex   |
| `label`   | string  | No       | Optional annotation label                        |

**Example** (from `data/mock_therapy_session.csv`):

```
start,end,text,speaker
0.0,3.3,"Hello, how are you feeling today?",Therapist
4.66,8.7,I've been having trouble sleeping and feeling anxious.,Client
9.43,13.56,Can you tell me more about what's been keeping you up at night?,Therapist
```

Any two-speaker dialogue fits this format — clinical interviews, customer service calls, research conversations. The `speaker` column is optional but unlocks per-speaker affect breakdowns.

---

## Outputs

Running the pipeline produces:

| Output | Description |
|--------|-------------|
| `conversation_features.csv` | Sentence-level DataFrame: `sentence`, `speaker`, `valence`, `arousal`, `topic`, `topic_label`, and optional `salience` |
| `{speaker}_circumplex.png` | Scatter plot of valence vs. arousal for each speaker |
| `topic_circumplex.png` | Scatter plot color-coded by LDA topic |

### Reading a Circumplex Plot

The circumplex places every sentence in a 2D emotional space bounded by the unit circle:

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

Points in the upper-left quadrant (high arousal, negative valence) indicate distress. Points in the lower-right (low arousal, positive valence) indicate calm contentment. Clustering patterns across speakers or topics reveal emotional dynamics in the conversation.

---

## Architecture Overview

```
affect_analyzer/
├── data_types/transcript.py      TranscriptFile     — chunked CSV ingestion
├── preprocessing/language.py     LanguageProcessor  — spaCy pipeline, sentence splitting, tokenization
├── modelling/valence_arousal.py  ValenceArousalModel — XLM-RoBERTa regression + embeddings + salience
├── topics/topic_modeler.py       TopicModeler       — global LDA, sliding-window drift, JS divergence
├── features/extractor.py         FeatureExtractor   — plug-in registry for custom sentence features
├── plotting/circumplex.py        plot_circumplex    — Russell circumplex scatter plot
├── pipeline.py                   ConversationPipeline — CLI-facing orchestrator
└── cli.py                        main               — Click CLI entry point

conversation_affect_analyzer.py   ConversationAffectAnalyzer — full-featured Python API orchestrator
sentiment_exporter.py             SentimentFeatureExporter   — word- and sentence-level affect export
```

**`FeatureExtractor`** uses a plug-in registry, so you can register arbitrary sentence-level feature functions and they'll be included in the output CSV alongside valence and arousal:

```python
from affect_analyzer.features.extractor import FeatureExtractor

fe = FeatureExtractor(duration_col="duration", input_col="sentence")
fe.register_feature("word_count", lambda s: len(s.split()))
fe.register_feature("question", lambda s: int(s.strip().endswith("?")))
result = fe.extract(df)
```

---

## The Model & Affect Theory

**Russell's circumplex model of affect** (1980) proposes that all emotional states can be described by two independent dimensions:

- **Valence** — the degree of pleasantness or unpleasantness (−1 = maximally unpleasant, +1 = maximally pleasant)
- **Arousal** — the degree of activation or deactivation (−1 = maximally calm, +1 = maximally activated)

The bundled model (`models/XLM-RoBERTa-base-MSE/`) is an XLM-RoBERTa-base checkpoint fine-tuned with MSE loss to directly regress (valence, arousal) from text. Outputs are clamped to [−1, 1] on both axes via `Hardtanh`. The multilingual backbone supports English, German, French, Spanish, and other languages covered by XLM-RoBERTa's pretraining.

The model also exposes sentence embeddings (CLS token) which are used internally for **salience scoring** — filtering to the most semantically central sentences before topic modeling, controlled by `--salience_top_percent`.

See `papers/Acircumplexmodelofaffect.pdf` for the theoretical foundation.

---

## CLI Reference

```
python conversation_affect_analyzer.py [OPTIONS]

Options:
  -i, --input PATH               Path to transcript CSV  [required]
  -l, --language TEXT            spaCy language code (default: en)
                                 Supported: en, de, fr, es
  -m, --model_dir PATH           Path to model weights or HuggingFace model ID
                                 (default: models/XLM-RoBERTa-base-MSE)
      --device TEXT              Inference device: cpu or cuda  (default: cpu)
      --chunksize INT            CSV read chunk size  (default: 10000)
      --topic_method CHOICE      Topic strategy: global | sliding | js
                                   global   — fit LDA on all sentences
                                   sliding  — sliding-window topic drift
                                   js       — Jensen-Shannon divergence from global unigram
      --num_topics INT           Number of LDA topics  (default: 5)
      --max_features INT         Max vocabulary size for LDA  (default: 1000)
      --topic_keywords_top_n INT Top-N keywords per topic label  (default: 3)
      --salience_top_percent FLOAT
                                 Pre-filter to the top X% most salient sentences
      --window_size INT          Sentences per window (sliding method)  (default: 10)
      --window_step INT          Step size in sentences (sliding method)  (default: 5)
      --plot_circumplex          Show interactive Russell circumplex plot
```

---

## Project Status & Limitations

LexiPlex is an **alpha research prototype**. It is functional and usable, but carries the following caveats:

- **No test suite.** There are no automated tests; results should be validated manually.
- **APIs may change.** Class interfaces and CLI flags are not yet stable.
- **Model weights in git.** The XLM-RoBERTa checkpoint is checked into the repository, which is not ideal for distribution. A future release will separate model storage.
- **Transcripts only.** LexiPlex operates on text. It assumes ASR (automatic speech recognition) has already been run if your source is audio.
- **Sliding-window and JS-divergence topic methods** are less tested than global LDA and may behave unexpectedly on short transcripts.
- **Language support** depends on spaCy model availability. Only `en`, `de`, `fr`, and `es` are explicitly mapped; other languages fall back to `{lang}_core_news_sm`.

Issues and pull requests are welcome. Planned directions include a proper test suite, a pip-installable package, and an integrated audio-to-transcript preprocessing step.
