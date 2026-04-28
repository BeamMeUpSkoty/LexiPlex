"""LexiPlex guided walkthrough demonstrator."""
from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import spacy
import streamlit as st

from affect_analyzer import AnalyzerRegistry, LexiPlexPipeline
from affect_analyzer.analyzers.affect import AffectAnalyzer
from affect_analyzer.analyzers.clinical import ClinicalMarkerAnalyzer
from affect_analyzer.analyzers.complexity import ComplexityAnalyzer
from affect_analyzer.analyzers.dynamics import DynamicsAnalyzer
from affect_analyzer.modelling.valence_arousal import ValenceArousalModel

DEFAULT_CSV = Path("data/mock_therapy_session.csv")
SPEAKER_COLORS = {"Client": "#e07070", "Therapist": "#7ec97e"}
DEFAULT_COLOR = "#7e9ec9"

st.set_page_config(page_title="LexiPlex", layout="wide", page_icon="🧠")


# ── Resource loading ──────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading language models…")
def _load_pipeline() -> LexiPlexPipeline:
    try:
        model = ValenceArousalModel()
    except Exception:
        model = MagicMock()
        model.embed_sentences.side_effect = lambda s, **kw: np.zeros((len(s), 768))
        model.batch_score.side_effect = lambda s, **kw: np.random.uniform(-1, 1, (len(s), 2))

    nlp = spacy.load("en_core_web_sm")
    registry = AnalyzerRegistry()
    registry.register(AffectAnalyzer(model=model))
    registry.register(ComplexityAnalyzer(nlp=nlp))
    registry.register(ClinicalMarkerAnalyzer())
    registry.register(DynamicsAnalyzer())
    return LexiPlexPipeline(registry=registry, model=model, language="en")


@st.cache_data(show_spinner="Analysing transcript…")
def _run_analysis(csv_path: str) -> dict:
    pipeline = _load_pipeline()
    results = pipeline.run(csv_path)
    return {
        name: {
            "per_sentence": r.per_sentence,
            "global_metrics": r.global_metrics,
            "metadata": {
                k: v for k, v in r.metadata.items()
                if not isinstance(v, pd.DataFrame)
            },
            "metadata_dfs": {
                k: v for k, v in r.metadata.items()
                if isinstance(v, pd.DataFrame)
            },
        }
        for name, r in results.items()
    }


# ── Sidebar ───────────────────────────────────────────────────────────────────

def _sidebar() -> tuple[str, str]:
    with st.sidebar:
        st.title("LexiPlex")
        st.markdown("*Language analysis for health contexts*")
        st.divider()
        uploaded = st.file_uploader("Upload transcript CSV", type=["csv"])
        if uploaded:
            tmp = Path("data/_uploaded.csv")
            tmp.write_bytes(uploaded.read())
            csv_path = str(tmp)
        else:
            csv_path = str(DEFAULT_CSV)
            st.caption(f"Using: `{DEFAULT_CSV.name}`")
        st.divider()
        chapter = st.radio(
            "Chapter",
            ["① Introduction", "② Affect", "③ Complexity", "④ Clinical Markers", "⑤ Dynamics"],
        )
        with st.expander("Advanced"):
            st.caption("Model and language parameters — configurable in a future version.")
    return csv_path, chapter


# ── Chapter 1 ─────────────────────────────────────────────────────────────────

def _chapter_intro(csv_path: str) -> None:
    st.header("What is affect in language?")
    st.markdown(
        "Every sentence we speak carries an emotional charge. Russell's **circumplex model of affect** "
        "(1980) proposes that all emotions can be described by two independent dimensions:\n\n"
        "- **Valence** — pleasant ↔ unpleasant (−1 to +1)\n"
        "- **Arousal** — activated ↔ calm (−1 to +1)\n\n"
        "Mapping language onto this circle lets us track the emotional trajectory of a conversation "
        "in a way that is rigorous, interpretable, and speaker-resolved."
    )
    col1, col2 = st.columns([1, 1])
    with col1:
        st.plotly_chart(_make_circumplex_explainer(), use_container_width=True)
    with col2:
        st.markdown("### The mock therapy session")
        df = pd.read_csv(csv_path)
        st.dataframe(df.head(10), use_container_width=True)
        n_speakers = df["speaker"].nunique() if "speaker" in df.columns else "unknown"
        st.caption(f"{len(df)} utterances · {n_speakers} speakers")
    st.info(
        "Navigate to **② Affect** in the sidebar to see the temporal playback — "
        "each utterance plotted on the circumplex in real time."
    )


def _make_circumplex_explainer() -> go.Figure:
    theta = np.linspace(0, 2 * np.pi, 120)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.cos(theta), y=np.sin(theta), mode="lines",
        line=dict(color="#555", width=1), showlegend=False, hoverinfo="skip",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="#444")
    fig.add_vline(x=0, line_dash="dash", line_color="#444")
    for x, y, text in [
        (-0.75, -0.72, "Tense / Anxious"), (0.7, -0.72, "Excited / Elated"),
        (-0.75, 0.72, "Sad / Depressed"), (0.7, 0.72, "Calm / Relaxed"),
    ]:
        fig.add_annotation(x=x, y=y, text=text, showarrow=False,
                           font=dict(color="#888", size=11))
    for x, y, text in [(0, 1.18, "↑ High Arousal"), (0, -1.18, "Low Arousal ↓"),
                        (-1.18, 0, "−"), (1.18, 0, "+")]:
        fig.add_annotation(x=x, y=y, text=text, showarrow=False,
                           font=dict(color="#666", size=10))
    fig.update_layout(
        xaxis=dict(range=[-1.3, 1.3], showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(range=[-1.3, 1.3], showticklabels=False, showgrid=False,
                   zeroline=False, scaleanchor="x"),
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        margin=dict(l=10, r=10, t=10, b=10), height=300, showlegend=False,
    )
    return fig


# ── Stubs for chapters 2–5 (filled in Tasks 12–13) ───────────────────────────

def _chapter_affect(results: dict, utterances: pd.DataFrame) -> None:
    st.info("Affect chapter — implemented in Task 12.")


def _chapter_complexity(results: dict) -> None:
    st.info("Complexity chapter — implemented in Task 13.")


def _chapter_clinical(results: dict) -> None:
    st.info("Clinical markers chapter — implemented in Task 13.")


def _chapter_dynamics(results: dict) -> None:
    st.info("Dynamics chapter — implemented in Task 13.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    csv_path, chapter = _sidebar()
    if chapter == "① Introduction":
        _chapter_intro(csv_path)
    else:
        with st.spinner("Running analysis…"):
            results = _run_analysis(csv_path)
        utterances = pd.read_csv(csv_path)
        if chapter == "② Affect":
            _chapter_affect(results, utterances)
        elif chapter == "③ Complexity":
            _chapter_complexity(results)
        elif chapter == "④ Clinical Markers":
            _chapter_clinical(results)
        elif chapter == "⑤ Dynamics":
            _chapter_dynamics(results)


if __name__ == "__main__":
    main()
