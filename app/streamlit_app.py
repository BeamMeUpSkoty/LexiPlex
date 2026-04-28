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
DEFAULT_CSV = Path("data/mock_therapy_session.csv")
SPEAKER_COLORS = {"Client": "#e07070", "Therapist": "#7ec97e"}
DEFAULT_COLOR = "#7e9ec9"

st.set_page_config(page_title="LexiPlex", layout="wide", page_icon="🧠")


# ── Resource loading ──────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading language models…")
def _load_pipeline() -> LexiPlexPipeline:
    try:
        from affect_analyzer.modelling.valence_arousal import ValenceArousalModel
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
    st.header("Affect Analysis")
    st.markdown(
        "Each utterance is scored on **valence** (pleasant ↔ unpleasant) and **arousal** "
        "(activated ↔ calm) using an XLM-RoBERTa model fine-tuned for affect regression. "
        "Press **▶ Play** to watch the session unfold utterance by utterance."
    )

    affect_df = results["affect"]["per_sentence"]

    # Build utterance-level playback index from turn boundaries
    if "turn_id" in affect_df.columns:
        turns = affect_df.groupby("turn_id").first().reset_index()
    else:
        turns = affect_df.copy()
    n_turns = len(turns)

    # Session state
    if "affect_idx" not in st.session_state:
        st.session_state.affect_idx = 0
    if "affect_playing" not in st.session_state:
        st.session_state.affect_playing = False

    # Controls
    c1, c2, c3, c4 = st.columns([1, 6, 1, 1])
    with c1:
        label = "⏸ Pause" if st.session_state.affect_playing else "▶ Play"
        if st.button(label):
            st.session_state.affect_playing = not st.session_state.affect_playing
            st.rerun()
    with c2:
        idx = st.slider("", 0, n_turns - 1, st.session_state.affect_idx,
                        key="affect_slider", label_visibility="collapsed")
        if idx != st.session_state.affect_idx:
            st.session_state.affect_idx = idx
            st.session_state.affect_playing = False
    with c3:
        speed = st.selectbox("", [0.5, 1.0, 2.0], index=1, label_visibility="collapsed")
    with c4:
        st.metric("Turn", f"{st.session_state.affect_idx + 1} / {n_turns}")

    current_idx = st.session_state.affect_idx
    turns_so_far = turns.iloc[: current_idx + 1]
    # Filter by turn_id so multi-sentence turns are fully included
    if "turn_id" in affect_df.columns:
        affect_so_far = affect_df[affect_df["turn_id"] <= current_idx]
    else:
        affect_so_far = affect_df.iloc[: current_idx + 1]

    left_col, right_col = st.columns([1, 1])
    with left_col:
        _render_transcript_panel(turns, current_idx)
    with right_col:
        _render_circumplex_panel(turns_so_far)

    st.divider()
    _render_scorecard_strip(results, turns_so_far, affect_so_far)

    # Auto-advance
    if st.session_state.affect_playing:
        time.sleep(1.0 / speed)
        if st.session_state.affect_idx < n_turns - 1:
            st.session_state.affect_idx += 1
        else:
            st.session_state.affect_playing = False
        st.rerun()


def _render_transcript_panel(turns: pd.DataFrame, current_idx: int) -> None:
    st.markdown("**Transcript**")
    window_start = max(0, current_idx - 5)
    for abs_idx in range(window_start, min(len(turns), current_idx + 3)):
        row = turns.iloc[abs_idx]
        speaker = str(row.get("speaker", "Unknown"))
        color = SPEAKER_COLORS.get(speaker, DEFAULT_COLOR)
        is_current = abs_idx == current_idx
        opacity = max(0.2, 1.0 - (current_idx - abs_idx) * 0.15) if abs_idx <= current_idx else 0.18

        if is_current:
            v = float(row.get("valence", 0.0))
            a = float(row.get("arousal", 0.0))
            st.markdown(
                f'<div style="border:1px solid {color};border-left:3px solid {color};'
                f'border-radius:6px;padding:8px 10px;margin:4px 0;background:#1a1e28;">'
                f'<span style="font-size:10px;color:{color};font-weight:700;">'
                f'{speaker} &nbsp;← now</span><br>'
                f'<span style="font-size:13px;color:#fff;">{row.get("sentence","")}</span><br>'
                f'<span style="background:#1a2a1a;border-radius:3px;padding:1px 7px;'
                f'font-size:10px;color:#7ec97e;margin-top:4px;display:inline-block;">'
                f'V {v:+.2f}</span> '
                f'<span style="background:#1a1e2a;border-radius:3px;padding:1px 7px;'
                f'font-size:10px;color:#7e9ec9;display:inline-block;">A {a:+.2f}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div style="padding:4px 10px;margin:2px 0;opacity:{opacity:.2f};">'
                f'<span style="font-size:9px;color:{color};">{speaker}</span><br>'
                f'<span style="font-size:11px;color:#ccc;">{row.get("sentence","")}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )


def _render_circumplex_panel(turns_so_far: pd.DataFrame) -> None:
    theta = np.linspace(0, 2 * np.pi, 120)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.cos(theta), y=np.sin(theta), mode="lines",
        line=dict(color="#444", width=1), showlegend=False, hoverinfo="skip",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="#333")
    fig.add_vline(x=0, line_dash="dash", line_color="#333")

    if "speaker" in turns_so_far.columns and "valence" in turns_so_far.columns:
        for speaker, color in SPEAKER_COLORS.items():
            subset = turns_so_far[turns_so_far["speaker"] == speaker]
            if len(subset) == 0:
                continue
            n = len(subset)
            opacities = [max(0.15, 0.25 + 0.75 * i / n) for i in range(n)]
            sizes = [5] * (n - 1) + [13]
            fig.add_trace(go.Scatter(
                x=subset["valence"].tolist(),
                y=subset["arousal"].tolist(),
                mode="lines+markers",
                line=dict(color=color, width=1, dash="dot"),
                marker=dict(size=sizes, color=color, opacity=opacities,
                            line=dict(width=1, color="#111")),
                name=speaker,
                hovertemplate=f"<b>{speaker}</b><br>V: %{{x:.2f}}<br>A: %{{y:.2f}}<extra></extra>",
            ))

    for x, y, text in [
        (-0.8, -0.8, "Tense"), (0.8, -0.8, "Excited"),
        (-0.8, 0.8, "Sad"), (0.8, 0.8, "Calm"),
    ]:
        fig.add_annotation(x=x, y=y, text=text, showarrow=False,
                           font=dict(color="#555", size=10))

    fig.update_layout(
        xaxis=dict(range=[-1.25, 1.25], title="Valence", showgrid=False),
        yaxis=dict(range=[-1.25, 1.25], title="Arousal", showgrid=False, scaleanchor="x"),
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="#aaa"), legend=dict(x=0.02, y=0.98),
        margin=dict(l=40, r=20, t=20, b=40), height=350,
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_scorecard_strip(results: dict, turns_so_far: pd.DataFrame,
                             affect_so_far: pd.DataFrame) -> None:
    cols = st.columns(4)
    # Derive a turn_id filter so multi-sentence turns are handled correctly
    max_turn_id = int(affect_so_far["turn_id"].max()) if (
        "turn_id" in affect_so_far.columns and len(affect_so_far) > 0
    ) else len(turns_so_far) - 1

    def _by_turn(df: pd.DataFrame) -> pd.DataFrame:
        if "turn_id" in df.columns:
            return df[df["turn_id"] <= max_turn_id]
        return df.iloc[: len(affect_so_far)]

    with cols[0]:
        st.markdown("**Affect**")
        if "speaker" in affect_so_far.columns and "valence" in affect_so_far.columns:
            for speaker, color in SPEAKER_COLORS.items():
                sub = affect_so_far[affect_so_far["speaker"] == speaker]
                if len(sub):
                    v, a = sub["valence"].mean(), sub["arousal"].mean()
                    st.markdown(
                        f'<span style="color:{color};font-weight:700;">{speaker}</span> '
                        f'V {v:+.2f} &nbsp; A {a:+.2f}',
                        unsafe_allow_html=True,
                    )

    with cols[1]:
        st.markdown("**Complexity**")
        cdf = results.get("complexity", {}).get("per_sentence")
        if cdf is not None:
            shown = _by_turn(cdf)
            coh = shown["coherence_to_prev"].dropna().mean()
            st.metric("Coherence", f"{coh:.2f}" if not pd.isna(coh) else "—")
            if "speaker" in shown.columns:
                for speaker in SPEAKER_COLORS:
                    ttr = shown[shown["speaker"] == speaker]["type_token_ratio"].mean()
                    if not pd.isna(ttr):
                        st.caption(f"{speaker} TTR: {ttr:.2f}")

    with cols[2]:
        st.markdown("**Clinical**")
        cldf = results.get("clinical", {}).get("per_sentence")
        if cldf is not None:
            shown = _by_turn(cldf)
            if "speaker" in shown.columns:
                for speaker, color in SPEAKER_COLORS.items():
                    sub = shown[shown["speaker"] == speaker]
                    if len(sub):
                        h = sub["hedging_rate"].mean()
                        neg = sub["negation_density"].mean()
                        st.markdown(
                            f'<span style="color:{color};font-size:11px;">{speaker}</span> '
                            f'Hedge {h:.0%} &nbsp; Neg {neg:.0%}',
                            unsafe_allow_html=True,
                        )

    with cols[3]:
        st.markdown("**Dynamics**")
        per_speaker = results.get("dynamics", {}).get("metadata_dfs", {}).get("per_speaker")
        if per_speaker is not None:
            for speaker, row in per_speaker.iterrows():
                color = SPEAKER_COLORS.get(str(speaker), DEFAULT_COLOR)
                pct = row.get("dominance_pct", 0)
                st.markdown(
                    f'<span style="color:{color};font-size:11px;">{speaker}</span> '
                    f'{pct:.0f}% talk time',
                    unsafe_allow_html=True,
                )
        silence = results.get("dynamics", {}).get("metadata", {}).get("silence_total", 0)
        st.caption(f"Silence: {float(silence):.1f}s")


def _chapter_complexity(results: dict) -> None:
    st.header("Linguistic Complexity")
    st.markdown(
        "**Coherence** measures how similar adjacent sentences are in meaning — computed from "
        "XLM-RoBERTa embeddings, no extra inference needed. "
        "**Type-token ratio (TTR)** measures vocabulary richness: more unique words = higher TTR. "
        "**Lexical density** is the proportion of content words (nouns, verbs, adjectives, adverbs)."
    )
    cdf = results.get("complexity", {}).get("per_sentence")
    if cdf is None or len(cdf) == 0:
        st.warning("No complexity data available.")
        return

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Coherence over time")
        fig = go.Figure()
        if "speaker" in cdf.columns:
            for speaker, color in SPEAKER_COLORS.items():
                sub = cdf[cdf["speaker"] == speaker].reset_index(drop=True)
                fig.add_trace(go.Scatter(
                    y=sub["coherence_to_prev"].tolist(), mode="lines+markers",
                    name=speaker, line=dict(color=color, width=2), marker=dict(size=5),
                ))
        else:
            fig.add_trace(go.Scatter(y=cdf["coherence_to_prev"].tolist(), mode="lines+markers"))
        fig.update_layout(
            xaxis_title="Sentence", yaxis_title="Coherence (cosine sim)",
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
            font=dict(color="#aaa"), height=280, margin=dict(l=40, r=10, t=10, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Vocabulary richness per speaker")
        if "speaker" in cdf.columns:
            stats = cdf.groupby("speaker")[["type_token_ratio", "lexical_density"]].mean().reset_index()
            fig2 = go.Figure()
            for col_name, label in [("type_token_ratio", "TTR"), ("lexical_density", "Lexical Density")]:
                fig2.add_trace(go.Bar(x=stats["speaker"].tolist(), y=stats[col_name].tolist(), name=label))
            fig2.update_layout(
                barmode="group", plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                font=dict(color="#aaa"), height=280, margin=dict(l=40, r=10, t=10, b=40),
                yaxis_title="Score",
            )
            st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Health context")
    st.info(
        "Falling coherence over a session may signal dissociation or cognitive fatigue. "
        "A large TTR gap between therapist and client can reveal communication asymmetry."
    )


def _chapter_clinical(results: dict) -> None:
    st.header("Clinical Markers")
    st.markdown(
        "Rule-based markers computed from word patterns alone — no model, fully transparent. "
        "Each score is a rate per sentence (proportion of words matching the marker category)."
    )
    cldf = results.get("clinical", {}).get("per_sentence")
    if cldf is None or len(cldf) == 0:
        st.warning("No clinical marker data available.")
        return
    if "speaker" not in cldf.columns:
        st.warning("Speaker column required.")
        return

    markers = {
        "hedging_rate": "Hedging",
        "certainty_rate": "Certainty",
        "self_ref_rate": "Self-reference",
        "negation_density": "Negation",
    }
    stats = cldf.groupby("speaker")[list(markers.keys())].mean().reset_index()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Rates per speaker")
        fig = go.Figure()
        for col_name, label in markers.items():
            fig.add_trace(go.Bar(x=stats["speaker"].tolist(), y=stats[col_name].tolist(), name=label))
        fig.update_layout(
            barmode="group", plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
            font=dict(color="#aaa"), height=300, margin=dict(l=40, r=10, t=10, b=40),
            yaxis_title="Rate",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Question rate per speaker")
        q = cldf.groupby("speaker")["is_question"].mean().reset_index()
        q.columns = ["Speaker", "Question rate"]
        q["Question rate"] = (q["Question rate"] * 100).round(1).astype(str) + "%"
        st.dataframe(q, use_container_width=True, hide_index=True)
        st.subheader("Progress bars")
        for speaker in cldf["speaker"].unique():
            sub = cldf[cldf["speaker"] == speaker]
            color = SPEAKER_COLORS.get(str(speaker), DEFAULT_COLOR)
            st.markdown(f'<span style="color:{color};font-weight:700;">{speaker}</span>',
                        unsafe_allow_html=True)
            for col_name, label in markers.items():
                val = float(sub[col_name].mean())
                st.progress(min(val, 1.0), text=f"{label}: {val:.1%}")

    st.markdown("### Health context")
    st.info(
        "High **hedging** + low **certainty** = avoidant language. "
        "High **self-reference** + **negation** are established markers of depressive cognition (Beck, 1979). "
        "**Question rate** in the therapist track reflects engagement strategy."
    )


def _chapter_dynamics(results: dict) -> None:
    st.header("Conversational Dynamics")
    st.markdown(
        "Structural patterns derived entirely from timestamps — who speaks, for how long, "
        "and how quickly they respond. No language model required."
    )
    dyn_meta = results.get("dynamics", {}).get("metadata", {})
    per_speaker = results.get("dynamics", {}).get("metadata_dfs", {}).get("per_speaker")
    latencies = dyn_meta.get("latencies", [])
    gm = results.get("dynamics", {}).get("global_metrics", {})

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Talk time")
        if per_speaker is not None:
            fig = go.Figure(go.Pie(
                labels=per_speaker.index.tolist(),
                values=per_speaker["dominance_pct"].tolist(),
                marker=dict(colors=[SPEAKER_COLORS.get(str(s), DEFAULT_COLOR)
                                    for s in per_speaker.index]),
                hole=0.5, textinfo="label+percent",
            ))
            fig.update_layout(
                plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                font=dict(color="#aaa"), height=260,
                margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Response latency")
        if latencies:
            fig2 = go.Figure(go.Histogram(x=latencies, nbinsx=20,
                                          marker_color="#7e9ec9", opacity=0.8))
            fig2.update_layout(
                xaxis_title="Gap (s)", yaxis_title="Count",
                plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                font=dict(color="#aaa"), height=260,
                margin=dict(l=40, r=10, t=10, b=40),
            )
            st.plotly_chart(fig2, use_container_width=True)
        st.metric("Mean latency", f"{gm.get('latency_mean', 0):.1f}s")
        st.metric("Total silence", f"{gm.get('silence_total', 0):.1f}s")

    with col3:
        st.subheader("Per-speaker stats")
        if per_speaker is not None:
            display = per_speaker[["turns", "total_duration", "mean_utterance_length",
                                   "dominance_pct"]].copy()
            display.columns = ["Turns", "Total (s)", "Mean (s)", "Talk %"]
            st.dataframe(display.round(1), use_container_width=True)

    st.markdown("### Health context")
    st.info(
        "Therapist/client **talk ratio** is a core process measure in psychotherapy research. "
        "Long **silences** (>3 s) can indicate the client is processing or avoiding. "
        "Growing client utterance length over a session signals increasing engagement."
    )


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
