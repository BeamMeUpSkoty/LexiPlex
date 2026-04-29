"""LexiPlex guided walkthrough demonstrator."""
from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import spacy
import streamlit as st
import streamlit.components.v1 as components

from affect_analyzer import AnalyzerRegistry, LexiPlexPipeline
from affect_analyzer.analyzers.affect import AffectAnalyzer
from affect_analyzer.analyzers.clinical import ClinicalMarkerAnalyzer
from affect_analyzer.analyzers.complexity import ComplexityAnalyzer
from affect_analyzer.analyzers.dynamics import DynamicsAnalyzer
DEFAULT_CSV = Path("data/mock_therapy_session.csv")
SPEAKER_COLORS = {"Client": "#e07070", "Therapist": "#7ec97e"}
DEFAULT_COLOR = "#7e9ec9"
REQUIRED_COLUMNS = {"start", "end", "text"}

st.set_page_config(page_title="LexiPlex", layout="wide", page_icon="🧠")


def _inject_theme() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2rem;
            max-width: 1180px;
        }
        [data-testid="stSidebar"] {
            border-right: 1px solid rgba(255,255,255,0.08);
        }
        .lp-eyebrow {
            color: #93a4b8;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0;
            text-transform: uppercase;
        }
        .lp-title {
            font-size: 2.1rem;
            line-height: 1.08;
            font-weight: 760;
            margin: 0.1rem 0 0.5rem;
        }
        .lp-muted {
            color: #aab4c0;
            font-size: 0.96rem;
            max-width: 820px;
        }
        .lp-card {
            border: 1px solid rgba(255,255,255,0.10);
            border-radius: 8px;
            padding: 0.85rem 0.95rem;
            background: rgba(255,255,255,0.035);
            min-height: 104px;
        }
        .lp-card-label {
            color: #94a3b8;
            font-size: 0.75rem;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }
        .lp-card-value {
            color: #f4f7fb;
            font-size: 1.45rem;
            line-height: 1.1;
            font-weight: 760;
        }
        .lp-card-note {
            color: #aab4c0;
            font-size: 0.8rem;
            margin-top: 0.35rem;
        }
        .lp-chip {
            display: inline-block;
            border: 1px solid rgba(255,255,255,0.12);
            border-radius: 999px;
            padding: 0.18rem 0.55rem;
            margin: 0.12rem 0.16rem 0.12rem 0;
            color: #d7dee8;
            background: rgba(255,255,255,0.04);
            font-size: 0.78rem;
        }
        div[data-testid="stMetric"] {
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 8px;
            padding: 0.65rem 0.75rem;
            background: rgba(255,255,255,0.03);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


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


@st.cache_data(show_spinner=False)
def _load_transcript(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def _validate_transcript(df: pd.DataFrame) -> list[str]:
    issues = []
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        issues.append(f"Missing required column(s): {', '.join(missing)}")
    if "speaker" not in df.columns:
        issues.append("No speaker column found, so speaker comparisons will be unavailable.")
    if {"start", "end"}.issubset(df.columns) and (df["end"] < df["start"]).any():
        issues.append("At least one row has an end time before its start time.")
    return issues


def _write_uploaded_file(uploaded) -> str:
    content = uploaded.getvalue()
    digest = hashlib.sha1(content).hexdigest()[:12]
    tmp = Path("data") / f"_uploaded_{digest}.csv"
    if not tmp.exists():
        tmp.write_bytes(content)
    return str(tmp)


def _build_feature_table(results: dict) -> pd.DataFrame:
    base = results.get("affect", {}).get("per_sentence")
    if base is None:
        return pd.DataFrame()

    out = base.copy()
    preferred = {
        "complexity": ["word_count", "type_token_ratio", "lexical_density", "coherence_to_prev"],
        "clinical": [
            "hedging_rate", "certainty_rate", "self_ref_rate",
            "negation_density", "is_question",
        ],
        "dynamics": ["turn_id", "is_turn_start"],
    }
    for analyzer, columns in preferred.items():
        df = results.get(analyzer, {}).get("per_sentence")
        if df is None:
            continue
        for col in columns:
            if col in df.columns and col not in out.columns:
                out[col] = df[col].values
    return out


def _metric_card(label: str, value: str, note: str = "") -> None:
    st.markdown(
        f"""
        <div class="lp-card">
          <div class="lp-card-label">{html.escape(label)}</div>
          <div class="lp-card-value">{html.escape(value)}</div>
          <div class="lp-card-note">{html.escape(note)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _plot_theme(fig: go.Figure, height: int = 300) -> go.Figure:
    fig.update_layout(
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font=dict(color="#b8c2cf"),
        margin=dict(l=40, r=16, t=24, b=40),
        height=height,
    )
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────

def _sidebar() -> tuple[str, str]:
    with st.sidebar:
        st.title("LexiPlex")
        st.markdown("*Language analysis for health contexts*")
        st.divider()
        uploaded = st.file_uploader("Upload transcript CSV", type=["csv"])
        if uploaded:
            csv_path = _write_uploaded_file(uploaded)
            st.caption(f"Uploaded: `{uploaded.name}`")
        else:
            csv_path = str(DEFAULT_CSV)
            st.caption(f"Using: `{DEFAULT_CSV.name}`")
        st.divider()
        chapter = st.radio(
            "View",
            [
                "① Introduction",
                "② Overview",
                "③ Affect",
                "④ Complexity",
                "⑤ Clinical Markers",
                "⑥ Dynamics",
            ],
        )
        st.divider()
        st.caption("Expected columns: start, end, text. Speaker is optional but recommended.")
        with st.expander("Advanced"):
            st.caption("Model and language parameters — configurable in a future version.")
    return csv_path, chapter


# ── Chapter 1 ─────────────────────────────────────────────────────────────────

def _chapter_intro(csv_path: str) -> None:
    st.markdown('<div class="lp-eyebrow">Guided Walkthrough</div>', unsafe_allow_html=True)
    st.markdown('<div class="lp-title">What is affect in language?</div>', unsafe_allow_html=True)
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
        df = _load_transcript(csv_path)
        st.dataframe(df.head(10), use_container_width=True)
        n_speakers = df["speaker"].nunique() if "speaker" in df.columns else "unknown"
        st.caption(f"{len(df)} utterances · {n_speakers} speakers")
    st.info(
        "Navigate to **② Overview** for the session summary or **③ Affect** for temporal playback — "
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


def _chapter_overview(results: dict, utterances: pd.DataFrame) -> None:
    st.markdown('<div class="lp-eyebrow">Session Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="lp-title">Transcript at a Glance</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="lp-muted">A compact readout of affect, language complexity, '
        'clinical markers, and conversational balance for the selected transcript.</div>',
        unsafe_allow_html=True,
    )

    affect_df = results.get("affect", {}).get("per_sentence", pd.DataFrame())
    dyn_metrics = results.get("dynamics", {}).get("global_metrics", {})
    complexity_metrics = results.get("complexity", {}).get("global_metrics", {})
    speakers = sorted(utterances["speaker"].dropna().unique()) if "speaker" in utterances.columns else []

    total_time = 0.0
    if {"start", "end"}.issubset(utterances.columns):
        total_time = float(utterances["end"].max() - utterances["start"].min())

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        _metric_card("Utterances", f"{len(utterances):,}", f"{len(speakers)} speaker(s)")
    with m2:
        _metric_card("Duration", f"{total_time:.0f}s", "Transcript time span")
    with m3:
        valence = affect_df["valence"].mean() if "valence" in affect_df.columns else np.nan
        _metric_card("Mean Valence", f"{valence:+.2f}" if not pd.isna(valence) else "-", "Pleasantness")
    with m4:
        silence = dyn_metrics.get("silence_total", 0.0)
        _metric_card("Total Silence", f"{float(silence):.1f}s", f"{dyn_metrics.get('turn_count', 0):.0f} turns")

    if speakers:
        chips = "".join(
            f'<span class="lp-chip" style="border-color:{SPEAKER_COLORS.get(str(s), DEFAULT_COLOR)};">'
            f'{html.escape(str(s))}</span>'
            for s in speakers
        )
        st.markdown(chips, unsafe_allow_html=True)

    left, right = st.columns([1.25, 1])
    with left:
        st.subheader("Affect trajectory")
        if {"valence", "arousal"}.issubset(affect_df.columns):
            fig = go.Figure()
            x_values = list(range(1, len(affect_df) + 1))
            for metric, color in [("valence", "#e07070"), ("arousal", "#7e9ec9")]:
                fig.add_trace(go.Scatter(
                    x=x_values,
                    y=affect_df[metric],
                    mode="lines+markers",
                    name=metric.title(),
                    line=dict(color=color, width=2),
                    marker=dict(size=5),
                ))
            fig.add_hline(y=0, line_dash="dash", line_color="#596273")
            fig.update_yaxes(range=[-1, 1], title="Score")
            fig.update_xaxes(title="Sentence")
            st.plotly_chart(_plot_theme(fig, 320), use_container_width=True)

    with right:
        st.subheader("Speaker balance")
        per_speaker = results.get("dynamics", {}).get("metadata_dfs", {}).get("per_speaker")
        if per_speaker is not None and len(per_speaker):
            fig = go.Figure(go.Bar(
                x=per_speaker.index.tolist(),
                y=per_speaker["dominance_pct"].tolist(),
                marker_color=[SPEAKER_COLORS.get(str(s), DEFAULT_COLOR) for s in per_speaker.index],
                text=[f"{v:.0f}%" for v in per_speaker["dominance_pct"]],
                textposition="outside",
            ))
            fig.update_yaxes(range=[0, 100], title="Talk time")
            st.plotly_chart(_plot_theme(fig, 320), use_container_width=True)
        st.metric("Coherence", f"{complexity_metrics.get('coherence_mean', 0):.2f}")
        st.metric("Mean latency", f"{dyn_metrics.get('latency_mean', 0):.1f}s")

    st.subheader("Analyzed feature table")
    features = _build_feature_table(results)
    if len(features):
        visible_cols = [
            col for col in [
                "start", "end", "speaker", "sentence", "valence", "arousal",
                "word_count", "coherence_to_prev", "hedging_rate",
                "self_ref_rate", "negation_density", "is_question",
            ]
            if col in features.columns
        ]
        st.dataframe(features[visible_cols], use_container_width=True, hide_index=True)
        st.download_button(
            "Download analyzed CSV",
            data=features.to_csv(index=False).encode("utf-8"),
            file_name="lexiplex_analyzed_features.csv",
            mime="text/csv",
        )


# ── Analysis views ────────────────────────────────────────────────────────────

def _chapter_affect(results: dict, utterances: pd.DataFrame) -> None:
    st.header("Affect Analysis")
    st.markdown(
        "Each utterance is scored on **valence** (pleasant ↔ unpleasant) and **arousal** "
        "(activated ↔ deactivation) using an XLM-RoBERTa model fine-tuned for affect. "
        "Press **▶** to watch the session unfold."
    )

    affect_df = results["affect"]["per_sentence"]

    # Build utterance-level playback index from turn boundaries
    if "turn_id" in affect_df.columns:
        turns = affect_df.groupby("turn_id").first().reset_index()
    else:
        turns = affect_df.copy()
    n_turns = len(turns)
    if n_turns == 0:
        st.warning("No affect rows available for playback.")
        return

    _render_smooth_affect_player(turns)

    st.divider()
    _render_scorecard_strip(results, turns, affect_df)


def _render_smooth_affect_player(turns: pd.DataFrame) -> None:
    rows = []
    for _, row in turns.iterrows():
        speaker = str(row.get("speaker", "Unknown"))
        rows.append({
            "speaker": speaker,
            "sentence": str(row.get("sentence", "")),
            "valence": float(row.get("valence", 0.0)),
            "arousal": float(row.get("arousal", 0.0)),
            "color": SPEAKER_COLORS.get(speaker, DEFAULT_COLOR),
        })

    payload = json.dumps(rows).replace("</", "<\\/")
    component_html = f"""
    <div class="lp-player">
      <div class="lp-controls" aria-label="Affect playback controls">
        <button id="back" title="Back one turn">⏮</button>
        <button id="play" title="Play or pause">▶</button>
        <button id="next" title="Forward one turn">⏭</button>
        <input id="scrub" type="range" min="0" max="{len(rows) - 1}" value="0" step="1" />
        <select id="speed" aria-label="Playback speed">
          <option value="0.65">0.65x</option>
          <option value="1" selected>1x</option>
          <option value="1.5">1.5x</option>
          <option value="2.25">2.25x</option>
        </select>
        <div id="turn">Turn 1 / {len(rows)}</div>
      </div>
      <div class="lp-grid">
        <section class="lp-panel transcript">
          <div class="panel-title">Transcript</div>
          <div id="utterances"></div>
        </section>
        <section class="lp-panel viz">
          <canvas id="circumplex" width="560" height="420"></canvas>
        </section>
      </div>
      <div class="lp-live">
        <div><span>Speaker</span><strong id="speaker">-</strong></div>
        <div><span>Valence</span><strong id="valence">-</strong></div>
        <div><span>Arousal</span><strong id="arousal">-</strong></div>
        <div><span>Running Mean</span><strong id="mean">-</strong></div>
      </div>
    </div>
    <script>
    const data = {payload};
    const canvas = document.getElementById("circumplex");
    const ctx = canvas.getContext("2d");
    const playBtn = document.getElementById("play");
    const backBtn = document.getElementById("back");
    const nextBtn = document.getElementById("next");
    const scrub = document.getElementById("scrub");
    const speed = document.getElementById("speed");
    const turn = document.getElementById("turn");
    const utterances = document.getElementById("utterances");
    const speakerEl = document.getElementById("speaker");
    const valenceEl = document.getElementById("valence");
    const arousalEl = document.getElementById("arousal");
    const meanEl = document.getElementById("mean");
    const speakerNames = [...new Set(data.map((row) => row.speaker))];
    let idx = 0;
    let progress = 0;
    let playing = false;
    let lastTs = null;
    const baseMs = 1150;

    function ease(t) {{
      return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
    }}
    function clamp(value, min, max) {{
      return Math.min(max, Math.max(min, value));
    }}
    function point(v, a) {{
      const pad = 56;
      const size = Math.min(canvas.width, canvas.height) - pad * 2;
      const cx = canvas.width / 2;
      const cy = canvas.height / 2;
      const radius = size / 2;
      return [cx + clamp(v, -1, 1) * radius, cy - clamp(a, -1, 1) * radius];
    }}
    function drawAxes() {{
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = "#0e1117";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      const [cx, cy] = point(0, 0);
      const [rx] = point(1, 0);
      const r = rx - cx;
      ctx.strokeStyle = "#46505e";
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.stroke();
      ctx.setLineDash([5, 6]);
      ctx.beginPath();
      ctx.moveTo(cx - r - 12, cy);
      ctx.lineTo(cx + r + 12, cy);
      ctx.moveTo(cx, cy - r - 12);
      ctx.lineTo(cx, cy + r + 12);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = "#7b8797";
      ctx.font = "12px system-ui, -apple-system, Segoe UI, sans-serif";
      ctx.fillText("Tense", cx - r + 10, cy - r + 22);
      ctx.fillText("Excited", cx + r - 48, cy - r + 22);
      ctx.fillText("Sad", cx - r + 10, cy + r - 12);
      ctx.fillText("Calm", cx + r - 62, cy + r - 12);
      ctx.fillText("Valence", cx + r + 20, cy + 22);
      ctx.save();
      ctx.translate(cx - 30, cy - r - 20);
      ctx.fillText("Arousal", 0, 0);
      ctx.restore();
    }}
    function drawPath() {{
      const grouped = new Map();
      data.slice(0, idx + 1).forEach((row) => {{
        if (!grouped.has(row.speaker)) grouped.set(row.speaker, []);
        grouped.get(row.speaker).push(row);
      }});
      grouped.forEach((items) => {{
        if (!items.length) return;
        ctx.strokeStyle = items[0].color;
        ctx.lineWidth = 2;
        ctx.globalAlpha = 0.7;
        ctx.beginPath();
        items.forEach((row, i) => {{
          const [x, y] = point(row.valence, row.arousal);
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }});
        ctx.stroke();
        ctx.globalAlpha = 1;
        items.forEach((row, i) => {{
          const [x, y] = point(row.valence, row.arousal);
          ctx.fillStyle = row.color;
          ctx.globalAlpha = 0.25 + 0.55 * (i + 1) / items.length;
          ctx.beginPath();
          ctx.arc(x, y, 4, 0, Math.PI * 2);
          ctx.fill();
        }});
        ctx.globalAlpha = 1;
      }});
    }}
    function latestForSpeaker(speaker, maxIndex) {{
      for (let i = Math.min(maxIndex, data.length - 1); i >= 0; i--) {{
        if (data[i].speaker === speaker) return data[i];
      }}
      return null;
    }}
    function firstForSpeaker(speaker) {{
      return data.find((row) => row.speaker === speaker) || null;
    }}
    function drawSpeakerMarkers() {{
      const nextRow = data[Math.min(idx + 1, data.length - 1)];
      const t = ease(progress);
      speakerNames.forEach((speaker) => {{
        const current = latestForSpeaker(speaker, idx);
        const target = current
          ? (nextRow && nextRow.speaker === speaker ? nextRow : current)
          : firstForSpeaker(speaker);
        const source = current || target;
        if (!source || !target) return;
        const valence = source.valence + (target.valence - source.valence) * t;
        const arousal = source.arousal + (target.arousal - source.arousal) * t;
        const [x, y] = point(valence, arousal);
        const isMoving = nextRow && nextRow.speaker === speaker && progress > 0;
        ctx.shadowColor = target.color;
        ctx.shadowBlur = isMoving ? 18 : 8;
        ctx.fillStyle = target.color;
        ctx.globalAlpha = current ? 1 : 0.35;
        ctx.beginPath();
        ctx.arc(x, y, isMoving ? 11 : 9, 0, Math.PI * 2);
        ctx.fill();
        ctx.shadowBlur = 0;
        ctx.strokeStyle = "#10151f";
        ctx.lineWidth = 3;
        ctx.stroke();
        ctx.globalAlpha = 1;
        ctx.fillStyle = "#d7dee8";
        ctx.font = "11px system-ui, -apple-system, Segoe UI, sans-serif";
        ctx.fillText(speaker, x + 13, y + 4);
      }});
    }}
    function renderTranscript() {{
      const start = Math.max(0, idx - 5);
      const end = Math.min(data.length, idx + 3);
      utterances.innerHTML = "";
      for (let i = start; i < end; i++) {{
        const row = data[i];
        const item = document.createElement("div");
        item.className = "utterance" + (i === idx ? " active" : "");
        item.style.borderColor = i === idx ? row.color : "transparent";
        item.style.opacity = i > idx ? "0.32" : String(Math.max(0.35, 1 - (idx - i) * 0.12));
        const meta = document.createElement("div");
        meta.className = "speaker";
        meta.style.color = row.color;
        meta.textContent = i === idx ? `${{row.speaker}}  ← now` : row.speaker;
        const text = document.createElement("div");
        text.className = "sentence";
        text.textContent = row.sentence;
        item.appendChild(meta);
        item.appendChild(text);
        utterances.appendChild(item);
      }}
    }}
    function updateStats() {{
      const row = data[idx];
      const shown = data.slice(0, idx + 1);
      const val = shown.reduce((sum, item) => sum + item.valence, 0) / shown.length;
      const aro = shown.reduce((sum, item) => sum + item.arousal, 0) / shown.length;
      speakerEl.textContent = row.speaker;
      speakerEl.style.color = row.color;
      valenceEl.textContent = row.valence.toFixed(2);
      arousalEl.textContent = row.arousal.toFixed(2);
      meanEl.textContent = `V ${{val.toFixed(2)}} · A ${{aro.toFixed(2)}}`;
      turn.textContent = `Turn ${{idx + 1}} / ${{data.length}}`;
      scrub.value = String(idx);
      backBtn.disabled = idx === 0;
      nextBtn.disabled = idx === data.length - 1;
    }}
    function render() {{
      drawAxes();
      drawPath();
      drawSpeakerMarkers();
      renderTranscript();
      updateStats();
    }}
    function setIndex(nextIndex) {{
      idx = clamp(nextIndex, 0, data.length - 1);
      progress = 0;
      render();
    }}
    function tick(ts) {{
      if (!playing) return;
      if (lastTs === null) lastTs = ts;
      const elapsed = ts - lastTs;
      lastTs = ts;
      progress += elapsed / (baseMs / Number(speed.value));
      if (progress >= 1) {{
        progress = 0;
        if (idx >= data.length - 1) {{
          playing = false;
          playBtn.textContent = "▶";
          render();
          return;
        }}
        idx += 1;
      }}
      render();
      requestAnimationFrame(tick);
    }}
    playBtn.addEventListener("click", () => {{
      if (playing) {{
        playing = false;
        playBtn.textContent = "▶";
        return;
      }}
      if (idx >= data.length - 1) setIndex(0);
      playing = true;
      lastTs = null;
      playBtn.textContent = "⏸";
      requestAnimationFrame(tick);
    }});
    backBtn.addEventListener("click", () => {{
      playing = false;
      playBtn.textContent = "▶";
      setIndex(idx - 1);
    }});
    nextBtn.addEventListener("click", () => {{
      playing = false;
      playBtn.textContent = "▶";
      setIndex(idx + 1);
    }});
    scrub.addEventListener("input", (event) => {{
      playing = false;
      playBtn.textContent = "▶";
      setIndex(Number(event.target.value));
    }});
    render();
    </script>
    <style>
    .lp-player {{
      box-sizing: border-box;
      color: #d7dee8;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #0e1117;
      border: 1px solid rgba(255,255,255,0.10);
      border-radius: 8px;
      height: 650px;
      overflow: hidden;
      padding: 14px;
    }}
    .lp-controls {{
      align-items: center;
      display: grid;
      gap: 8px;
      grid-template-columns: 42px 42px 42px 1fr 92px 118px;
      margin-bottom: 12px;
    }}
    .lp-controls button, .lp-controls select {{
      background: #18202d;
      border: 1px solid rgba(255,255,255,0.14);
      border-radius: 6px;
      color: #e9eef6;
      font: inherit;
      height: 36px;
    }}
    .lp-controls button:disabled {{
      color: #596273;
    }}
    #scrub {{
      accent-color: #7e9ec9;
      width: 100%;
    }}
    #turn {{
      color: #aab4c0;
      font-size: 13px;
      text-align: right;
    }}
    .lp-grid {{
      display: grid;
      gap: 14px;
      grid-template-columns: minmax(260px, 0.95fr) minmax(320px, 1.15fr);
      height: 440px;
    }}
    .lp-panel {{
      background: rgba(255,255,255,0.03);
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 8px;
      box-sizing: border-box;
      height: 440px;
      min-height: 440px;
      overflow: hidden;
      padding: 12px;
    }}
    .panel-title {{
      color: #94a3b8;
      font-size: 12px;
      font-weight: 700;
      margin-bottom: 10px;
      text-transform: uppercase;
    }}
    .utterance {{
      border-left: 3px solid transparent;
      border-radius: 6px;
      margin: 4px 0;
      padding: 7px 9px;
      transition: background 160ms ease, opacity 160ms ease;
    }}
    .utterance.active {{
      background: rgba(255,255,255,0.06);
    }}
    #utterances {{
      height: 388px;
      overflow: hidden;
    }}
    .speaker {{
      font-size: 11px;
      font-weight: 750;
      margin-bottom: 3px;
    }}
    .sentence {{
      color: #e7edf5;
      font-size: 13px;
      line-height: 1.35;
    }}
    .viz {{
      display: grid;
      place-items: center;
    }}
    canvas {{
      height: 420px;
      max-width: 100%;
      width: min(100%, 560px);
    }}
    .lp-live {{
      display: grid;
      gap: 10px;
      grid-template-columns: repeat(4, 1fr);
      margin-top: 12px;
    }}
    .lp-live div {{
      background: rgba(255,255,255,0.035);
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 8px;
      padding: 10px;
    }}
    .lp-live span {{
      color: #94a3b8;
      display: block;
      font-size: 11px;
      font-weight: 700;
      margin-bottom: 4px;
      text-transform: uppercase;
    }}
    .lp-live strong {{
      color: #f4f7fb;
      font-size: 17px;
    }}
    @media (max-width: 760px) {{
      .lp-player {{
        height: 950px;
      }}
      .lp-controls, .lp-grid, .lp-live {{
        grid-template-columns: 1fr;
      }}
      .lp-grid {{
        height: 650px;
      }}
      .lp-panel {{
        height: 318px;
        min-height: 318px;
      }}
      #utterances {{
        height: 266px;
      }}
      canvas {{
        height: 294px;
      }}
      #turn {{
        text-align: left;
      }}
    }}
    </style>
    """
    components.html(component_html, height=650, scrolling=False)


def _render_transcript_panel(turns: pd.DataFrame, current_idx: int) -> None:
    st.markdown("**Transcript**")
    window_start = max(0, current_idx - 5)
    for abs_idx in range(window_start, min(len(turns), current_idx + 3)):
        row = turns.iloc[abs_idx]
        speaker = str(row.get("speaker", "Unknown"))
        safe_speaker = html.escape(speaker)
        safe_sentence = html.escape(str(row.get("sentence", "")))
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
                f'{safe_speaker} &nbsp;&larr; now</span><br>'
                f'<span style="font-size:13px;color:#fff;">{safe_sentence}</span><br>'
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
                f'<span style="font-size:9px;color:{color};">{safe_speaker}</span><br>'
                f'<span style="font-size:11px;color:#ccc;">{safe_sentence}</span>'
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
        (-0.8, -0.8, "Sad"), (0.8, -0.8, "Calm"),
        (-0.8, 0.8, "Tense"), (0.8, 0.8, "Excited"),
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
        silence = results.get("dynamics", {}).get("global_metrics", {}).get("silence_total", 0)
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
    _inject_theme()
    csv_path, chapter = _sidebar()
    utterances = _load_transcript(csv_path)
    issues = _validate_transcript(utterances)
    blocking = [issue for issue in issues if issue.startswith("Missing") or "end time" in issue]
    for issue in issues:
        if issue in blocking:
            st.error(issue)
        else:
            st.warning(issue)
    if blocking:
        st.stop()

    if chapter == "① Introduction":
        _chapter_intro(csv_path)
    else:
        with st.spinner("Running analysis…"):
            results = _run_analysis(csv_path)
        if chapter == "② Overview":
            _chapter_overview(results, utterances)
        elif chapter == "③ Affect":
            _chapter_affect(results, utterances)
        elif chapter == "④ Complexity":
            _chapter_complexity(results)
        elif chapter == "⑤ Clinical Markers":
            _chapter_clinical(results)
        elif chapter == "⑥ Dynamics":
            _chapter_dynamics(results)


if __name__ == "__main__":
    main()
