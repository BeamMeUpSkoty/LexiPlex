import json
import logging
import os

import click
import pandas as pd

from .core.registry import AnalyzerRegistry
from .pipeline import LexiPlexPipeline
from .analyzers.affect import AffectAnalyzer
from .analyzers.clinical import ClinicalMarkerAnalyzer
from .analyzers.complexity import ComplexityAnalyzer
from .analyzers.dynamics import DynamicsAnalyzer


@click.command()
@click.argument("transcript", type=click.Path(exists=True), metavar="<TRANSCRIPT_PATH>")
@click.argument("output_dir", type=click.Path(), metavar="<OUTPUT_DIR>")
@click.option("--model", "-m", default=None, help="HuggingFace model name or local path")
@click.option("--language", "-l", default="en", help="spaCy language code (default: en)")
@click.option("--topics", "-t", "use_topics", is_flag=True, help="Enable LDA topic assignment")
@click.option("--verbose", "-v", is_flag=True, help="Enable DEBUG logging")
def main(transcript, output_dir, model, language, use_topics, verbose):
    """Run the LexiPlex analysis pipeline and save results to OUTPUT_DIR."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    os.makedirs(output_dir, exist_ok=True)

    # Load model lazily so the CLI stays importable without torch
    from .modelling.valence_arousal import ValenceArousalModel
    import spacy

    va_model = ValenceArousalModel(model or "models/XLM-RoBERTa-base-MSE")
    nlp = spacy.load(_spacy_model(language))

    registry = AnalyzerRegistry()
    registry.register(AffectAnalyzer(model=va_model, use_topics=use_topics))
    registry.register(ComplexityAnalyzer(nlp=nlp))
    registry.register(ClinicalMarkerAnalyzer())
    registry.register(DynamicsAnalyzer())

    pipeline = LexiPlexPipeline(registry=registry, model=va_model, language=language)
    results = pipeline.run(transcript)

    # ── 1. Merged features CSV ────────────────────────────────────────────────
    # Start with the affect DataFrame (has all base columns + valence/arousal)
    base = results["affect"].per_sentence.copy()
    analyzer_cols = {
        "complexity": ["word_count", "type_token_ratio", "lexical_density", "coherence_to_prev"],
        "clinical":   ["hedging_rate", "certainty_rate", "self_ref_rate", "negation_density", "is_question"],
    }
    for name, cols in analyzer_cols.items():
        if name in results:
            src = results[name].per_sentence
            for col in cols:
                if col in src.columns:
                    base[col] = src[col].values

    features_path = os.path.join(output_dir, "features.csv")
    base.to_csv(features_path, index=False)
    logger.info("Saved %s", features_path)

    # ── 2. Global metrics JSON ────────────────────────────────────────────────
    global_metrics: dict = {}
    for name, result in results.items():
        global_metrics[name] = result.global_metrics
    metrics_path = os.path.join(output_dir, "global_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(global_metrics, f, indent=2)
    logger.info("Saved %s", metrics_path)

    # ── 3. Dynamics per-speaker CSV ───────────────────────────────────────────
    if "dynamics" in results:
        per_speaker = results["dynamics"].metadata.get("per_speaker")
        if per_speaker is not None and isinstance(per_speaker, pd.DataFrame):
            ps_path = os.path.join(output_dir, "dynamics_per_speaker.csv")
            per_speaker.to_csv(ps_path)
            logger.info("Saved %s", ps_path)

    # ── 4. Circumplex PNG ─────────────────────────────────────────────────────
    if "valence" in base.columns and "arousal" in base.columns:
        from .plotting.circumplex import plot_circumplex
        group_col = "speaker" if "speaker" in base.columns else None
        if group_col:
            fig = plot_circumplex(base, group_col)
            plot_path = os.path.join(output_dir, "circumplex.png")
            fig.savefig(plot_path, bbox_inches="tight")
            import matplotlib.pyplot as plt
            plt.close(fig)
            logger.info("Saved %s", plot_path)

    # ── 5. stdout summary ─────────────────────────────────────────────────────
    n = len(base)
    print(f"\nProcessed {n} sentences → {output_dir}/")
    print(f"  features.csv          ({n} rows, {len(base.columns)} columns)")
    print(f"  global_metrics.json")
    if "dynamics" in results:
        print(f"  dynamics_per_speaker.csv")
    if group_col:
        print(f"  circumplex.png")
    print()
    print("Affect global metrics:")
    for k, v in results["affect"].global_metrics.items():
        print(f"  {k}: {v:.4f}")


def _spacy_model(language: str) -> str:
    return {
        "en": "en_core_web_sm",
        "de": "de_core_news_sm",
        "fr": "fr_core_news_sm",
        "es": "es_core_news_sm",
    }.get(language, f"{language}_core_news_sm")


if __name__ == "__main__":
    main()
