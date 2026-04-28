import click
import logging

from .core.registry import AnalyzerRegistry
from .pipeline import LexiPlexPipeline
from .analyzers.affect import AffectAnalyzer
from .modelling.valence_arousal import ValenceArousalModel


@click.command()
@click.argument("transcript", type=click.Path(exists=True), metavar="<TRANSCRIPT_PATH>")
@click.argument("output_dir", type=click.Path(), metavar="<OUTPUT_DIR>")
@click.option("--model", "-m", default=None, help="HuggingFace model name or local path")
@click.option("--language", "-l", default="en", help="spaCy language code (default: en)")
@click.option("--topics", "-t", "use_topics", is_flag=True, help="Enable LDA topic assignment")
@click.option("--verbose", "-v", is_flag=True, help="Enable DEBUG logging")
def main(transcript, output_dir, model, language, use_topics, verbose):
    """Run the LexiPlex affect analysis pipeline."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    va_model = ValenceArousalModel(model or "models/XLM-RoBERTa-base-MSE")
    registry = AnalyzerRegistry()
    registry.register(AffectAnalyzer(model=va_model, use_topics=use_topics))

    pipeline = LexiPlexPipeline(registry=registry, model=va_model, language=language)
    results = pipeline.run(transcript)

    affect = results["affect"]
    print(f"Processed {len(affect.per_sentence)} sentences")
    print("Global metrics:")
    for k, v in affect.global_metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
