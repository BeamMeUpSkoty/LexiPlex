import click
import logging
from .pipeline import ConversationPipeline

@click.command()
@click.argument('transcript', type=click.Path(exists=True), metavar='<TRANSCRIPT_PATH>')
@click.argument('output', type=click.Path(), metavar='<OUTPUT_DIR>')
@click.option('--model', '-m', default=None,
              help='HuggingFace model name for valence/arousal scoring')
@click.option('--topics', '-t', 'num_topics', default=5, type=int,
              help='Number of topics for topic modeling')
@click.option('--verbose', '-v', is_flag=True,
              help='Enable verbose (DEBUG) logging')
def main(transcript, output, model, num_topics, verbose):
    """
    Run the conversation affect analysis pipeline.

    TRANSCRIPT_PATH: Path to the transcript file (CSV, JSON, etc.)
    OUTPUT_DIR: Directory for saving plots and CSV outputs
    """
    # Configure logging
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level,
                        format='%(asctime)s %(levelname)-8s %(name)s: %(message)s')
    logger = logging.getLogger(__name__)

    logger.info('Starting pipeline')
    pipeline = ConversationPipeline(
        transcript_path=transcript,
        output_dir=output,
        model_name=model,
        num_topics=num_topics
    )
    df = pipeline.run()
    logger.info('Pipeline finished: processed %d sentences', len(df))

if __name__ == '__main__':
    main()
