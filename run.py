import logging
import time
from datetime import datetime
from pathlib import Path

from src.graph import run_graph
from src.logger.logger import setup_logging, log_header, format_duration
from src.state import PipelineState, create_initial_state
from src.utils.io_utils import ROOT_PATH
from src.utils.kaggle_utils import get_kaggle_competition, prepare_kaggle_data
from src.utils.guardrails import validate_csv
from src.utils.rag import init_store


def create_run_dir() -> Path:
    """Create a timestamped directory for one pipeline run."""
    run_id = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    run_dir = ROOT_PATH / "artifacts" / run_id

    run_dir.mkdir(parents=True, exist_ok=True)

    return run_dir


def bootstrap_run() -> PipelineState:
    """Initialize logging, artifacts directory, and the initial pipeline state."""
    run_dir = create_run_dir()
    setup_logging(run_dir)

    logger = logging.getLogger(__name__)

    state = create_initial_state(run_dir)
    state.update(prepare_kaggle_data(ROOT_PATH / "data"))

    data_files = sorted(f.name for f in (ROOT_PATH / "data").glob("*.csv"))
    log_header(
        "WATAFA Pipeline Started",
        f"Run: {run_dir}",
        f"Competition: {get_kaggle_competition()}",
        f"Downloaded: {', '.join(data_files)}",
        f"Target column (auto-detected): {state['target_column']}",
    )

    for csv_file in (ROOT_PATH / "data").glob("*.csv"):
        csv_warnings = validate_csv(str(csv_file))
        if csv_warnings:
            for w in csv_warnings:
                logger.warning("CSV guardrail [%s]: %s", csv_file.name, w)

    init_store()
    logger.info("RAG store initialized")

    return state


def run_pipeline() -> PipelineState:
    """Run the minimal MVP pipeline."""
    state = bootstrap_run()

    logger = logging.getLogger(__name__)
    t0 = time.monotonic()
    state = run_graph(state)
    elapsed = time.monotonic() - t0

    log_header(f"Pipeline finished in {format_duration(elapsed)}")

    return state


if __name__ == "__main__":
    run_pipeline()
