import os
import logging
from datetime import datetime
from pathlib import Path

from src.graph import run_graph
from src.logger.logger import setup_logging
from src.state import PipelineState, create_initial_state
from src.utils.io_utils import ROOT_PATH
from src.utils.kaggle_utils import prepare_kaggle_data


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
    logger.info("Run directory: %s", run_dir)

    state = create_initial_state(run_dir)
    state.update(prepare_kaggle_data(ROOT_PATH / "data"))

    logger.info("Train data: %s", state["train_path"])
    logger.info("Test data: %s", state["test_path"])
    logger.info("Sample submission: %s", state["sample_submission_path"])
    logger.info("Target column: %s", state["target_column"])

    return state


def run_pipeline() -> PipelineState:
    """Run the minimal MVP pipeline."""
    state = bootstrap_run()

    logger = logging.getLogger(__name__)
    logger.info("Pipeline started")
    state = run_graph(state)
    logger.info("Pipeline finished")

    return state


if __name__ == "__main__":
    run_pipeline()
