"""Submission AGENT"""

import logging
import time
from pathlib import Path

import joblib
import pandas as pd

from src.state import PipelineState
from src.utils.kaggle_utils import submit_to_kaggle


logger = logging.getLogger(__name__)


def build_submission_file(state: PipelineState) -> Path:
    model_bundle = joblib.load(state["model_path"])
    model = model_bundle["model"]
    feature_columns = model_bundle["feature_columns"]
    fill_values = model_bundle["fill_values"]

    test_path = state["processed_test_path"] or state["test_path"]
    test_df = pd.read_csv(test_path)
    sample_submission = pd.read_csv(state["sample_submission_path"])

    x_test = test_df[feature_columns].fillna(fill_values)
    predictions = model.predict(x_test)

    submission = sample_submission.copy()
    target_col = submission.columns[-1]
    submission[target_col] = predictions

    stage_dir = state["run_dir"] / "submission"
    stage_dir.mkdir(parents=True, exist_ok=True)
    submission_path = stage_dir / "submission.csv"
    submission.to_csv(submission_path, index=False)
    return submission_path


def run_submission_agent(state: PipelineState) -> PipelineState:
    logger.info("Submission node started")

    if "model_path" not in state:
        logger.error("Submission skipped: best model path is missing")
        return state

    start = time.time()
    submission_path = build_submission_file(state)
    logger.info("Submission saved to %s", submission_path)

    submit_to_kaggle(submission_path, "Agentic ML pipeline submission")
    logger.info("Submission sent to Kaggle")
    duration = time.time() - start

    state["submission_report"].log_attempt(
        attempt=1, duration_sec=duration, returncode=0,
        stdout=f"Submission saved to {submission_path}",
    )

    return {
        **state,
        "submission_path": submission_path,
    }
