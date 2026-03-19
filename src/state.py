from pathlib import Path
from typing import TypedDict

from src.utils.io_utils import ROOT_PATH


class PipelineState(TypedDict):
    run_dir: Path
    train_path: Path
    test_path: Path
    sample_submission_path: Path
    eda_report_path: Path
    model_path: Path
    submission_path: Path
    metrics: dict[str, float]

    eda_attempts: int
    eda_max_attempts: int
    eda_feedback: str
    eda_validation_reports: list[str]
    eda_valid: bool


def create_initial_state(run_dir: Path) -> PipelineState:
    state: PipelineState = {
        "run_dir": Path(run_dir),
        "train_path": ROOT_PATH / "data" / "train.csv",
        "test_path": ROOT_PATH / "data" / "test.csv",
        "sample_submission_path": ROOT_PATH / "data" / "sample_submission.csv",

        # EDA fields
        "eda_attempts": 0,
        "eda_max_attempts": 2,
        "eda_feedback": None,
        "eda_validation_reports": [],
        "eda_valid": False,
    }
    return state
