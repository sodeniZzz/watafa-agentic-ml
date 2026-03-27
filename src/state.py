from pathlib import Path
from typing import Any, TypedDict

from src.utils.io_utils import ROOT_PATH
from src.utils.metrics_utils import StageReport


class PipelineState(TypedDict, total=False):
    run_dir: Path
    train_path: Path
    test_path: Path
    sample_submission_path: Path
    target_column: str
    model_path: Path
    submission_path: Path
    metrics: dict[str, float]
    best_model_name: str

    # StageReport objects (one per agent)
    eda_report: Any
    fe_report: Any
    train_report: Any
    eval_report: Any
    submission_report: Any

    # Clean content paths for next agent
    eda_output_path: Path
    feature_summary_path: Path

    # EDA
    eda_attempts: int
    eda_max_attempts: int
    eda_feedback: str
    eda_valid: bool

    # Feature engineering
    fe_attempts: int
    fe_max_attempts: int
    fe_feedback: str
    fe_valid: bool
    processed_train_path: str
    processed_test_path: str

    # Train
    train_attempts: int
    train_max_attempts: int
    train_feedback: str
    train_valid: bool
    train_phase: str                # "explore" or "tune"
    exploration_results: list[dict]  # list of {model_name, metrics} from exploration
    selected_model: str              # chosen model name after exploration
    exploration_metrics_path: str

    # Eval
    eval_attempts: int
    eval_max_attempts: int
    eval_feedback: str
    eval_valid: bool


def create_initial_state(run_dir: Path) -> PipelineState:
    run_dir = Path(run_dir)
    return {
        "run_dir": run_dir,
        "train_path": ROOT_PATH / "data" / "train.csv",
        "test_path": ROOT_PATH / "data" / "test.csv",
        "sample_submission_path": ROOT_PATH / "data" / "sample_submission.csv",
        "target_column": "target",

        "model_path": None,
        "submission_path": None,
        "metrics": None,
        "best_model_name": None,

        # Stage reports
        "eda_report": StageReport(run_dir / "eda"),
        "fe_report": StageReport(run_dir / "feature_engineering"),
        "train_report": StageReport(run_dir / "train"),
        "eval_report": StageReport(run_dir / "evaluation"),
        "submission_report": StageReport(run_dir / "submission"),

        "eda_output_path": None,
        "feature_summary_path": None,

        "eda_attempts": 0, "eda_max_attempts": 3,
        "eda_feedback": None, "eda_valid": False,

        "processed_train_path": None, "processed_test_path": None,
        "fe_attempts": 0, "fe_max_attempts": 3,
        "fe_feedback": None, "fe_valid": False,

        "train_attempts": 0, "train_max_attempts": 3,
        "train_feedback": None, "train_valid": False,
        "train_phase": "explore",
        "exploration_results": [],
        "selected_model": None,
        "exploration_metrics_path": None,

        "eval_attempts": 0, "eval_max_attempts": 3,
        "eval_feedback": None, "eval_valid": False,
    }
