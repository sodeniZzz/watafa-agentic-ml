"""FEATURE ENGINEERING AGENT"""

import logging
from pathlib import Path

from src.state import PipelineState
from src.utils.code_utils import (
    extract_python_code,
    run_python_code,
)
from src.utils.llm_utils import invoke_llm

logger = logging.getLogger(__name__)


FEATURE_ENGINEERING_PROMPT_TEMPLATE = """You are a feature engineering expert. Based on the EDA report below, write Python code to create new features for the training and test datasets.

The data files:
- Train: {train_path}
- Test: {test_path}

Your code should:
1. Load train and test CSVs using pandas.
2. Perform feature engineering:
   - Create at least 3 new meaningful features based on insights from EDA.
   - Handle missing values if necessary (e.g., fill with median/mode, or create indicator columns).
   - Encode categorical variables if appropriate (e.g., one-hot encoding for low-cardinality, label encoding for high cardinality).
   - You may also create interaction features, polynomial features, or date-based features if applicable.
3. Save the processed datasets as new CSV files:
   - Train: {processed_train_path}
   - Test: {processed_test_path}
4. Print a brief summary of what has been added / changed
    - New features - their pandas type, type(Categorical / Numeric / ...), description, short explanation how they were calculated
    - New shapes of Train, Test dfs

EDA report excerpt:
{eda_summary}
"""


def should_continue_after_fe_validation(state: PipelineState) -> str:
    if state.get("fe_valid", False) or state.get("fe_attempts", 0) >= state.get(
        "fe_max_attempts", 2
    ):
        return "train"
    else:
        return "feature_engineering"


def _generate_feature_eng_code(state: PipelineState, feedback: str = None) -> str:
    eda_report_path = state["eda_report_path"]
    eda_summary = (
        eda_report_path.read_text(encoding="utf-8")[:2000]
        if eda_report_path.exists()
        else "EDA report not available."
    )

    prompt = FEATURE_ENGINEERING_PROMPT_TEMPLATE.format(
        train_path=state["train_path"],
        test_path=state["test_path"],
        processed_train_path=state["run_dir"] / "processed_train.csv",
        processed_test_path=state["run_dir"] / "processed_test.csv",
        eda_summary=eda_summary,
    )
    if feedback:
        prompt += f"\nPrevious attempt had the following issues. Please fix them:\n{feedback}\n"

    prompt += "\nWrite only the Python code, no explanations. The code must be self-contained and ready to execute."
    return extract_python_code(invoke_llm(prompt))


def run_feature_eng_agent(state: PipelineState) -> PipelineState:
    logger.info("Feature engineering node started")
    current_attempt = state.get("fe_attempts", 0)
    new_attempt = current_attempt + 1
    logger.info(f"Feature engineering attempt {new_attempt}")

    feedback = state.get("fe_feedback")
    code = _generate_feature_eng_code(state, feedback)

    # Save code
    code_dir = state["run_dir"] / "code"
    code_dir.mkdir(parents=True, exist_ok=True)
    code_path = code_dir / f"feature_eng_attempt_{new_attempt}.py"
    code_path.write_text(code, encoding="utf-8")
    logger.info("Feature engineering code saved to %s", code_path)

    # Execute code
    execution_result = run_python_code(
        code_path, work_dir=state["run_dir"] / "code", timeout=120
    )

    # Save execution report
    report_content = f"""STDOUT:
{execution_result['stdout']}

STDERR:
{execution_result['stderr']}

Return code: {execution_result['returncode']}
"""
    report_path = (
        state["run_dir"] / "reports" / f"feature_eng_report_attempt_{new_attempt}.txt"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_content, encoding="utf-8")
    logger.info("Feature engineering report saved to %s", report_path)

    processed_train = state["run_dir"] / "processed_train.csv"
    processed_test = state["run_dir"] / "processed_test.csv"

    new_state = dict(state)
    new_state.update(
        {
            "fe_attempts": new_attempt,
            "processed_train_path": (
                processed_train if processed_train.exists() else None
            ),
            "processed_test_path": processed_test if processed_test.exists() else None,
            "feature_eng_report_path": report_path,
            "fe_valid": False,  # not validated yet
            # Keep feedback (validator will overwrite)
        }
    )
    return new_state


def run_fe_validator(state: PipelineState) -> PipelineState:
    logger.info("Feature engineering validator started")

    processed_train = state.get("processed_train_path")
    processed_test = state.get("processed_test_path")
    report_path = state.get("feature_eng_report_path")

    valid = True
    feedback = []

    if not processed_train or not Path(processed_train).exists():
        valid = False
        feedback.append("Processed train file not found.")
    if not processed_test or not Path(processed_test).exists():
        valid = False
        feedback.append("Processed test file not found.")

    # If files exist, optionally check they are non‑empty
    if (
        processed_train
        and Path(processed_train).exists()
        and Path(processed_train).stat().st_size == 0
    ):
        valid = False
        feedback.append("Processed train file is empty.")
    if (
        processed_test
        and Path(processed_test).exists()
        and Path(processed_test).stat().st_size == 0
    ):
        valid = False
        feedback.append("Processed test file is empty.")

    # Check for errors in execution report
    if report_path and report_path.exists():
        report = report_path.read_text(encoding="utf-8")
        if "Traceback" in report or "Error" in report:
            valid = False
            feedback.append("Code execution produced an error (see report).")
        if "Return code: 1" in report:
            valid = False
            feedback.append("Process exited with non‑zero code.")
    else:
        valid = False
        feedback.append("Execution report missing.")

    feedback_text = (
        "\n".join(feedback) if feedback else "Feature engineering looks good."
    )

    # Save validation report
    attempt = state.get("fe_attempts", 0)
    val_report_path = (
        state["run_dir"] / "reports" / f"fe_validation_attempt_{attempt}.txt"
    )
    val_report_path.write_text(
        f"VALID: {valid}\nFEEDBACK:\n{feedback_text}", encoding="utf-8"
    )

    new_state = dict(state)
    new_state["fe_valid"] = valid
    new_state["fe_feedback"] = feedback_text
    reports = new_state.get("fe_validation_reports", [])
    reports.append(val_report_path)
    new_state["fe_validation_reports"] = reports

    logger.info(f"Feature engineering validation completed. Valid: {valid}")
    return new_state
