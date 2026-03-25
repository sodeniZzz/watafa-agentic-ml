"""FEATURE ENGINEERING AGENT"""

import logging
import time
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


def _generate_feature_eng_code(state: PipelineState, feedback: str = None):
    eda_output_path = state.get("eda_output_path")
    eda_summary = "EDA report not available."
    if eda_output_path and Path(eda_output_path).exists():
        eda_summary = Path(eda_output_path).read_text(encoding="utf-8")[:2000]

    stage_dir = state["run_dir"] / "feature_engineering"
    prompt = FEATURE_ENGINEERING_PROMPT_TEMPLATE.format(
        train_path=state["train_path"],
        test_path=state["test_path"],
        processed_train_path=stage_dir / "processed_train.csv",
        processed_test_path=stage_dir / "processed_test.csv",
        eda_summary=eda_summary,
    )
    if feedback:
        prompt += f"\nPrevious attempt had the following issues. Please fix them:\n{feedback}\n"

    prompt += "\nWrite only the Python code, no explanations. The code must be self-contained and ready to execute."
    llm_result = invoke_llm(prompt)
    return extract_python_code(llm_result["text"]), llm_result["tokens_in"], llm_result["tokens_out"]


def run_feature_eng_agent(state: PipelineState) -> PipelineState:
    logger.info("Feature engineering node started")
    current_attempt = state.get("fe_attempts", 0)
    new_attempt = current_attempt + 1
    logger.info(f"Feature engineering attempt {new_attempt}")

    start = time.time()
    feedback = state.get("fe_feedback")
    code, tokens_in, tokens_out = _generate_feature_eng_code(state, feedback)

    stage_dir = state["run_dir"] / "feature_engineering"
    stage_dir.mkdir(parents=True, exist_ok=True)
    code_path = stage_dir / f"code_attempt_{new_attempt}.py"
    code_path.write_text(code, encoding="utf-8")
    logger.info("Feature engineering code saved to %s", code_path)

    execution_result = run_python_code(code_path, work_dir=stage_dir, timeout=120)
    duration = time.time() - start

    # Save clean stdout for train agent
    summary_path = stage_dir / "feature_summary.txt"
    summary_path.write_text(execution_result["stdout"], encoding="utf-8")

    # Log to stage report
    state["fe_report"].log_attempt(
        attempt=new_attempt, duration_sec=duration,
        tokens_in=tokens_in, tokens_out=tokens_out,
        returncode=execution_result["returncode"],
        stdout=execution_result["stdout"],
        stderr=execution_result["stderr"],
        error=execution_result["error"],
    )

    logger.info("Feature engineering summary saved to %s", summary_path)

    processed_train = stage_dir / "processed_train.csv"
    processed_test = stage_dir / "processed_test.csv"

    new_state = dict(state)
    new_state.update(
        {
            "fe_attempts": new_attempt,
            "processed_train_path": (
                processed_train if processed_train.exists() else None
            ),
            "processed_test_path": processed_test if processed_test.exists() else None,
            "feature_summary_path": summary_path,
            "fe_valid": False,
        }
    )
    return new_state


def run_fe_validator(state: PipelineState) -> PipelineState:
    logger.info("Feature engineering validator started")

    processed_train = state.get("processed_train_path")
    processed_test = state.get("processed_test_path")

    valid = True
    feedback = []

    if not processed_train or not Path(processed_train).exists():
        valid = False
        feedback.append("Processed train file not found.")
    elif Path(processed_train).stat().st_size == 0:
        valid = False
        feedback.append("Processed train file is empty.")

    if not processed_test or not Path(processed_test).exists():
        valid = False
        feedback.append("Processed test file not found.")
    elif Path(processed_test).stat().st_size == 0:
        valid = False
        feedback.append("Processed test file is empty.")

    # Check for errors via stage report
    last = state["fe_report"].last_attempt
    if last.get("returncode", 0) != 0:
        valid = False
        feedback.append("Code execution failed (non-zero return code).")
        if last.get("stderr"):
            feedback.append(f"Stderr:\n{last['stderr'][-1000:]}")
    elif "Traceback" in last.get("stderr", ""):
        valid = False
        feedback.append("Code execution produced a traceback error.")
        feedback.append(f"Stderr:\n{last['stderr'][-1000:]}")

    feedback_text = (
        "\n".join(feedback) if feedback else "Feature engineering looks good."
    )

    new_state = dict(state)
    new_state["fe_valid"] = valid
    new_state["fe_feedback"] = feedback_text
    state["fe_report"].log_validation(valid, feedback_text)

    logger.info(f"Feature engineering validation completed. Valid: {valid}")
    return new_state
