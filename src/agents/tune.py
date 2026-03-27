"""Tune AGENT — select best model from exploration and tune with Optuna."""

import json
import logging
import time
from pathlib import Path

from src.state import PipelineState
from src.utils.code_utils import extract_python_code, run_python_code
from src.utils.llm_utils import invoke_llm


logger = logging.getLogger(__name__)


TUNE_PROMPT_TEMPLATE = """You are an ML training expert. Write Python code to tune and train the best model identified from the exploration phase. The task type (regression or classification) is inferred from the EDA report.

Input:
- Processed train dataset: {train_path}
- Target column: {target_column}
- Exploration metrics file: {metrics_path} (JSON file with model performance)
- Previous feedback (if any): {feedback}

Requirements:
1. Load the dataset with pandas.
2. Use target column exactly named {target_column}.
3. **DO NOT do any additional feature engineering or encoding.** The data is already fully processed by the feature engineering step. Use only numeric columns from the loaded dataset. Drop any remaining non-numeric columns (object dtype) before training.
4. Split rows into train and validation sets with test_size=0.2, random_state=42.
5. Fill missing values using median for all columns.
6. Read the exploration metrics from {metrics_path} and select the best model based on the appropriate metric:
   - For regression: choose the model with the lowest MSE (or highest R²).
   - For classification: choose the model with the highest accuracy or F1 (based on EDA report).
7. Perform hyperparameter tuning for that selected model using **Optuna**:
   - Define an objective function that:
     - Takes a trial and suggests hyperparameters specific to the selected model.
     - Trains the model on the training split with suggested parameters.
     - Evaluates on the validation split using the appropriate metric (e.g., negative MSE for regression, accuracy for classification).
   - Use a reasonable search space for the selected model (e.g., for XGBoost: n_estimators, max_depth, learning_rate, etc.; for RandomForest: n_estimators, max_depth, etc.). Important: NEVER run validation over 'max_features' - do not ever pass it to optuna for running exps
   - Run a study with n_trials=30 (or a number that fits within time constraints).
   - Use `optuna.create_study(direction="maximize" or "minimize")` as appropriate.
8. After tuning, retrieve the best parameters and train a final model on the **full train split** (the combined train+validation) with those parameters.
9. Save the model bundle (including model, feature columns, fill values, best params) to:
   {model_path}
   You can use joblib or pickle.
   The keys in the model bundle should be 'model', 'feature_columns', 'fill_values', 'best_params'.
10. Print a short summary with the selected model, validation performance, and best parameters.
11. Write only executable Python code, no explanations.

**Note:** Ensure `optuna` is installed (pip install optuna). Use standard libraries: pandas, numpy, sklearn, optuna.

EDA report excerpt:
{eda_report}

Feature engineering report excerpt:
{feature_report}
"""


def should_continue_after_tune_validation(state: PipelineState) -> str:
    if state["tune_valid"] or state["tune_attempts"] >= state["tune_max_attempts"]:
        return "submission"
    return "tune"


def _generate_tune_code(state: PipelineState, feedback: str):
    feature_report = "Feature engineering report not available."
    feature_summary_path = state.get("feature_summary_path")
    if feature_summary_path and Path(feature_summary_path).exists():
        feature_report = Path(feature_summary_path).read_text(encoding="utf-8")[:2000]

    eda_report = "EDA report not available."
    eda_summary_path = state.get("eda_output_path")
    if eda_summary_path and Path(eda_summary_path).exists():
        eda_report = Path(eda_summary_path).read_text(encoding="utf-8")[:5000]

    train_path = state["processed_train_path"] or state["train_path"]
    stage_dir = state["run_dir"] / "tune"
    model_path = stage_dir / "model.joblib"
    metrics_path = state["run_dir"] / "train" / "exploration_metrics.json"

    prompt = TUNE_PROMPT_TEMPLATE.format(
        train_path=train_path,
        target_column=state["target_column"],
        metrics_path=metrics_path,
        model_path=model_path,
        feature_report=feature_report,
        eda_report=eda_report,
        feedback=feedback if feedback else "None",
    )
    prompt += "\nWrite only executable Python code. No explanations."
    llm_result = invoke_llm(prompt)
    return extract_python_code(llm_result["text"]), llm_result["tokens_in"], llm_result["tokens_out"]


def run_tune_agent(state: PipelineState) -> PipelineState:
    logger.info("Tune node started")

    new_attempt = state["tune_attempts"] + 1
    logger.info("Tune attempt %s", new_attempt)

    start = time.time()
    code, tokens_in, tokens_out = _generate_tune_code(state, state["tune_feedback"])

    stage_dir = state["run_dir"] / "tune"
    stage_dir.mkdir(parents=True, exist_ok=True)
    code_path = stage_dir / f"code_attempt_{new_attempt}.py"
    code_path.write_text(code, encoding="utf-8")
    logger.info("Tune code saved to %s", code_path)

    execution_result = run_python_code(code_path, work_dir=stage_dir, timeout=1800)
    duration = time.time() - start

    state["tune_report"].log_attempt(
        attempt=new_attempt, duration_sec=duration,
        tokens_in=tokens_in, tokens_out=tokens_out,
        returncode=execution_result["returncode"],
        stdout=execution_result["stdout"],
        stderr=execution_result["stderr"],
        error=execution_result["error"],
    )

    model_path = stage_dir / "model.joblib"

    return {
        **state,
        "tune_attempts": new_attempt,
        "tune_valid": False,
        "model_path": model_path if model_path.exists() else None,
    }


def run_tune_validator(state: PipelineState) -> PipelineState:
    logger.info("Tune validator started")

    valid = False
    feedback = []

    model_path = state.get("model_path")
    last = state["tune_report"].last_attempt

    if last.get("returncode", 0) != 0:
        feedback.append("Tuning code execution failed.")
        if last.get("stderr"):
            feedback.append(f"Stderr:\n{last['stderr'][-1000:]}")
    if last.get("error"):
        feedback.append(f"Tuning error: {last['error']}")
    if not model_path or not Path(model_path).exists():
        feedback.append("Final model artifact not created.")
    else:
        valid = True
        feedback.append(f"Tuning successful. Model saved to {model_path}.")

    if not valid and state["tune_attempts"] >= state["tune_max_attempts"]:
        feedback.append("Max attempts reached. Proceeding with possibly missing model.")
        valid = True

    feedback_text = "\n".join(feedback) if feedback else "Tuning looks good."
    state["tune_report"].log_validation(valid, feedback_text)

    logger.info("Tune validation completed. Valid: %s", valid)
    return {
        **state,
        "tune_feedback": feedback_text,
        "tune_valid": valid,
    }
