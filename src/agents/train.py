"""Train AGENT — explore multiple models and compare validation performance."""

import json
import logging
import time
from pathlib import Path

from src.state import PipelineState
from src.utils.code_utils import extract_python_code, run_python_code
from src.utils.llm_utils import invoke_llm


logger = logging.getLogger(__name__)


EXPLORE_PROMPT_TEMPLATE = """You are an ML training expert. Write Python code to train multiple models and compare their validation performance. The task type (regression or classification) will be determined from the EDA report.

Input:
- Processed train dataset: {train_path}
- Target column: {target_column}
- Task type: inferred from EDA report (regression, binary classification, or multiclass classification)

Requirements:
1. Load the dataset with pandas.
2. Use target column exactly named {target_column}.
3. **DO NOT do any additional feature engineering or encoding.** The data is already fully processed by the feature engineering step. Use only numeric columns from the loaded dataset. Drop any remaining non-numeric columns (object dtype) before training.
4. Split data into train and validation (test_size=0.2, random_state=42).
5. Fill missing values using median for all columns.
6. Based on the task type (identified from the EDA report):
   - If **regression**, train the following models with default parameters:
     - RandomForestRegressor
     - XGBRegressor
     - LGBMRegressor
     - GradientBoostingRegressor
     - ExtraTreesRegressor
     - CatBoostRegressor
     - Ridge
     - Lasso
     - ElasticNet
     - KNeighborsRegressor
     Compute metrics: Mean Squared Error (MSE), Mean Absolute Error (MAE), R² score.
   - If **binary classification** (2 classes), train:
     - RandomForestClassifier
     - XGBClassifier
     - LGBMClassifier
     - GradientBoostingClassifier
     - LogisticRegression
     - SVC (with probability=True for metrics)
     - KNeighborsClassifier
     - CatBoostClassifier
     Compute metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC.
   - If **multiclass classification** (>2 classes), train:
     - RandomForestClassifier
     - XGBClassifier
     - LGBMClassifier
     - GradientBoostingClassifier
     - LogisticRegression (multinomial)
     - SVC (with probability=True)
     - KNeighborsClassifier
     - CatBoostClassifier
     Compute metrics: Accuracy, Macro F1-score, Weighted F1-score.
7. For each model, fit on train, predict on validation, compute the appropriate metrics.
8. Save the metrics for each model in a JSON file at:
   {metrics_path}
   Format: {{"model_name": {{"metric1": value, "metric2": value, ...}}}}
   Use the appropriate metric names (e.g., for regression: "mse", "mae", "r2"; for classification: "accuracy", "f1_macro", etc.).
9. Print a short summary of the results (e.g., top 3 models by main metric).
10. Do not save the models (only the JSON report is needed).
11. Write only executable Python code, no explanations.

EDA report excerpt:
{eda_report}

Feature engineering report excerpt:
{feature_report}
"""


def should_continue_after_train_validation(state: PipelineState) -> str:
    if state["train_valid"] or state["train_attempts"] >= state["train_max_attempts"]:
        return "tune"
    return "train"


def _generate_train_code(state: PipelineState, feedback: str):
    feature_report = "Feature engineering report not available."
    feature_summary_path = state.get("feature_summary_path")
    if feature_summary_path and Path(feature_summary_path).exists():
        feature_report = Path(feature_summary_path).read_text(encoding="utf-8")[:2000]

    eda_report = "EDA report not available."
    eda_summary_path = state.get("eda_output_path")
    if eda_summary_path and Path(eda_summary_path).exists():
        eda_report = Path(eda_summary_path).read_text(encoding="utf-8")[:5000]

    train_path = state["processed_train_path"] or state["train_path"]
    stage_dir = state["run_dir"] / "train"
    metrics_path = stage_dir / "exploration_metrics.json"

    prompt = EXPLORE_PROMPT_TEMPLATE.format(
        train_path=train_path,
        target_column=state["target_column"],
        metrics_path=metrics_path,
        feature_report=feature_report,
        eda_report=eda_report,
    )
    if feedback:
        prompt += f"\nPrevious attempt feedback:\n{feedback}\n"
    prompt += "\nWrite only executable Python code. No explanations."
    llm_result = invoke_llm(prompt)
    return extract_python_code(llm_result["text"]), llm_result["tokens_in"], llm_result["tokens_out"]


def run_train_agent(state: PipelineState) -> PipelineState:
    logger.info("Train node started")

    new_attempt = state["train_attempts"] + 1
    logger.info("Train attempt %s", new_attempt)

    start = time.time()
    code, tokens_in, tokens_out = _generate_train_code(state, state["train_feedback"])

    stage_dir = state["run_dir"] / "train"
    stage_dir.mkdir(parents=True, exist_ok=True)
    code_path = stage_dir / f"code_attempt_{new_attempt}.py"
    code_path.write_text(code, encoding="utf-8")
    logger.info("Train code saved to %s", code_path)

    execution_result = run_python_code(code_path, work_dir=stage_dir, timeout=1800)
    duration = time.time() - start

    state["train_report"].log_attempt(
        attempt=new_attempt, duration_sec=duration,
        tokens_in=tokens_in, tokens_out=tokens_out,
        returncode=execution_result["returncode"],
        stdout=execution_result["stdout"],
        stderr=execution_result["stderr"],
        error=execution_result["error"],
    )

    metrics_path = stage_dir / "exploration_metrics.json"

    return {
        **state,
        "train_attempts": new_attempt,
        "train_valid": False,
        "exploration_metrics_path": metrics_path if metrics_path.exists() else None,
    }


def run_train_validator(state: PipelineState) -> PipelineState:
    logger.info("Train validator started")

    valid = False
    feedback = []

    last = state["train_report"].last_attempt
    if last.get("returncode", 0) != 0:
        feedback.append("Training code execution failed.")
        if last.get("stderr"):
            feedback.append(f"Stderr:\n{last['stderr'][-1000:]}")
    if last.get("error"):
        feedback.append(f"Training error: {last['error']}")

    metrics_path = state.get("exploration_metrics_path")
    if metrics_path and Path(metrics_path).exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
        if metrics:
            valid = True
            feedback.append(f"Exploration completed. {len(metrics)} models evaluated.")
        else:
            feedback.append("Exploration metrics file is empty.")
    else:
        feedback.append("Exploration metrics file missing. Training code may have failed.")

    feedback_text = "\n".join(feedback) if feedback else "Training looks good."
    state["train_report"].log_validation(valid, feedback_text)

    logger.info("Train validation completed. Valid: %s", valid)
    return {
        **state,
        "train_feedback": feedback_text,
        "train_valid": valid,
    }
