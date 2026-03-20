"""Eval AGENT ---- PLAN -> CODE -> EXECUTE"""

import json
import logging
from pathlib import Path

from src.state import PipelineState
from src.utils.code_utils import extract_python_code, run_python_code
from src.utils.llm_utils import invoke_llm


logger = logging.getLogger(__name__)


MODEL_FILE_NAMES = {
    "linear_regression": "linear_regression.joblib",
    "random_forest": "random_forest.joblib",
    "catboost": "catboost.joblib",
}

EVAL_PROMPT_TEMPLATE = """You are an ML evaluation expert. Write Python code to evaluate trained regression models on a validation split.

Inputs:
- Processed train dataset: {train_path}
- Model bundles directory: {models_dir}

Requirements:
1. Load the processed train dataset with pandas.
2. Use target column exactly named target.
3. Recreate the same train/validation split with test_size=0.2 and random_state=42.
4. Load these model bundles:
   - {linear_regression_path}
   - {random_forest_path}
   - {catboost_path}
5. Build X by dropping only the target column and keeping the remaining columns exactly as features.
6. For each model:
   - use its feature_columns
   - fill missing values with its fill_values
   - predict on the validation subset
   - calculate MSE
7. Before selecting columns, verify that every feature in feature_columns exists in X_val.
   - if any are missing, raise a clear error that lists the missing feature names
8. Choose the best model by minimum MSE.
9. Save evaluation metrics to JSON:
   - {eval_metrics_path}
10. Save the JSON in exactly this structure:
{{
  "models": {{
    "linear_regression": {{"mse": 0.0}},
    "random_forest": {{"mse": 0.0}},
    "catboost": {{"mse": 0.0}}
  }},
  "best_model_name": "random_forest",
  "best_model_path": "/absolute/path/to/model.joblib"
}}
11. Print a short summary with all MSE values and the best model.
"""


def should_continue_after_eval_validation(state: PipelineState) -> str:
    if state["eval_valid"] or state["eval_attempts"] >= state["eval_max_attempts"]:
        return "submission"
    return "evaluation"


def _generate_eval_code(state: PipelineState, feedback: str) -> str:
    train_path = state["processed_train_path"] or state["train_path"]
    eval_metrics_path = state["run_dir"] / "reports" / "evaluation_metrics.json"
    models_dir = state["run_dir"] / "models"
    prompt = EVAL_PROMPT_TEMPLATE.format(
        train_path=train_path,
        models_dir=models_dir,
        linear_regression_path=models_dir / MODEL_FILE_NAMES["linear_regression"],
        random_forest_path=models_dir / MODEL_FILE_NAMES["random_forest"],
        catboost_path=models_dir / MODEL_FILE_NAMES["catboost"],
        eval_metrics_path=eval_metrics_path,
    )
    if feedback:
        prompt += f"\nPrevious attempt feedback:\n{feedback}\n"

    prompt += "\nWrite only executable Python code. No explanations."
    return extract_python_code(invoke_llm(prompt))


def run_eval_agent(state: PipelineState) -> PipelineState:
    logger.info("Evaluation node started")

    new_attempt = state["eval_attempts"] + 1
    logger.info("Evaluation attempt %s", new_attempt)

    code = _generate_eval_code(state, state["eval_feedback"])

    code_dir = state["run_dir"] / "code"
    code_dir.mkdir(parents=True, exist_ok=True)
    code_path = code_dir / f"eval_attempt_{new_attempt}.py"
    code_path.write_text(code, encoding="utf-8")
    logger.info("Evaluation code saved to %s", code_path)

    execution_result = run_python_code(code, work_dir=code_dir, timeout=300)

    report_path = state["run_dir"] / "reports" / f"eval_report_attempt_{new_attempt}.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_content = f"""STDOUT:
{execution_result['stdout']}

STDERR:
{execution_result['stderr']}

Return code: {execution_result['returncode']}

Error: {execution_result['error']}
"""
    report_path.write_text(report_content, encoding="utf-8")
    logger.info("Evaluation report saved to %s", report_path)

    eval_metrics_path = state["run_dir"] / "reports" / "evaluation_metrics.json"
    eval_metrics = {}
    if eval_metrics_path.exists():
        eval_metrics = json.loads(eval_metrics_path.read_text(encoding="utf-8"))

    next_state = {
        **state,
        "eval_attempts": new_attempt,
        "eval_report_path": report_path,
        "eval_metrics_path": eval_metrics_path if eval_metrics_path.exists() else None,
        "eval_valid": False,
    }

    if eval_metrics:
        best_model_name = eval_metrics["best_model_name"]
        best_model_path = Path(eval_metrics["best_model_path"])
        model_metrics = eval_metrics["models"][best_model_name]
        next_state["best_model_name"] = best_model_name
        next_state["model_path"] = best_model_path
        next_state["metrics"] = {"mse": model_metrics["mse"]}

    return next_state


def run_eval_validator(state: PipelineState) -> PipelineState:
    logger.info("Evaluation validator started")

    valid = True
    feedback = []

    report_path = state["eval_report_path"]
    metrics_path = state["eval_metrics_path"]

    if not report_path.exists():
        valid = False
        feedback.append("Evaluation execution report is missing.")
    else:
        report = report_path.read_text(encoding="utf-8")
        if "Traceback" in report or "Return code: -1" in report or "Return code: 1" in report:
            valid = False
            feedback.append("Evaluation code execution failed.")
        if "Error:" in report and "Error: None" not in report:
            valid = False
            feedback.append("Evaluation execution returned an error.")

    if not metrics_path or not metrics_path.exists():
        valid = False
        feedback.append("Evaluation metrics JSON was not created.")
    else:
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            metrics = {}
            valid = False
            feedback.append("Evaluation metrics JSON could not be parsed.")

        model_metrics = metrics.get("models", {})
        for model_name in MODEL_FILE_NAMES:
            if model_name not in model_metrics or "mse" not in model_metrics[model_name]:
                valid = False
                feedback.append(f"MSE for {model_name} is missing.")
        best_model_path = metrics.get("best_model_path")
        if not metrics.get("best_model_name") or not best_model_path:
            valid = False
            feedback.append("Best model information is missing.")
        elif not Path(best_model_path).exists():
            valid = False
            feedback.append("Best model path does not exist.")

    feedback_text = "\n".join(feedback) if feedback else "Evaluation looks good."

    validation_report_path = (
        state["run_dir"] / "reports" / f"eval_validation_attempt_{state['eval_attempts']}.txt"
    )
    validation_report_path.write_text(
        f"VALID: {valid}\nFEEDBACK:\n{feedback_text}",
        encoding="utf-8",
    )

    logger.info("Evaluation validation completed. Valid: %s", valid)
    return {
        **state,
        "eval_validation_reports": state["eval_validation_reports"] + [validation_report_path],
        "eval_feedback": feedback_text,
        "eval_valid": valid,
    }
