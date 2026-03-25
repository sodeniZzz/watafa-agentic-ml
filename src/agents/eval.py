"""Eval AGENT"""

import json
import logging
from pathlib import Path

from src.state import PipelineState
from src.utils.code_utils import extract_python_code, run_python_code
from src.utils.llm_utils import invoke_llm


logger = logging.getLogger(__name__)


MODEL_FILE_NAME = "random_forest.joblib"
MODEL_NAME = "random_forest"

EVAL_PROMPT_TEMPLATE = """You are an ML evaluation expert. Write Python code to evaluate trained regression models on a validation split.

Inputs:
- Processed train dataset: {train_path}
- Target column: {target_column}
- Model bundle path: {model_path}

Requirements:
1. Load the processed train dataset with pandas.
2. Use target column exactly named {target_column}. Do not hardcode "target" if the name is different.
3. Recreate the same train/validation split with test_size=0.2 and random_state=42.
4. Load this model bundle:
   - {model_path}
5. The loaded bundle is a Python dict, not the model itself.
6. Read these values from the dict:
   - model = model_bundle["model"]
   - feature_columns = model_bundle["feature_columns"]
   - fill_values = model_bundle["fill_values"]
7. Build X by dropping only the target column and keeping the remaining columns exactly as features.
8. For the model:
   - before prediction, verify that every feature in feature_columns exists in X_val
   - subset X_val to feature_columns only
   - fill missing values with fill_values
   - predict on the validation subset using model.predict(...)
   - calculate MSE
9. Never call predict() on model_bundle because model_bundle is a dict.
10. Do not read feature_columns or fill_values as model attributes. Read them only from the loaded bundle dict.
11. Save evaluation metrics to JSON:
   - {eval_metrics_path}
12. Save the JSON in exactly this structure:
{{
  "models": {{
    "random_forest": {{"mse": 0.0}}
  }},
  "best_model_name": "random_forest",
  "best_model_path": "{model_path}"
}}
13. Print a short summary with RandomForest MSE and the best model.
"""


def should_continue_after_eval_validation(state: PipelineState) -> str:
    if state["eval_valid"] or state["eval_attempts"] >= state["eval_max_attempts"]:
        return "submission"
    return "evaluation"


def _generate_eval_code(state: PipelineState, feedback: str) -> str:
    train_path = state["processed_train_path"] or state["train_path"]
    eval_metrics_path = state["run_dir"] / "reports" / "evaluation_metrics.json"
    model_path = state["run_dir"] / "models" / MODEL_FILE_NAME
    prompt = EVAL_PROMPT_TEMPLATE.format(
        train_path=train_path,
        target_column=state["target_column"],
        model_path=model_path,
        eval_metrics_path=eval_metrics_path,
    )
    if feedback:
        prompt += f"\nPrevious attempt feedback:\n{feedback}\n"

    prompt += "\nWrite only executable Python code. No explanations."
    code = extract_python_code(invoke_llm(prompt))
    code = code.replace("model_bundle.predict(", 'model_bundle["model"].predict(')
    return code


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

    execution_result = run_python_code(code_path, work_dir=code_dir, timeout=300)

    report_path = (
        state["run_dir"] / "reports" / f"eval_report_attempt_{new_attempt}.txt"
    )
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
        next_state["best_model_name"] = MODEL_NAME
        next_state["model_path"] = Path(eval_metrics["best_model_path"])
        next_state["metrics"] = {"mse": eval_metrics["models"][MODEL_NAME]["mse"]}

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
        if (
            "Traceback" in report
            or "Return code: -1" in report
            or "Return code: 1" in report
        ):
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
        if MODEL_NAME not in model_metrics or "mse" not in model_metrics[MODEL_NAME]:
            valid = False
            feedback.append("MSE for RandomForest is missing.")
        best_model_path = metrics.get("best_model_path")
        if metrics.get("best_model_name") != MODEL_NAME or not best_model_path:
            valid = False
            feedback.append("Best model information is missing.")
        elif not Path(best_model_path).exists():
            valid = False
            feedback.append("Best model path does not exist.")

    feedback_text = "\n".join(feedback) if feedback else "Evaluation looks good."

    validation_report_path = (
        state["run_dir"]
        / "reports"
        / f"eval_validation_attempt_{state['eval_attempts']}.txt"
    )
    validation_report_path.write_text(
        f"VALID: {valid}\nFEEDBACK:\n{feedback_text}",
        encoding="utf-8",
    )

    logger.info("Evaluation validation completed. Valid: %s", valid)
    return {
        **state,
        "eval_validation_reports": state["eval_validation_reports"]
        + [validation_report_path],
        "eval_feedback": feedback_text,
        "eval_valid": valid,
    }
