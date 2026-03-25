"""Eval AGENT"""

import json
import logging
import time
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


def _generate_eval_code(state: PipelineState, feedback: str):
    train_path = state["processed_train_path"] or state["train_path"]
    stage_dir = state["run_dir"] / "evaluation"
    eval_metrics_path = stage_dir / "evaluation_metrics.json"
    model_path = state["run_dir"] / "train" / MODEL_FILE_NAME
    prompt = EVAL_PROMPT_TEMPLATE.format(
        train_path=train_path,
        target_column=state["target_column"],
        model_path=model_path,
        eval_metrics_path=eval_metrics_path,
    )
    if feedback:
        prompt += f"\nPrevious attempt feedback:\n{feedback}\n"

    prompt += "\nWrite only executable Python code. No explanations."
    llm_result = invoke_llm(prompt)
    code = extract_python_code(llm_result["text"])
    code = code.replace("model_bundle.predict(", 'model_bundle["model"].predict(')
    return code, llm_result["tokens_in"], llm_result["tokens_out"]


def run_eval_agent(state: PipelineState) -> PipelineState:
    logger.info("Evaluation node started")

    new_attempt = state["eval_attempts"] + 1
    logger.info("Evaluation attempt %s", new_attempt)

    start = time.time()
    code, tokens_in, tokens_out = _generate_eval_code(state, state["eval_feedback"])

    stage_dir = state["run_dir"] / "evaluation"
    stage_dir.mkdir(parents=True, exist_ok=True)
    code_path = stage_dir / f"code_attempt_{new_attempt}.py"
    code_path.write_text(code, encoding="utf-8")
    logger.info("Evaluation code saved to %s", code_path)

    execution_result = run_python_code(code_path, work_dir=stage_dir, timeout=300)
    duration = time.time() - start

    # Log to stage report
    state["eval_report"].log_attempt(
        attempt=new_attempt, duration_sec=duration,
        tokens_in=tokens_in, tokens_out=tokens_out,
        returncode=execution_result["returncode"],
        stdout=execution_result["stdout"],
        stderr=execution_result["stderr"],
        error=execution_result["error"],
    )

    eval_metrics_path = stage_dir / "evaluation_metrics.json"

    next_state = {
        **state,
        "eval_attempts": new_attempt,
        "eval_valid": False,
    }

    if eval_metrics_path.exists():
        eval_metrics = json.loads(eval_metrics_path.read_text(encoding="utf-8"))
        next_state["best_model_name"] = MODEL_NAME
        next_state["model_path"] = Path(eval_metrics["best_model_path"])
        next_state["metrics"] = {"mse": eval_metrics["models"][MODEL_NAME]["mse"]}

    return next_state


def run_eval_validator(state: PipelineState) -> PipelineState:
    logger.info("Evaluation validator started")

    valid = True
    feedback = []

    last = state["eval_report"].last_attempt
    metrics_path = state["run_dir"] / "evaluation" / "evaluation_metrics.json"

    if last.get("returncode", 0) != 0:
        valid = False
        feedback.append("Evaluation code execution failed.")
        if last.get("stderr"):
            feedback.append(f"Stderr:\n{last['stderr'][-1000:]}")
    if last.get("error"):
        valid = False
        feedback.append(f"Evaluation error: {last['error']}")

    if not metrics_path.exists():
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
        if MODEL_NAME not in model_metrics or "mse" not in model_metrics.get(MODEL_NAME, {}):
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

    state["eval_report"].log_validation(valid, feedback_text)

    logger.info("Evaluation validation completed. Valid: %s", valid)
    return {
        **state,
        "eval_feedback": feedback_text,
        "eval_valid": valid,
    }
