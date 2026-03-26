"""Train AGENT"""

import logging
import time
from pathlib import Path

from src.state import PipelineState
from src.utils.code_utils import extract_python_code, run_python_code
from src.utils.llm_utils import invoke_llm


logger = logging.getLogger(__name__)


MODEL_FILE_NAME = "random_forest.joblib"

TRAIN_PROMPT_TEMPLATE = """You are an ML training expert. Write Python code to tune and train a single RandomForestRegressor for a tabular regression dataset.

Input:
- Processed train dataset: {train_path}
- Target column: {target_column}

Requirements:
1. Load the dataset with pandas.
2. Use target column exactly named {target_column}. Do not hardcode "target" if the name is different.
3. Use only numeric features.
4. Split the rows into train and validation sets with test_size=0.2 and random_state=42.
5. Compute median fill values on the train split only and use them for both train and validation.
6. Tune only one model family: RandomForestRegressor.
7. Use GridSearchCV for a compact hyperparameter search using only the train split.
8. Use cv=2 and scoring="neg_mean_squared_error".
9. Tune at least these parameters with a smaller grid:
   - n_estimators in [150]
   - max_depth in [2, 4, 8, 16]
   - min_samples_split in [2, 4, 6, 8]
   - min_samples_leaf in [1, 2, 4, 6, 8]
   - max_features in ["sqrt", None]
10. Use RandomForestRegressor with random_state=42 and n_jobs=-1.
11. Fit GridSearchCV on the train split only.
12. Do not calculate validation MSE in this step. The evaluation agent will do that separately.
13. After grid search, take best_params_ and train a fresh final RandomForestRegressor on the same train split with the same best params, but use a larger n_estimators value:
   - final_n_estimators = max(best_params["n_estimators"], 300)
14. Save a model bundle with keys model, model_name, feature_columns, fill_values, best_params to:
   - {model_path}
15. Do not train any other models.
16. Print a short summary with the split sizes, best params from grid search, final_n_estimators, and saved model path.

Feature engineering report excerpt:
{feature_report}
"""


def should_continue_after_train_validation(state: PipelineState) -> str:
    if state["train_valid"] or state["train_attempts"] >= state["train_max_attempts"]:
        return "evaluation"
    return "train"


def _generate_train_code(state: PipelineState, feedback: str):
    feature_report = "Feature engineering report not available."
    feature_summary_path = state.get("feature_summary_path")
    if feature_summary_path and Path(feature_summary_path).exists():
        feature_report = Path(feature_summary_path).read_text(encoding="utf-8")[:2000]

    train_path = state["processed_train_path"] or state["train_path"]
    stage_dir = state["run_dir"] / "train"
    model_path = stage_dir / MODEL_FILE_NAME
    prompt = TRAIN_PROMPT_TEMPLATE.format(
        train_path=train_path,
        target_column=state["target_column"],
        model_path=model_path,
        feature_report=feature_report,
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

    # Log to stage report
    state["train_report"].log_attempt(
        attempt=new_attempt, duration_sec=duration,
        tokens_in=tokens_in, tokens_out=tokens_out,
        returncode=execution_result["returncode"],
        stdout=execution_result["stdout"],
        stderr=execution_result["stderr"],
        error=execution_result["error"],
    )

    return {
        **state,
        "train_attempts": new_attempt,
        "train_valid": False,
    }


def run_train_validator(state: PipelineState) -> PipelineState:
    logger.info("Train validator started")

    valid = True
    feedback = []

    model_path = state["run_dir"] / "train" / MODEL_FILE_NAME
    last = state["train_report"].last_attempt

    if last.get("returncode", 0) != 0:
        valid = False
        feedback.append("Training code execution failed.")
        if last.get("stderr"):
            feedback.append(f"Stderr:\n{last['stderr'][-1000:]}")
    if last.get("error"):
        valid = False
        feedback.append(f"Training error: {last['error']}")
    if not model_path.exists():
        valid = False
        feedback.append("RandomForest model artifact was not created.")

    feedback_text = "\n".join(feedback) if feedback else "Training looks good."

    state["train_report"].log_validation(valid, feedback_text)

    logger.info("Train validation completed. Valid: %s", valid)
    return {
        **state,
        "train_feedback": feedback_text,
        "train_valid": valid,
    }
