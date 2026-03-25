"""Train AGENT"""

import logging
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


def _generate_train_code(state: PipelineState, feedback: str) -> str:
    feature_report = "Feature engineering report not available."
    feature_report_path = state["feature_eng_report_path"]
    if feature_report_path and Path(feature_report_path).exists():
        feature_report = Path(feature_report_path).read_text(encoding="utf-8")[:2000]

    train_path = state["processed_train_path"] or state["train_path"]
    model_path = state["run_dir"] / "models" / MODEL_FILE_NAME
    prompt = TRAIN_PROMPT_TEMPLATE.format(
        train_path=train_path,
        target_column=state["target_column"],
        model_path=model_path,
        feature_report=feature_report,
    )
    if feedback:
        prompt += f"\nPrevious attempt feedback:\n{feedback}\n"

    prompt += "\nWrite only executable Python code. No explanations."
    return extract_python_code(invoke_llm(prompt))


def run_train_agent(state: PipelineState) -> PipelineState:
    logger.info("Train node started")

    new_attempt = state["train_attempts"] + 1
    logger.info("Train attempt %s", new_attempt)

    code = _generate_train_code(state, state["train_feedback"])

    code_dir = state["run_dir"] / "code"
    code_dir.mkdir(parents=True, exist_ok=True)
    code_path = code_dir / f"train_attempt_{new_attempt}.py"
    code_path.write_text(code, encoding="utf-8")
    logger.info("Train code saved to %s", code_path)

    execution_result = run_python_code(code_path, work_dir=code_dir, timeout=1800)

    report_path = (
        state["run_dir"] / "reports" / f"train_report_attempt_{new_attempt}.txt"
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
    logger.info("Train report saved to %s", report_path)

    return {
        **state,
        "train_attempts": new_attempt,
        "train_report_path": report_path,
        "train_valid": False,
    }


def run_train_validator(state: PipelineState) -> PipelineState:
    logger.info("Train validator started")

    valid = True
    feedback = []

    report_path = state["train_report_path"]
    model_path = state["run_dir"] / "models" / MODEL_FILE_NAME

    if not report_path.exists():
        valid = False
        feedback.append("Training execution report is missing.")
    else:
        report = Path(report_path).read_text(encoding="utf-8")
        if (
            "Traceback" in report
            or "Return code: -1" in report
            or "Return code: 1" in report
        ):
            valid = False
            feedback.append("Training code execution failed.")
        if "Error:" in report and "Error: None" not in report:
            valid = False
            feedback.append("Training execution returned an error.")

    if not model_path.exists():
        valid = False
        feedback.append("RandomForest model artifact was not created.")

    feedback_text = "\n".join(feedback) if feedback else "Training looks good."

    validation_report_path = (
        state["run_dir"]
        / "reports"
        / f"train_validation_attempt_{state['train_attempts']}.txt"
    )
    validation_report_path.write_text(
        f"VALID: {valid}\nFEEDBACK:\n{feedback_text}",
        encoding="utf-8",
    )

    logger.info("Train validation completed. Valid: %s", valid)
    return {
        **state,
        "train_validation_reports": state["train_validation_reports"]
        + [validation_report_path],
        "train_feedback": feedback_text,
        "train_valid": valid,
    }
