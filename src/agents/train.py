"""Train AGENT ---- PLAN -> CODE -> EXECUTE"""

import logging
from pathlib import Path

from src.state import PipelineState
from src.utils.code_utils import extract_python_code, run_python_code
from src.utils.llm_utils import invoke_llm


logger = logging.getLogger(__name__)


MODEL_FILE_NAMES = {
    "ridge": "ridge.joblib",
    "random_forest": "random_forest.joblib",
    "catboost": "catboost.joblib",
}

TRAIN_PROMPT_TEMPLATE = """You are an ML training expert. Write Python code to train regression models for a tabular dataset.

Input:
- Processed train dataset: {train_path}

Requirements:
1. Load the dataset with pandas.
2. Use target column exactly named target.
3. Use only numeric features.
4. Split the rows into train and validation sets with test_size=0.2 and random_state=42.
5. Train exactly these models on the train split only:
   - Ridge
   - RandomForestRegressor
   - CatBoostRegressor
6. For each model, compute median fill values on the train split only and use them for training.
7. Use only columns that already exist in the processed dataset. Do not create additional temporary feature columns for model training.
8. Keep resource usage conservative:
   - use n_jobs=1 for sklearn models and tuning
   - do not use large parallel grids
   - keep the search bounded and deterministic
9. Do a very small hyperparameter tuning using only the train split:
   - Ridge: compare these candidates with 3-fold CV using neg_mean_squared_error and keep the best:
     - Pipeline(StandardScaler(), Ridge(alpha=0.01))
     - Pipeline(StandardScaler(), Ridge(alpha=0.1))
     - Pipeline(StandardScaler(), Ridge(alpha=1.0))
     - Pipeline(StandardScaler(), Ridge(alpha=10.0))
     - Pipeline(StandardScaler(), Ridge(alpha=100.0))
   - RandomForestRegressor: tune at least these parameters:
     - n_estimators in [10, 20, 50, 100, 150, 200]
     - max_depth in [2, 4, 8, 16, 24, 32]
     - min_samples_split in [2, 4, 6, 8, 10]
     - min_samples_leaf in [1, 2, 4, 6, 8, 10]
   - CatBoostRegressor: tune at least these parameters:
     - depth in [2, 4, 6, 8, 10]
     - learning_rate in [0.001, 0.01, 0.03, 0.1, 0.3]
     - l2_leaf_reg in [0.001, 0.01, 0.1, 1.0, 3.0, 6.0]
     - use iterations=200
10. Use a small cross-validation on the training split only to choose the best parameters.
11. Prefer RandomizedSearchCV or a small manual random search instead of exhaustive full grid search.
12. Keep the tuning bounded:
   - RandomForestRegressor: around 12 sampled configurations
   - CatBoostRegressor: around 12 sampled configurations
   - cv=3
   - random_state=42 where applicable
13. For CatBoostRegressor:
   - use verbose=0
   - use allow_writing_files=False
14. Save model bundles with keys model, model_name, feature_columns, fill_values, best_params to:
   - {ridge_path}
   - {random_forest_path}
   - {catboost_path}
15. Do not evaluate validation MSE in this step.
16. Print a short summary of the split sizes, best parameters, and saved model files.

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
    models_dir = state["run_dir"] / "models"
    prompt = TRAIN_PROMPT_TEMPLATE.format(
        train_path=train_path,
        ridge_path=models_dir / MODEL_FILE_NAMES["ridge"],
        random_forest_path=models_dir / MODEL_FILE_NAMES["random_forest"],
        catboost_path=models_dir / MODEL_FILE_NAMES["catboost"],
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

    execution_result = run_python_code(code, work_dir=code_dir, timeout=600)

    report_path = state["run_dir"] / "reports" / f"train_report_attempt_{new_attempt}.txt"
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

    candidate_model_paths = [
        str(state["run_dir"] / "models" / file_name)
        for file_name in MODEL_FILE_NAMES.values()
        if (state["run_dir"] / "models" / file_name).exists()
    ]

    return {
        **state,
        "train_attempts": new_attempt,
        "train_report_path": report_path,
        "candidate_model_paths": candidate_model_paths,
        "train_valid": False,
    }


def run_train_validator(state: PipelineState) -> PipelineState:
    logger.info("Train validator started")

    valid = True
    feedback = []

    report_path = state["train_report_path"]
    candidate_model_paths = state["candidate_model_paths"]

    if not report_path.exists():
        valid = False
        feedback.append("Training execution report is missing.")
    else:
        report = Path(report_path).read_text(encoding="utf-8")
        if "Traceback" in report or "Return code: -1" in report or "Return code: 1" in report:
            valid = False
            feedback.append("Training code execution failed.")
        if "Error:" in report and "Error: None" not in report:
            valid = False
            feedback.append("Training execution returned an error.")

    if len(candidate_model_paths) != 3:
        valid = False
        feedback.append("Not all three model artifacts were created.")

    feedback_text = "\n".join(feedback) if feedback else "Training looks good."

    validation_report_path = (
        state["run_dir"] / "reports" / f"train_validation_attempt_{state['train_attempts']}.txt"
    )
    validation_report_path.write_text(
        f"VALID: {valid}\nFEEDBACK:\n{feedback_text}",
        encoding="utf-8",
    )

    logger.info("Train validation completed. Valid: %s", valid)
    return {
        **state,
        "train_validation_reports": state["train_validation_reports"] + [validation_report_path],
        "train_feedback": feedback_text,
        "train_valid": valid,
    }
