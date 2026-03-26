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

EXPLORE_PROMPT_TEMPLATE = """You are an ML training expert. Write Python code to train multiple regression models and compare their validation performance.

Input:
- Processed train dataset: {train_path}
- Target column: {target_column}

Requirements:
1. Load the dataset with pandas.
2. Use target column exactly named {target_column}.
3. Use only numeric features (automatically select numeric columns).
4. Split data into train and validation (test_size=0.2, random_state=42).
5. Fill missing values using median from train split.
6. Train the following models with default parameters (or simple default if needed):
   - RandomForestRegressor (from sklearn.ensemble)
   - XGBRegressor (import xgboost)
   - LGBMRegressor (import lightgbm)
   - GradientBoostingRegressor (from sklearn.ensemble)
   - ExtraTreesRegressor (from sklearn.ensemble)
   - CatBoostRegressor (import catboost)
   - Ridge (from sklearn.linear_model)
   - Lasso (from sklearn.linear_model)
   - ElasticNet (from sklearn.linear_model)
   - KNeighborsRegressor (from sklearn.neighbors)

7. For each model, fit on train, predict on validation, compute:
   - Mean Squared Error (MSE)
   - Mean Absolute Error (MAE)
   - R² score
8. Save the metrics for each model in a JSON file at:
   {metrics_path}
   Format: {{"model_name": {{"mse": ..., "mae": ..., "r2": ...}}}}
9. Print a short summary of the results.
10. Do not save the models (only the JSON report is needed).
11. Write only executable Python code, no explanations.

Feature engineering report excerpt:
{feature_report}
"""

TUNE_PROMPT_TEMPLATE = """You are an ML training expert. Write Python code to tune and train a single {model_name} model for a tabular regression dataset.

Input:
- Processed train dataset: {train_path}
- Target column: {target_column}
- Selected model: {model_name}
- Previous feedback (if any): {feedback}

Requirements:
1. Load the dataset with pandas.
2. Use target column exactly named {target_column}.
3. Use only numeric features.
4. Split rows into train and validation sets with test_size=0.2, random_state=42.
5. Fill missing values using median from train split.
6. Perform hyperparameter tuning for {model_name} using GridSearchCV or RandomizedSearchCV (cv=3, scoring="neg_mean_squared_error").
7. Use a compact search space (e.g., for XGBoost: n_estimators, max_depth, learning_rate).
8. After tuning, train a final model on the full train split with the best parameters.
9. Save the model bundle (including model, feature columns, fill values, best params) to:
   {model_path}
10. Print a short summary with validation performance and best parameters.
11. Write only executable Python code, no explanations.

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
    model_path = stage_dir / "model.joblib"          # for tuning phase
    metrics_path = stage_dir / "exploration_metrics.json"

    phase = state.get("train_phase", "explore")

    if phase == "explore":
        prompt = EXPLORE_PROMPT_TEMPLATE.format(
            train_path=train_path,
            target_column=state["target_column"],
            metrics_path=metrics_path,
            feature_report=feature_report,
        )
        if feedback:
            prompt += f"\nPrevious attempt feedback:\n{feedback}\n"
        prompt += "\nWrite only executable Python code. No explanations."
        llm_result = invoke_llm(prompt)
        return extract_python_code(llm_result["text"]), llm_result["tokens_in"], llm_result["tokens_out"], phase

    else:  # tune
        selected_model = state.get("selected_model", "RandomForestRegressor")
        prompt = TUNE_PROMPT_TEMPLATE.format(
            train_path=train_path,
            target_column=state["target_column"],
            model_name=selected_model,
            model_path=model_path,
            feature_report=feature_report,
            feedback=feedback if feedback else "None",
        )
        prompt += "\nWrite only executable Python code. No explanations."
        llm_result = invoke_llm(prompt)
        return extract_python_code(llm_result["text"]), llm_result["tokens_in"], llm_result["tokens_out"], phase

def run_train_agent(state: PipelineState) -> PipelineState:
    logger.info("Train node started (phase: %s)", state.get("train_phase", "explore"))

    new_attempt = state["train_attempts"] + 1
    logger.info("Train attempt %s", new_attempt)

    start = time.time()
    code, tokens_in, tokens_out, phase = _generate_train_code(state, state["train_feedback"])

    stage_dir = state["run_dir"] / "train"
    stage_dir.mkdir(parents=True, exist_ok=True)
    code_path = stage_dir / f"code_attempt_{new_attempt}_{phase}.py"
    code_path.write_text(code, encoding="utf-8")
    logger.info("Train code saved to %s", code_path)

    execution_result = run_python_code(code_path, work_dir=stage_dir, timeout=1800)
    duration = time.time() - start

    # Log attempt
    state["train_report"].log_attempt(
        attempt=new_attempt, duration_sec=duration,
        tokens_in=tokens_in, tokens_out=tokens_out,
        returncode=execution_result["returncode"],
        stdout=execution_result["stdout"],
        stderr=execution_result["stderr"],
        error=execution_result["error"],
    )

    # Store phase-specific outputs
    new_state = dict(state)
    new_state["train_attempts"] = new_attempt
    new_state["train_valid"] = False

    if phase == "explore":
        # Save exploration results path for validator
        metrics_path = stage_dir / "exploration_metrics.json"
        print("METRICS PATH", metrics_path, metrics_path.exists())
        new_state["exploration_metrics_path"] = metrics_path if metrics_path.exists() else None
    else:
        # Tuning phase: model path is known
        model_path = stage_dir / "model.joblib"
        new_state["model_path"] = model_path if model_path.exists() else None

    return new_state


def run_train_validator(state: PipelineState) -> PipelineState:
    logger.info("Train validator started (phase: %s)", state.get("train_phase", "explore"))

    phase = state.get("train_phase", "explore")
    valid = False
    feedback = []
    new_state = dict(state)

    if phase == "explore":
        # Read exploration metrics
        metrics_path = state.get("exploration_metrics_path")
        print("GOT METRICS PATH", metrics_path)
        if metrics_path and Path(metrics_path).exists():
            import json
            with open(metrics_path) as f:
                metrics = json.load(f)

            # Determine best model based on R² (higher better) or MSE (lower better)
            best_model = None
            best_score = float('inf')
            for model, scores in metrics.items():
                mse = scores.get("mse", 1e9)
                if  mse < best_score:
                    best_score = mse
                    best_model = model

            if best_model:
                new_state["selected_model"] = best_model
                new_state["train_phase"] = "tune"      # switch to tuning
                feedback.append(f"Exploration completed. Best model: {best_model} (R²={best_score:.4f}). Moving to tuning phase.")
                valid = False                           # ❗ Do not exit the train loop yet
            else:
                # No best model – keep phase as "explore" and retry (with feedback)
                feedback.append("No model metrics found. Retrying exploration.")
                valid = False
        else:
            feedback.append("Exploration metrics file missing. Training code may have failed.")
            valid = False

    else:  # tune
        # Check if tuning succeeded
        model_path = state.get("model_path")
        last = state["train_report"].last_attempt

        if last.get("returncode", 0) != 0:
            feedback.append("Tuning code execution failed.")
            if last.get("stderr"):
                feedback.append(f"Stderr:\n{last['stderr'][-1000:]}")
        if last.get("error"):
            feedback.append(f"Training error: {last['error']}")
        if not model_path or not Path(model_path).exists():
            feedback.append("Final model artifact not created.")
        else:
            valid = True
            feedback.append(f"Tuning successful. Model saved to {model_path}.")

        # If tuning still failed after max attempts, we might still move to evaluation (but with error)
        if not valid and state["train_attempts"] >= state["train_max_attempts"]:
            feedback.append("Max attempts reached. Proceeding with possibly missing model.")
            valid = True   # to break the loop

    feedback_text = "\n".join(feedback) if feedback else "Training looks good."
    state["train_report"].log_validation(valid, feedback_text)

    logger.info("Train validation completed. Phase: %s, Valid: %s", phase, valid)
    return {
        **new_state,
        "train_feedback": feedback_text,
        "train_valid": valid,
    }
