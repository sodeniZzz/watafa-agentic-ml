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

EDA report excerpt:
{eda_report}

Feature engineering report excerpt:
{feature_report}
"""

EXPLORE_PROMPT_TEMPLATE = """You are an ML training expert. Write Python code to train multiple models and compare their validation performance. The task type (regression or classification) will be determined from the EDA report.

Input:
- Processed train dataset: {train_path}
- Target column: {target_column}
- Task type: inferred from EDA report (regression, binary classification, or multiclass classification)

Requirements:
1. Load the dataset with pandas.
2. Use target column exactly named {target_column}.
3. **Handle categorical features**:
   - Identify columns with `object` dtype or low‑cardinality integer columns (nunique ≤ 20).
   - For those columns, apply **one‑hot encoding** using `pd.get_dummies(drop_first=True)` to avoid multicollinearity.
   - For high‑cardinality categorical columns (nunique > 20), **drop them** or use **frequency encoding** (replace each category with its count in the training set). Do not create hundreds of dummy columns.
   - If date columns exist (e.g., `last_dt`), convert them to datetime and extract features (year, month, day, dayofweek, etc.) if they haven't already been processed in feature engineering.
   - **Important**: Use the feature engineering report to know which new features already exist; you may skip encoding columns that are already numeric or have been already transformed.
4. Split data into train and validation (test_size=0.2, random_state=42).
5. Fill missing values using median (for numeric) or mode (for categorical) – apply **after** encoding so that dummy columns have no missing values.
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

TUNE_PROMPT_TEMPLATE = """You are an ML training expert. Write Python code to tune and train the best model identified from the exploration phase. The task type (regression or classification) is inferred from the EDA report.

Input:
- Processed train dataset: {train_path}
- Target column: {target_column}
- Exploration metrics file: {metrics_path} (JSON file with model performance)
- Previous feedback (if any): {feedback}

Requirements:
1. Load the dataset with pandas.
2. Use target column exactly named {target_column}.
3. **Handle categorical features** exactly as in the exploration step (see exploration requirements). Use the same encoding strategy to ensure consistency.
4. Split rows into train and validation sets with test_size=0.2, random_state=42.
5. Fill missing values using median (numeric) or mode (categorical) – after encoding.
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

def should_continue_after_train_validation(state: PipelineState) -> str:
    if state["train_valid"] or state["train_attempts"] >= state["train_max_attempts"]:
        return "evaluation"
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
    model_path = stage_dir / "model.joblib" 
    metrics_path = stage_dir / "exploration_metrics.json"

    phase = state.get("train_phase", "explore")

    if phase == "explore":
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
        return extract_python_code(llm_result["text"]), llm_result["tokens_in"], llm_result["tokens_out"], phase

    else:  # tune
        prompt = TUNE_PROMPT_TEMPLATE.format(
            train_path=train_path,
            target_column=state["target_column"],
            metrics_path=metrics_path,                 # pass the path to the exploration metrics
            model_path=model_path,
            feature_report=feature_report,
            eda_report=eda_report,
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
