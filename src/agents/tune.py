"""Tune AGENT — select best model from exploration and tune with Optuna."""

import json
import logging
import time
from pathlib import Path

from src.state import PipelineState
from src.utils.code_utils import extract_python_code, run_python_code
from src.utils.llm_utils import invoke_llm


logger = logging.getLogger(__name__)


TUNE_PROMPT_TEMPLATE = """You are a senior ML engineer specializing in hyperparameter optimization. Write Python code to tune the best model from the exploration phase using Optuna.

Input:
- Processed train dataset: {train_path}
- Target column: {target_column}
- Exploration metrics file: {metrics_path} (JSON with model benchmarks)
- Save final model to: {model_path}
- Previous feedback (if any): {feedback}

## CRITICAL RULES:
1. **NO additional feature engineering.** Data is already processed. Use only numeric columns. Drop object dtype.
2. Fill missing values with median.
3. Suppress all warnings: `warnings.filterwarnings("ignore")` and `optuna.logging.set_verbosity(optuna.logging.WARNING)`.
4. NEVER tune 'max_features' parameter.

## STEP-BY-STEP PROCEDURE:

### Step 1: Select Best Model
- Load metrics from {metrics_path}.
- For regression: select model with highest r2_mean (or lowest mse_mean).
- For classification: select model with highest f1_mean (or accuracy_mean).
- Print which model was selected and why.

### Step 2: Define Search Space
Define a comprehensive search space for the selected model. Examples:

**For XGBRegressor/XGBClassifier:**
- n_estimators: int [200, 1500]
- max_depth: int [3, 12]
- learning_rate: float log [0.005, 0.3]
- subsample: float [0.6, 1.0]
- colsample_bytree: float [0.5, 1.0]
- min_child_weight: int [1, 20]
- reg_alpha: float log [1e-8, 10]
- reg_lambda: float log [1e-8, 10]
- gamma: float log [1e-8, 5]

**For LGBMRegressor/LGBMClassifier:**
- n_estimators: int [200, 1500]
- max_depth: int [3, 12] (or -1)
- learning_rate: float log [0.005, 0.3]
- num_leaves: int [15, 127]
- min_child_samples: int [5, 100]
- subsample: float [0.6, 1.0]
- colsample_bytree: float [0.5, 1.0]
- reg_alpha: float log [1e-8, 10]
- reg_lambda: float log [1e-8, 10]

**For RandomForest/ExtraTrees (Regressor or Classifier):**
- n_estimators: int [200, 1000]
- max_depth: int [5, 30] or None
- min_samples_split: int [2, 20]
- min_samples_leaf: int [1, 10]

**For CatBoost (Regressor or Classifier):**
- iterations: int [200, 1500]
- depth: int [3, 10]
- learning_rate: float log [0.005, 0.3]
- l2_leaf_reg: float log [1e-2, 10]
- bagging_temperature: float [0, 1]
- random_strength: float [1e-2, 10]

**For GradientBoosting (Regressor or Classifier):**
- n_estimators: int [200, 1000]
- max_depth: int [3, 10]
- learning_rate: float log [0.005, 0.3]
- min_samples_split: int [2, 20]
- min_samples_leaf: int [1, 10]
- subsample: float [0.6, 1.0]

### Step 3: Optuna Optimization
- Use `optuna.create_study(direction=...)` with appropriate direction.
- Use `MedianPruner(n_startup_trials=5, n_warmup_steps=0)` to prune bad trials early.
- **Objective function uses a single holdout split** (test_size=0.2, random_state=42):
  - Train on train split, evaluate on validation split.
  - Return the validation metric.
- For boosting models (XGBoost, LightGBM, CatBoost): use early stopping (early_stopping_rounds=30, eval on validation split).
- Run **n_trials=40**.
- Set random seed: `optuna.create_study(sampler=optuna.samplers.TPESampler(seed=42), ...)`.

### Step 4: Train Final Model
- Get best parameters from study.best_params.
- Train final model on **ALL data** (no split) with best parameters.
- For boosting models: use the same n_estimators from best trial (not early-stopped count).

### Step 5: Save Model Bundle
Save with joblib to {model_path}:
```python
{{
    "model": final_model,
    "feature_columns": list(X.columns),
    "fill_values": dict(X.median()),
    "best_params": study.best_params,
}}
```
**IMPORTANT:** fill_values must be a **dict** (not list, not Series).

### Step 6: Print Summary
Print to stdout:
- Selected model name.
- Best validation score from best trial.
- Best hyperparameters.
- Number of trials completed.

EDA report excerpt:
{eda_report}

Feature engineering report excerpt:
{feature_report}
"""


def should_continue_after_tune_validation(state: PipelineState) -> str:
    if state["tune_valid"]:
        return "submission"
    if state["tune_attempts"] >= state["tune_max_attempts"]:
        logger.warning("Tune failed after %s attempts, early stop → report", state["tune_attempts"])
        return "report"
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
    new_attempt = state["tune_attempts"] + 1
    logger.info("Tune attempt %s", new_attempt)

    start = time.time()
    code, tokens_in, tokens_out = _generate_tune_code(state, state["tune_feedback"])

    stage_dir = state["run_dir"] / "tune"
    stage_dir.mkdir(parents=True, exist_ok=True)
    code_path = stage_dir / f"code_attempt_{new_attempt}.py"
    code_path.write_text(code, encoding="utf-8")
    logger.info("Tune code saved → %s", code_path)

    execution_result = run_python_code(code_path, work_dir=stage_dir, timeout=1200)
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
    logger.info("Tune validation started")

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
        feedback.append("Max attempts reached. Skipping to report.")

    feedback_text = "\n".join(feedback) if feedback else "Tuning looks good."
    state["tune_report"].log_validation(valid, feedback_text)

    logger.info("Tune validation: %s", "valid" if valid else "invalid")
    return {
        **state,
        "tune_feedback": feedback_text,
        "tune_valid": valid,
    }
