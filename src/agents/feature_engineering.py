"""FEATURE ENGINEERING AGENT"""

import logging
import time
from pathlib import Path

from src.state import PipelineState
from src.utils.code_utils import (
    extract_python_code,
    run_python_code,
)
from src.utils.llm_utils import invoke_llm
from src.utils.rag import retrieve_context

logger = logging.getLogger(__name__)


FEATURE_ENGINEERING_PROMPT_TEMPLATE = """You are a senior feature engineering expert. Based on the EDA report below, write Python code to create high-quality features for train and test datasets. The goal is strong generalization on unseen test data.

Target column name: {target_column}

Data files:
- Train: {train_path}
- Test: {test_path}

Save processed datasets to:
- Train: {processed_train_path}
- Test: {processed_test_path}

## CRITICAL RULES (violating any of these makes the output invalid):
1. The "{target_column}" column MUST be preserved in processed train. NEVER drop it.
2. Do NOT use "{target_column}" as an input feature or for feature construction (no target leakage).
3. Fit/compute ALL statistics, encoders, and mappings on TRAIN ONLY, then apply to both train and test.
4. Train and test must have IDENTICAL columns (except "{target_column}" which is only in train).
5. The output DataFrames must contain ONLY numeric columns (int/float). No object/string/category dtypes.

## FEATURE ENGINEERING STEPS (apply in this order):

### Step 1: Setup
- Load train and test with pandas.
- Separate target: `target = train["{target_column}"]`, then drop it from train temporarily.
- Identify column types: numeric_cols, categorical_cols (object dtype), date_cols (contains "dt" or "date" in name, or parseable as dates).
- Drop ID-like columns (unique values == number of rows) and constant columns (1 unique value).

### Step 2: Date features
- For each date column, parse with pd.to_datetime(errors="coerce").
- Extract: year, month, day, day_of_week, is_weekend (0/1).
- Compute days_since = (max_date_in_train - date).dt.days.
- Drop the original date column after extraction.

### Step 3: Categorical encoding
- For each categorical column:
  - If nunique <= 20: use pd.get_dummies (one-hot encoding). Align train and test columns after encoding.
  - If nunique > 20: use frequency encoding (value_counts normalized, computed on train, mapped to both).
- For any new categories in test not seen in train, fill with 0 (one-hot) or 0.0 (frequency).

### Step 4: Numeric features
- For pairs of related numeric columns, create:
  - Ratios (A / (B + 1)) — add 1 to avoid division by zero.
  - Products (A * B) for columns that logically interact.
- Apply np.log1p() to highly skewed numeric columns (skewness > 2).
- Create binned versions (pd.qcut with 5-10 bins, labels=False) for continuous columns with high cardinality.

### Step 5: Group aggregates
- For each categorical column (before encoding) with 2 < nunique <= 50:
  - Compute mean and median of top numeric columns, grouped by that category (on train only).
  - Map these aggregates to both train and test.
  - Name pattern: {{numeric_col}}_mean_by_{{cat_col}}.

### Step 6: Missing values
- For columns with >5% missing: create binary indicator column {{col}}_missing (1 if NaN, 0 otherwise).
- Fill numeric NaNs with median (computed on train).
- Fill categorical NaNs with mode (computed on train) before encoding.

### Step 7: Final cleanup
- Re-attach target to train: train["{target_column}"] = target.
- Drop any remaining non-numeric columns.
- Ensure train and test have the same columns (except "{target_column}").
- Print a detailed summary to stdout:
  - Final list of ALL column names in the processed dataset.
  - For each NEW feature: name, how it was computed (e.g. "ratio of sum to min_days", "frequency encoding of location_cluster").
  - Which columns were dropped and why (ID, constant, original date, etc.).
  - Final shapes of train and test.

## IMPORTANT NOTES:
- Use only pandas, numpy, sklearn. Do not use feature-engine or other specialized libraries.
- All transformations must be deterministic (set random_state where applicable).
- Handle edge cases: empty categories, all-NaN columns, zero-variance columns.

EDA report:
{eda_summary}
"""


def should_continue_after_fe_validation(state: PipelineState) -> str:
    if state.get("fe_valid", False) or state.get("fe_attempts", 0) >= state.get(
        "fe_max_attempts", 2
    ):
        return "train"
    else:
        return "feature_engineering"


def _generate_feature_eng_code(state: PipelineState, feedback: str = None):
    eda_output_path = state.get("eda_output_path")
    eda_summary = "EDA report not available."
    if eda_output_path and Path(eda_output_path).exists():
        eda_summary = Path(eda_output_path).read_text(encoding="utf-8")[:2000]

    stage_dir = state["run_dir"] / "feature_engineering"
    prompt = FEATURE_ENGINEERING_PROMPT_TEMPLATE.format(
        train_path=state["train_path"],
        test_path=state["test_path"],
        target_column=state["target_column"],
        processed_train_path=stage_dir / "processed_train.csv",
        processed_test_path=stage_dir / "processed_test.csv",
        eda_summary=eda_summary,
    )

    rag_context = retrieve_context(f"feature engineering {eda_summary[:200]}")
    if rag_context:
        prompt += f"\n\nRelevant reference material:\n{rag_context}\n"

    if feedback:
        prompt += f"\nPrevious attempt had the following issues. Please fix them:\n{feedback}\n"

    prompt += "\nWrite only the Python code, no explanations. The code must be self-contained and ready to execute."
    llm_result = invoke_llm(prompt)
    return extract_python_code(llm_result["text"]), llm_result["tokens_in"], llm_result["tokens_out"]


def run_feature_eng_agent(state: PipelineState) -> PipelineState:
    logger.info("Feature engineering node started")
    current_attempt = state.get("fe_attempts", 0)
    new_attempt = current_attempt + 1
    logger.info(f"Feature engineering attempt {new_attempt}")

    start = time.time()
    feedback = state.get("fe_feedback")
    code, tokens_in, tokens_out = _generate_feature_eng_code(state, feedback)

    stage_dir = state["run_dir"] / "feature_engineering"
    stage_dir.mkdir(parents=True, exist_ok=True)
    code_path = stage_dir / f"code_attempt_{new_attempt}.py"
    code_path.write_text(code, encoding="utf-8")
    logger.info("Feature engineering code saved to %s", code_path)

    execution_result = run_python_code(code_path, work_dir=stage_dir, timeout=120)
    duration = time.time() - start

    # Save clean stdout for train agent
    summary_path = stage_dir / "feature_summary.txt"
    summary_path.write_text(execution_result["stdout"], encoding="utf-8")

    # Log to stage report
    state["fe_report"].log_attempt(
        attempt=new_attempt, duration_sec=duration,
        tokens_in=tokens_in, tokens_out=tokens_out,
        returncode=execution_result["returncode"],
        stdout=execution_result["stdout"],
        stderr=execution_result["stderr"],
        error=execution_result["error"],
    )

    logger.info("Feature engineering summary saved to %s", summary_path)

    processed_train = stage_dir / "processed_train.csv"
    processed_test = stage_dir / "processed_test.csv"

    new_state = dict(state)
    new_state.update(
        {
            "fe_attempts": new_attempt,
            "processed_train_path": (
                processed_train if processed_train.exists() else None
            ),
            "processed_test_path": processed_test if processed_test.exists() else None,
            "feature_summary_path": summary_path,
            "fe_valid": False,
        }
    )
    return new_state


def run_fe_validator(state: PipelineState) -> PipelineState:
    logger.info("Feature engineering validator started")

    processed_train = state.get("processed_train_path")
    processed_test = state.get("processed_test_path")

    valid = True
    feedback = []

    if not processed_train or not Path(processed_train).exists():
        valid = False
        feedback.append("Processed train file not found.")
    elif Path(processed_train).stat().st_size == 0:
        valid = False
        feedback.append("Processed train file is empty.")

    if not processed_test or not Path(processed_test).exists():
        valid = False
        feedback.append("Processed test file not found.")
    elif Path(processed_test).stat().st_size == 0:
        valid = False
        feedback.append("Processed test file is empty.")

    # Check for errors via stage report
    last = state["fe_report"].last_attempt
    if last.get("returncode", 0) != 0:
        valid = False
        feedback.append("Code execution failed (non-zero return code).")
        if last.get("stderr"):
            feedback.append(f"Stderr:\n{last['stderr'][-1000:]}")
    elif "Traceback" in last.get("stderr", ""):
        valid = False
        feedback.append("Code execution produced a traceback error.")
        feedback.append(f"Stderr:\n{last['stderr'][-1000:]}")

    feedback_text = (
        "\n".join(feedback) if feedback else "Feature engineering looks good."
    )

    new_state = dict(state)
    new_state["fe_valid"] = valid
    new_state["fe_feedback"] = feedback_text
    state["fe_report"].log_validation(valid, feedback_text)

    logger.info(f"Feature engineering validation completed. Valid: {valid}")
    return new_state
