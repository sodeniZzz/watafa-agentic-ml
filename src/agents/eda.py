"""EDA AGENT"""

import json
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


EDA_PROMPT_TEMPLATE = """You are a senior data scientist. Write Python code for a comprehensive Exploratory Data Analysis. Print ALL results to stdout in a structured format.

Data files:
- Train: {train_path}
- Test: {test_path}

Column information (first 5 rows):
{columns_str}

## The code must perform the following analysis and print results in this EXACT order:

### 1. Task Type Detection
- Analyze the "target" column to determine the task type. Print ONE of:
  - "TASK_TYPE: Regression" — if target is continuous (float or many unique int values)
  - "TASK_TYPE: Binary Classification" — if target has exactly 2 unique values
  - "TASK_TYPE: Multiclass Classification" — if target has 3-30 unique values
- Print target distribution: value_counts for classification, describe() for regression.
- Print skewness of target. If skewness > 1, note that log-transform may help.

### 2. Dataset Overview
- Print shapes of train and test.
- Print column names, dtypes, and non-null counts (df.info style).
- Print number and percentage of missing values per column, sorted descending.
- Flag columns with >50% missing.

### 3. Numeric Feature Analysis
- For each numeric column: print min, max, mean, median, std, skewness, number of unique values.
- Print top-10 features most correlated with target (by absolute Pearson correlation).
- Detect outliers using IQR method: for each numeric column, print count of values below Q1-1.5*IQR or above Q3+1.5*IQR.

### 4. Categorical Feature Analysis
- For each categorical (object dtype) column: print nunique, and value_counts if nunique <= 20.
- For high-cardinality categoricals (nunique > 20): print nunique and top-5 most frequent values.

### 5. Train vs Test Comparison
- For each numeric column: compare mean and std between train and test. Flag columns where the difference in mean exceeds 1 std (potential data drift).
- For each categorical column: check if test has categories not present in train.

### 6. Summary and Recommendations
Print a concise summary:
- Task type and target characteristics.
- Most important features (by correlation with target).
- Columns to consider dropping (high missing %, zero variance, ID-like).
- Potential issues: class imbalance, skewed distributions, data drift between train/test.
- Recommended feature engineering strategies based on the data.
"""

EDA_VALIDATOR_PROMPT_TEMPLATE = """You are an expert data analyst validating an EDA report.

The EDA was supposed to:
- Determine the type of 
- Load data with pandas
- Output basic info: shape, columns, dtypes, missing values, descriptive statistics for numeric columns
- For numeric features: unique values count, min/max, quantiles
- For categorical features: output unique values (if <20)
- Output a brief textual summary

Here is the actual EDA output:
{eda_output}

Please evaluate:
1. Did the code execute without errors? (Check for tracebacks or error messages in the output)
2. Does the output contain all the required information? If something is missing, list what's missing.
3. Are there any obvious issues or areas for improvement?

Provide your assessment in the following JSON format:
{{
    "valid": true/false,
    "feedback": "Detailed feedback on what's missing or what could be improved. If valid, you can say 'EDA looks good.'",
    "missing_elements": ["list", "of", "missing", "items"] (optional)
}}

Respond only with the JSON, no other text.
"""


def should_continue_after_eda_validation(state: PipelineState) -> str:
    if state.get("eda_valid", False) or state.get("eda_attempts", 0) >= state.get("eda_max_attempts", 2):
        return "feature_engineering"
    else:
        return "eda"


def _generate_eda_code(state, feedback=None):
    try:
        import pandas as pd
        train_sample = pd.read_csv(state["train_path"], nrows=5)
        columns_info = []
        for col in train_sample.columns:
            dtype = train_sample[col].dtype
            sample_vals = train_sample[col].tolist()[:3]
            columns_info.append(f"- {col}: {dtype}, примеры: {sample_vals}")
        columns_str = "\n".join(columns_info)
    except Exception as e:
        columns_str = f"Не удалось загрузить образец: {e}"

    prompt = EDA_PROMPT_TEMPLATE.format(
        train_path=state["train_path"],
        test_path=state["test_path"],
        columns_str=columns_str,
    )

    rag_context = retrieve_context(f"exploratory data analysis {columns_str[:200]}")
    if rag_context:
        prompt += f"\n\nRelevant reference material:\n{rag_context}\n"

    if feedback:
        prompt += f"""
The previous EDA attempt had the following feedback. Please improve the code accordingly:
{feedback}
"""

    prompt += """
Write only the code, without any additional explanations. The code should be ready to execute as is.
"""
    llm_result = invoke_llm(prompt)
    return extract_python_code(llm_result["text"]), llm_result["tokens_in"], llm_result["tokens_out"]


def run_eda_agent(state: PipelineState) -> PipelineState:
    logger.info("EDA node started")

    current = state.get("eda_attempts", 0)
    new_attempt = current + 1
    logger.info(f"EDA attempt {new_attempt}")

    start = time.time()
    feedback = state.get("eda_feedback")
    code, tokens_in, tokens_out = _generate_eda_code(state, feedback)

    stage_dir = state["run_dir"] / "eda"
    stage_dir.mkdir(parents=True, exist_ok=True)
    code_path = stage_dir / f"code_attempt_{new_attempt}.py"
    code_path.write_text(code, encoding="utf-8")

    execution_result = run_python_code(code_path, stage_dir)
    duration = time.time() - start

    # Save clean stdout for next agent
    output_path = stage_dir / "eda_output.txt"
    output_path.write_text(execution_result["stdout"], encoding="utf-8")

    # Log to stage report
    state["eda_report"].log_attempt(
        attempt=new_attempt, duration_sec=duration,
        tokens_in=tokens_in, tokens_out=tokens_out,
        returncode=execution_result["returncode"],
        stdout=execution_result["stdout"],
        stderr=execution_result["stderr"],
        error=execution_result["error"],
    )

    logger.info("EDA output saved to %s", output_path)

    new_state = dict(state)
    new_state["eda_attempts"] = new_attempt
    new_state["eda_output_path"] = output_path
    return new_state


def run_eda_validator(state: PipelineState) -> PipelineState:
    logger.info("EDA validator node started")

    output_path = state.get("eda_output_path")
    if not output_path or not Path(output_path).exists():
        logger.error("EDA output not found")
        new_state = dict(state)
        new_state["eda_valid"] = False
        new_state["eda_feedback"] = "EDA output file not found. Please regenerate EDA code."
        return new_state

    eda_output = Path(output_path).read_text(encoding="utf-8")
    # Also include stderr from last attempt for error checking
    last = state["eda_report"].last_attempt
    if last.get("stderr"):
        eda_output += f"\n\nSTDERR:\n{last['stderr']}"
    if last.get("returncode", 0) != 0:
        eda_output += f"\n\nReturn code: {last['returncode']}"

    prompt = EDA_VALIDATOR_PROMPT_TEMPLATE.format(eda_output=eda_output)
    response = invoke_llm(prompt)
    try:
        validation = json.loads(response["text"])
        valid = validation.get("valid", False)
        feedback = validation.get("feedback", "No feedback provided.")
    except Exception:
        logger.warning("Failed to parse validator LLM response, assuming valid")
        valid = True
        feedback = "Validator could not parse response, proceeding."

    new_state = dict(state)
    new_state["eda_valid"] = valid
    new_state["eda_feedback"] = feedback
    state["eda_report"].log_validation(valid, feedback)

    logger.info(f"EDA validation completed. Valid: {valid}")
    return new_state
