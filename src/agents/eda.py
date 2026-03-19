"""EDA AGENT ---- PLAN -> CODE -> EXECUTE"""

import json
import logging
from pathlib import Path

import joblib
import pandas as pd
from langgraph.graph import END, StateGraph
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from src.state import PipelineState
from src.utils.llm_utils import invoke_llm
from src.utils.code_utils import (
    run_python_code,
)

logger = logging.getLogger(__name__)

def should_continue_after_eda_validation(state: PipelineState) -> str:
    if state.get("eda_valid", False) or state.get("eda_attempts", 0) >= state.get("eda_max_attempts", 2):
        return "feature_eng"   # go to feature engineering when EDA is done
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

    prompt = f"""You are a data analysis expert. Write Python code for Exploratory Data Analysis (EDA) for the training dataset.
The data is located in the file: {state['train_path']}
There is also a test file (you can use it for comparison): {state['test_path']}

The code should perform the following tasks:
- Load the data using pandas.
- Output basic information: shape, columns, dtypes, number of missing values, descriptive statistics for numeric columns.
- For numeric features, calculate and output: number of unique values, min/max, quantiles.
- For categorical features, output unique values (if there are fewer than 20).
- Output a brief textual summary (key observations) to stdout.

Column information (first 5 rows):
{columns_str}
"""
    if feedback:
        prompt += f"""
The previous EDA attempt had the following feedback. Please improve the code accordingly:
{feedback}
"""

    prompt += """
Write only the code, without any additional explanations. The code should be ready to execute as is.
"""
    response = invoke_llm(prompt)

    code = response.strip()
    if code.startswith("```python"):
        code = code.split("```python")[1]
    if code.endswith("```"):
        code = code.rsplit("```", 1)[0]
    code = code.strip()
    return code



def run_eda_validator(state: PipelineState) -> PipelineState:
    logger.info("EDA validator node started")

    report_path = state.get("eda_report_path")
    if not report_path or not Path(report_path).exists():
        logger.error("EDA report not found")
        feedback = "EDA report file not found. Please regenerate EDA code."
        valid = False
        validation_report_path = None
    else:
        report_content = Path(report_path).read_text(encoding="utf-8")

        # Use LLM to evaluate the EDA output
        prompt = f"""You are an expert data analyst validating an EDA report.

The EDA was supposed to:
- Load data with pandas
- Output basic info: shape, columns, dtypes, missing values, descriptive statistics for numeric columns
- For numeric features: unique values count, min/max, quantiles
- For categorical features: output unique values (if <20)
- Output a brief textual summary

Here is the actual EDA output (including stdout, stderr and return code):
{report_content}

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
        response = invoke_llm(prompt)
        try:
            import json
            validation = json.loads(response)
            valid = validation.get("valid", False)
            feedback = validation.get("feedback", "No feedback provided.")
        except Exception:
            logger.warning("Failed to parse validator LLM response, assuming valid")
            valid = True
            feedback = "Validator could not parse response, proceeding."

    # Save validation report
    attempt = state.get("eda_attempts", 0)   # current attempt (after EDA ran)
    validation_dir = state["run_dir"] / "reports"
    validation_dir.mkdir(parents=True, exist_ok=True)
    validation_report_path = validation_dir / f"eda_validation_attempt_{attempt}.txt"
    validation_report_path.write_text(f"VALID: {valid}\nFEEDBACK:\n{feedback}", encoding="utf-8")

    # Update state
    new_state = dict(state)
    new_state["eda_valid"] = valid
    new_state["eda_feedback"] = feedback
    reports = new_state.get("eda_validation_reports", [])
    reports.append(validation_report_path)
    new_state["eda_validation_reports"] = reports

    logger.info(f"EDA validation completed. Valid: {valid}")
    return new_state



def run_eda_agent(state: PipelineState) -> PipelineState:
    logger.info("EDA node started")

    current = state.get("eda_attempts", 0)
    new_attempt = current + 1
    logger.info(f"EDA attempt {new_attempt}")

    feedback = state.get("eda_feedback")
    code = _generate_eda_code(state, feedback)

    code_path = state["run_dir"] / "code" / f"eda_attempt_{new_attempt}.py"
    code_path.write_text(code, encoding="utf-8")

    execution_result = run_python_code(
        code,
        state["run_dir"] / "code",
    )

    report_content = f"""STDOUT:
{execution_result['stdout']}

STDERR:
{execution_result['stderr']}

Return code: {execution_result['returncode']}
"""
    report_path = state["run_dir"] / "reports" / f"eda_summary_attempt_{new_attempt}.txt"
    report_path.write_text(report_content, encoding="utf-8")
    logger.info("EDA report saved to %s", report_path)

    new_state = dict(state)
    new_state["eda_attempts"] = new_attempt
    new_state["eda_report_path"] = report_path
    return new_state
