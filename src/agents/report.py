"""Report AGENT"""

import json
import logging
from pathlib import Path

from src.state import PipelineState
from src.utils.io_utils import read_json
from src.utils.llm_utils import invoke_llm
from src.utils.metrics_utils import build_benchmark_summary

logger = logging.getLogger(__name__)

REPORT_PROMPT_TEMPLATE = """You are a data science report writer. Generate a clean, professional benchmark report in Markdown.

CRITICAL RULE — FAILED STAGES:
In the JSON below, each agent has `final_valid`. Check it FIRST before writing each section.
If `final_valid` is false or null for a stage, the ENTIRE section for that stage must be ONLY this one line:
> ⚠️ `<stage_name>` failed after <N> attempts and was skipped.
Do NOT invent or hallucinate data for failed stages. Do NOT write bullet points, tables, or any other content for them.

## INPUT DATA:

Pipeline metrics (JSON):
{summary_json}

EDA output (empty if EDA failed):
{eda_output}

Feature engineering summary (empty if FE failed):
{feature_summary}

Tuning results (empty if tune failed):
{tune_output}

## REPORT STRUCTURE (follow exactly, do not add or repeat sections):

# ML Pipeline Benchmark Report

## 1. Data Overview
If `eda` final_valid is false/null → one-liner failure message only.
Otherwise:
- Dataset size (rows × columns for train and test).
- Task type (regression / binary classification / multiclass).
- Target variable characteristics (from EDA output above).
- Key observations: missing values, class balance, notable patterns.

## 2. Feature Engineering
If `feature_engineering` final_valid is false/null → one-liner failure message only.
Otherwise:
- Number of original features → number of final features.
- Summary of transformations applied.
- Features dropped and why.

## 3. Model Comparison
If `train` final_valid is false/null → one-liner failure message only.
Otherwise create a markdown table with ALL models sorted by primary metric (best first):
For regression: columns Rank, Model, R² (mean ± std), MSE (mean ± std), MAE (mean ± std).
For classification: columns Rank, Model, F1 (mean ± std), Accuracy (mean ± std), ROC-AUC (mean ± std).
Mark the best model with **bold**.

## 4. Hyperparameter Tuning
If `tune` final_valid is false/null → one-liner failure message only.
Otherwise:
- Which model was selected for tuning.
- Best hyperparameters.
- Best score after tuning.
- Number of Optuna trials.
- Whether ensemble beat single model (if applicable).

## 5. Pipeline Execution Summary
ALWAYS include this section regardless of failures.
Table with columns: Agent, Attempts, Status, Duration (s), Tokens In, Tokens Out.
Status column: ✅ if final_valid is true, ❌ if false/null.
Add a totals row at the bottom.

## FORMATTING RULES:
- Use aligned markdown tables (not bullet lists for tabular data).
- Round metrics to 4 decimal places, time to 1 decimal.
- The report must have EXACTLY 5 sections numbered 1-5. No duplicates, no extra sections.
- Keep under 200 lines.
"""


def run_report_agent(state: PipelineState) -> PipelineState:
    logger.info("Report node started")

    reports_dir = state["run_dir"] / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Build benchmark_summary.json from all stage reports
    stage_reports = {
        "eda": state["eda_report"],
        "feature_engineering": state["fe_report"],
        "train": state["train_report"],
        "tune": state["tune_report"],
    }

    model_metrics = {}
    exploration_metrics_path = state["run_dir"] / "train" / "exploration_metrics.json"
    if exploration_metrics_path.exists():
        model_metrics = read_json(exploration_metrics_path)

    summary = build_benchmark_summary(
        output_path=reports_dir / "benchmark_summary.json",
        stage_reports=stage_reports,
        model_metrics=model_metrics,
    )

    # Collect context for LLM
    eda_output = ""
    eda_output_path = state.get("eda_output_path")
    if eda_output_path and Path(eda_output_path).exists():
        eda_output = Path(eda_output_path).read_text(encoding="utf-8")[:1500]

    feature_summary = ""
    feature_summary_path = state.get("feature_summary_path")
    if feature_summary_path and Path(feature_summary_path).exists():
        feature_summary = Path(feature_summary_path).read_text(encoding="utf-8")[:1500]

    tune_output = ""
    tune_last = state["tune_report"].last_attempt
    if tune_last.get("stdout"):
        tune_output = tune_last["stdout"][-1500:]

    prompt = REPORT_PROMPT_TEMPLATE.format(
        summary_json=json.dumps(summary, indent=2),
        eda_output=eda_output,
        feature_summary=feature_summary,
        tune_output=tune_output,
    )

    result = invoke_llm(prompt)
    report_path = reports_dir / "benchmark_report.md"
    report_path.write_text(result["text"], encoding="utf-8")
    logger.info("Benchmark report saved to %s", report_path)

    return state
