"""Report AGENT"""

import json
import logging
from pathlib import Path

from src.state import PipelineState
from src.utils.llm_utils import invoke_llm
from src.utils.metrics_utils import build_benchmark_summary

logger = logging.getLogger(__name__)

REPORT_PROMPT_TEMPLATE = """You are an ML pipeline analyst. Generate a benchmark report in Markdown.

Pipeline metrics:
{summary_json}

EDA output (excerpt):
{eda_output}

Feature engineering summary (excerpt):
{feature_summary}

Tuning results (excerpt):
{tune_output}

Write a Markdown report with:
## Pipeline Summary
- Total duration, total tokens used

## Agent Performance
- Table: Agent | Attempts | Duration (sec) | Tokens In | Tokens Out

## Model Exploration Results
- Table of all models and their metrics from the exploration phase

## Tuning Results
- Selected model, best hyperparameters, final validation score

## Observations
- Key insights about the pipeline run

Keep it concise and factual.
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
        model_metrics = json.loads(exploration_metrics_path.read_text(encoding="utf-8"))

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
