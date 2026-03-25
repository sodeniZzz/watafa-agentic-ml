import json
import logging
from pathlib import Path

from langgraph.graph import END, StateGraph

from src.state import PipelineState

from src.agents.eda import (
    run_eda_agent,
    run_eda_validator,
    should_continue_after_eda_validation,
)
from src.agents.feature_engineering import (
    run_feature_eng_agent,
    run_fe_validator,
    should_continue_after_fe_validation,
)
from src.agents.train import (
    run_train_agent,
    run_train_validator,
    should_continue_after_train_validation,
)
from src.agents.eval import (
    run_eval_agent,
    run_eval_validator,
    should_continue_after_eval_validation,
)
from src.agents.submission import run_submission_agent

logger = logging.getLogger(__name__)


def run_report_tool(state: PipelineState) -> PipelineState:
    logger.info("Report node started")

    eda_report = Path(state["eda_report_path"]).read_text(encoding="utf-8")
    metrics = state.get("metrics", {})
    model_path = state.get("model_path", "N/A")
    submission_path = state.get("submission_path", "N/A")
    final_report = "\n\n".join(
        [
            "Final Pipeline Report",
            f"EDA report path: {state['eda_report_path']}",
            eda_report,
            f"Validation metrics: {json.dumps(metrics, ensure_ascii=False)}",
            f"Model path: {model_path}",
            f"Submission path: {submission_path}",
        ]
    )

    report_path = state["run_dir"] / "reports" / "final_report.txt"
    report_path.write_text(final_report, encoding="utf-8")
    logger.info("Final report saved to %s", report_path)
    return state


def build_graph():
    graph_builder = StateGraph(PipelineState)

    graph_builder.add_node("eda", run_eda_agent)
    graph_builder.add_node("eda_validator", run_eda_validator)
    graph_builder.add_node("feature_engineering", run_feature_eng_agent)
    graph_builder.add_node("feature_engineering_validator", run_fe_validator)
    graph_builder.add_node("train", run_train_agent)
    graph_builder.add_node("train_validator", run_train_validator)
    graph_builder.add_node("evaluation", run_eval_agent)
    graph_builder.add_node("eval_validator", run_eval_validator)
    graph_builder.add_node("submission", run_submission_agent)
    graph_builder.add_node("report", run_report_tool)

    graph_builder.set_entry_point("eda")
    graph_builder.add_edge("eda", "eda_validator")
    graph_builder.add_conditional_edges(
        "eda_validator",
        should_continue_after_eda_validation,
        {"feature_engineering": "feature_engineering", "eda": "eda"},
    )
    graph_builder.add_edge("feature_engineering", "feature_engineering_validator")
    graph_builder.add_conditional_edges(
        "feature_engineering_validator",
        should_continue_after_fe_validation,
        {"train": "train", "feature_engineering": "feature_engineering"},
    )
    graph_builder.add_edge("train", "train_validator")
    graph_builder.add_conditional_edges(
        "train_validator",
        should_continue_after_train_validation,
        {
            "evaluation": "evaluation",
            "train": "train",
        },
    )
    graph_builder.add_edge("evaluation", "eval_validator")
    graph_builder.add_conditional_edges(
        "eval_validator",
        should_continue_after_eval_validation,
        {
            "submission": "submission",
            "evaluation": "evaluation",
        },
    )
    graph_builder.add_edge("submission", "report")
    graph_builder.add_edge("report", END)

    return graph_builder.compile()


def run_graph(state: PipelineState) -> PipelineState:
    logger.info("Graph execution started")
    final_state = build_graph().invoke(state)
    logger.info("Graph execution finished")
    return final_state
