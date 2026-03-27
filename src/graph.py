import logging

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
from src.agents.tune import (
    run_tune_agent,
    run_tune_validator,
    should_continue_after_tune_validation,
)
from src.agents.submission import run_submission_agent
from src.agents.report import run_report_agent

logger = logging.getLogger(__name__)


def build_graph():
    graph_builder = StateGraph(PipelineState)

    graph_builder.add_node("eda", run_eda_agent)
    graph_builder.add_node("eda_validator", run_eda_validator)
    graph_builder.add_node("feature_engineering", run_feature_eng_agent)
    graph_builder.add_node("feature_engineering_validator", run_fe_validator)
    graph_builder.add_node("train", run_train_agent)
    graph_builder.add_node("train_validator", run_train_validator)
    graph_builder.add_node("tune", run_tune_agent)
    graph_builder.add_node("tune_validator", run_tune_validator)
    graph_builder.add_node("submission", run_submission_agent)
    graph_builder.add_node("report", run_report_agent)

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
        {"tune": "tune", "train": "train"},
    )
    graph_builder.add_edge("tune", "tune_validator")
    graph_builder.add_conditional_edges(
        "tune_validator",
        should_continue_after_tune_validation,
        {"submission": "submission", "tune": "tune"},
    )
    graph_builder.add_edge("submission", "report")
    graph_builder.add_edge("report", END)

    return graph_builder.compile()


def run_graph(state: PipelineState) -> PipelineState:
    logger.info("Graph execution started")
    final_state = build_graph().invoke(state)
    logger.info("Graph execution finished")
    return final_state
