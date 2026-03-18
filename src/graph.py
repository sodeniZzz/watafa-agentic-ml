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


logger = logging.getLogger(__name__)


def _build_eda_context(train_df: pd.DataFrame, test_df: pd.DataFrame) -> str:
    missing_values = train_df.isna().sum().sort_values(ascending=False)
    numeric_summary = train_df.describe(include="all").transpose().head(20).to_string()

    context_parts = [
        f"Train shape: {train_df.shape}",
        f"Test shape: {test_df.shape}",
        f"Train columns: {list(train_df.columns)}",
        f"Test columns: {list(test_df.columns)}",
        "Train dtypes:",
        train_df.dtypes.to_string(),
        "Top missing values in train:",
        missing_values.head(20).to_string(),
        "Train summary statistics:",
        numeric_summary,
    ]
    return "\n\n".join(context_parts)


def run_eda_agent(state: PipelineState) -> PipelineState:
    logger.info("EDA node started")

    train_df = pd.read_csv(state["train_path"]).sample(frac=0.2, random_state=42)
    test_df = pd.read_csv(state["test_path"])
    eda_context = _build_eda_context(train_df, test_df)

    prompt = f"""
You are an ML engineer doing EDA for a tabular Kaggle competition.

Write a short EDA report in plain text.
Focus on:
- dataset size and structure
- missing values
- numeric features and target
- possible preprocessing concerns
- concise recommendations for training

Data summary:
{eda_context}
""".strip()

    eda_report = invoke_llm(prompt, temperature=0.0, max_tokens=1200)
    report_path = state["run_dir"] / "reports" / "eda_summary.txt"
    report_path.write_text(eda_report, encoding="utf-8")
    logger.info("EDA report saved to %s", report_path)

    return {**state, "eda_report_path": report_path}


def run_train_tool(state: PipelineState) -> PipelineState:
    logger.info("Train node started")

    train_df = pd.read_csv(state["train_path"])
    target_col = "target"
    feature_df = train_df.drop(columns=[target_col]).select_dtypes(include="number")
    fill_values = feature_df.median()
    feature_df = feature_df.fillna(fill_values)
    target = train_df[target_col]

    x_train, x_val, y_train, y_val = train_test_split(
        feature_df, target, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(x_train, y_train)

    model_bundle = {
        "model": model,
        "feature_columns": list(feature_df.columns),
        "fill_values": fill_values.to_dict(),
    }
    model_path = state["run_dir"] / "models" / "model.joblib"
    joblib.dump(model_bundle, model_path)

    val_predictions = model.predict(x_val)
    mse = float(mean_squared_error(y_val, val_predictions))
    logger.info("Model saved to %s", model_path)
    logger.info("Validation MSE: %.6f", mse)

    return {**state, "model_path": model_path, "metrics": {"mse": mse}}


def run_evaluation_tool(state: PipelineState) -> PipelineState:
    logger.info("Evaluation node started")

    metrics_path = state["run_dir"] / "reports" / "metrics.json"
    metrics_path.write_text(
        json.dumps(state["metrics"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Metrics saved to %s", metrics_path)
    return state


def run_submission_tool(state: PipelineState) -> PipelineState:
    logger.info("Submission node started")

    model_bundle = joblib.load(state["model_path"])
    model = model_bundle["model"]
    feature_columns = model_bundle["feature_columns"]
    fill_values = model_bundle["fill_values"]

    test_df = pd.read_csv(state["test_path"])
    sample_submission = pd.read_csv(state["sample_submission_path"])

    x_test = test_df[feature_columns].fillna(fill_values)
    predictions = model.predict(x_test)

    submission = sample_submission.copy()
    submission[submission.columns[-1]] = predictions

    submission_path = state["run_dir"] / "submission.csv"
    submission.to_csv(submission_path, index=False)
    logger.info("Submission saved to %s", submission_path)

    return {**state, "submission_path": submission_path}


def run_report_tool(state: PipelineState) -> PipelineState:
    logger.info("Report node started")

    eda_report = Path(state["eda_report_path"]).read_text(encoding="utf-8")
    final_report = "\n\n".join(
        [
            "Final Pipeline Report",
            f"EDA report path: {state['eda_report_path']}",
            eda_report,
            f"Validation metrics: {json.dumps(state['metrics'], ensure_ascii=False)}",
            f"Model path: {state['model_path']}",
            f"Submission path: {state['submission_path']}",
        ]
    )

    report_path = state["run_dir"] / "reports" / "final_report.txt"
    report_path.write_text(final_report, encoding="utf-8")
    logger.info("Final report saved to %s", report_path)
    return state


def build_graph():
    graph_builder = StateGraph(PipelineState)

    graph_builder.add_node("eda", run_eda_agent)
    graph_builder.add_node("train", run_train_tool)
    graph_builder.add_node("evaluation", run_evaluation_tool)
    graph_builder.add_node("submission", run_submission_tool)
    graph_builder.add_node("report", run_report_tool)

    graph_builder.set_entry_point("eda")
    graph_builder.add_edge("eda", "train")
    graph_builder.add_edge("train", "evaluation")
    graph_builder.add_edge("evaluation", "submission")
    graph_builder.add_edge("submission", "report")
    graph_builder.add_edge("report", END)

    return graph_builder.compile()


def run_graph(state: PipelineState) -> PipelineState:
    logger.info("Graph execution started")
    final_state = build_graph().invoke(state)
    logger.info("Graph execution finished")
    return final_state
