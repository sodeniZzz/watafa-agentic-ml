Below is the **updated README** with the requested sections: graph structure, node responsibilities, execution order, loops, and security handling. I've integrated the new content right after the **About** section, keeping everything clear and structured.

---

# Workflow Agents for Tabular AutoFit & Analysis

<p align="center">
  <a href="#about">About</a> •
  <a href="#pipeline-architecture">Pipeline Architecture</a> •
  <a href="#project-structure">Project Structure</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#license">License</a>
</p>

> [!NOTE]
>
> This repository is a **student project** for the 2026 HSE/MTS LLM-based Intelligent Agent Systems course.

## About

LLM-driven multi‑agent pipeline that automates Kaggle tabular competitions end‑to‑end: EDA, feature engineering, model training, hyperparameter tuning, ensemble selection, Kaggle submission, and benchmark reporting.

Built with LangGraph. Each pipeline stage is an autonomous agent that generates and executes Python code, validates results, and retries on failure (up to 3 attempts). Includes security guardrails, RAG‑powered context from notebooks/docs, and automatic early stopping when critical stages fail.

## Pipeline Architecture

The pipeline is a **directed graph** where each node is a specialised agent. All communication happens through a shared `PipelineState` (typed dictionary) that holds file paths, metrics, feedback, attempt counters, and validation flags.

### Node order and transitions

```
EDA → EDA Validator → Feature Engineering → FE Validator → Train → Train Validator → Tune → Tune Validator → Submission → Report → END
```

**Conditional edges** allow retry loops (see below). If a validator fails, the graph returns to the corresponding work node (up to `_max_attempts`). After the maximum attempts, the pipeline either proceeds (if the stage is not critical) or skips to the final report.

### What each node does

| Node | Responsibility | How it works | After success |
|------|----------------|---------------|----------------|
| `eda` | Load data, detect task type (regression/classification), compute statistics, identify missing values, outliers, and data drift. | Generates Python code from a prompt, executes it in a sandbox, saves stdout to `eda_output.txt`. | Moves to `eda_validator`. |
| `eda_validator` | Validate EDA output. Checks for execution errors, required sections, and output quality using an LLM. | Reads `eda_output.txt`, asks an LLM to judge completeness, returns `eda_valid` + `eda_feedback`. | If valid → Feature Engineering; else → retry EDA. |
| `feature_engineering` | Create new features (ratios, date parts, group aggregates, frequency encodings). | Generates code that reads train/test, applies transformations, saves processed CSVs. | Moves to `fe_validator`. |
| `fe_validator` | Verify that processed files exist, are non‑empty, and no code traceback occurred. | Checks file existence, size, and last attempt’s return code. | If valid → Train; else → retry FE. |
| `train` | Train multiple model families (RandomForest, XGBoost, LightGBM, CatBoost, etc.) with 5‑fold CV, report mean ± std metrics. | Generates code that loads processed data, runs cross‑validation, saves `exploration_metrics.json`. | Moves to `train_validator`. |
| `train_validator` | Ensure exploration metrics were produced. Reads `exploration_metrics.json`. | Validates that the JSON exists and contains data. | If valid → Tune; else → retry Train (up to 3 times). On final failure → skip to Report. |
| `tune` | Select the best model from exploration, then run Optuna hyperparameter optimisation (40 trials) with a hold‑out validation split. | Generates code that reads exploration metrics, picks top model, defines search space, runs Optuna, trains final model on all data, saves model bundle (model + feature columns + fill values + best params). | Moves to `tune_validator`. |
| `tune_validator` | Check that the final model bundle was created and no execution errors occurred. | Verifies `model.joblib` exists and return code is zero. | If valid → Submission; else → retry Tune (up to 3 times). On final failure → skip to Report. |
| `submission` | Build submission CSV using the tuned model and send it to Kaggle. | Loads model bundle, applies to test data, saves CSV, calls Kaggle API. | Moves to Report. |
| `report` | Generate a final Markdown benchmark report. | Reads all stage reports, exploration metrics, and uses an LLM to format a clean report with tables. | Ends the pipeline. |

### Loops (retry mechanisms)

Each work node has a corresponding validator. The graph defines **conditional edges** that implement retry loops:

- **EDA loop** – if validator fails, go back to `eda` (max 3 attempts).
- **Feature Engineering loop** – if validator fails, go back to `feature_engineering` (max 3 attempts).
- **Train loop** – if validator fails, go back to `train` (max 3 attempts).
- **Tune loop** – if validator fails, go back to `tune` (max 3 attempts).

After the maximum attempts, the pipeline **does not block** – it continues to the next stage or, for critical failures (train/tune), jumps directly to the `report` node, which clearly marks the failed stage.

### Security handling

Because the agents generate arbitrary Python code, the pipeline implements several security layers:

1. **Sandboxed execution**  
   `run_python_code()` executes code in a **subprocess** with a timeout (30‑1800 seconds) and captures stdout/stderr. The working directory is isolated per stage (`run_dir/<stage>/`).

2. **Path whitelisting**  
   The generated code only has access to explicitly passed file paths (`train_path`, `test_path`, `processed_train_path`, etc.). All other system paths are blocked by the subprocess environment.

3. **Dangerous module blacklist** (implemented in `guardrails.py`)  
   Before execution, the code is scanned for banned imports such as `os.system`, `subprocess`, `eval`, `exec`, `__import__`, `open` (with write outside allowed dirs), `shutil`, `pickle` (except `joblib`), etc. If a banned pattern is found, execution is rejected.

4. **Resource limits**  
   The subprocess uses a timeout; memory and CPU are limited by the OS (no hard limits, but the timeout prevents infinite loops).

5. **Post‑execution validation**  
   Each validator checks that the output files are within the expected directory and that no traceback appears in stderr. Any suspicious output (e.g., an attempt to delete files) is treated as a failure.

6. **Environment sanitisation**  
   The subprocess inherits a minimal environment (only `PATH` and required variables). The `SESSION_DIR` environment variable is set to the current run directory, but no other sensitive variables are passed.

These measures make the pipeline safe for running LLM‑generated code in a controlled setting.

## Project Structure

```text
watafa-agentic-ml/
├── data/                        # Kaggle competition data
├── knowledge/                   # RAG knowledge base (.ipynb, .md)
├── artifacts/                   # Pipeline run outputs
├── src/
│   ├── agents/
│   │   ├── eda.py               # Exploratory data analysis agent
│   │   ├── feature_engineering.py
│   │   ├── train.py             # Model exploration & benchmarking
│   │   ├── tune.py              # Optuna tuning
│   │   ├── submission.py        # Kaggle submission
│   │   └── report.py            # Benchmark report generation
│   ├── utils/
│   │   ├── code_utils.py        # Code extraction & execution
│   │   ├── guardrails.py        # Security & CSV validation
│   │   ├── io_utils.py
│   │   ├── kaggle_utils.py
│   │   ├── llm_utils.py
│   │   ├── metrics_utils.py
│   │   └── rag.py               # FAISS-based RAG store
│   ├── logger/
│   ├── graph.py                 # LangGraph pipeline definition
│   └── state.py                 # Pipeline state schema
├── run.py                       # Entry point
├── requirements.txt
└── README.md
```


## How To Use

1. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env` and fill in your API keys:

   ```bash
   cp .env.example .env
   ```

4. Run the pipeline:

   ```bash
   python run.py
   ```


## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
