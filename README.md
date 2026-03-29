# <center>Workflow Agents for Tabular AutoFit & Analysis</center>

<p align="center">
  <a href="#about">About</a> •
  <a href="#pipeline-architecture">Pipeline Architecture</a> •
  <a href="#retrieval-augmented-generation-rag">RAG</a> •
  <a href="#project-structure">Project Structure</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#monitoring">Monitoring</a> •
  <a href="#license">License</a>
</p>

> [!NOTE]
> This repository is a student project for the 2026 HSE/MTS LLM-based Intelligent Agent Systems course.

## About

WATAFA is a multi-agent pipeline that solves Kaggle tabular competitions end-to-end without manual intervention. It supports regression, binary classification, and multiclass classification — the problem type is detected automatically from the data.

The system is built on LangGraph. Each pipeline stage is a separate agent that prompts an LLM to generate Python code, runs it in a subprocess, checks the result through a validator, and retries with feedback if something goes wrong. Relevant context from public Kaggle notebooks can be injected into prompts via a FAISS-based RAG module.

Everything is automated: data download from Kaggle, EDA, feature engineering, model training and benchmarking, hyperparameter tuning, and submission of predictions.


## Pipeline Architecture

<img width="2570" height="531" alt="flow" src="https://github.com/user-attachments/assets/925a3c47-90ce-4f1e-a2b8-1882527b3c8b" />

### What each node does

| Node | Responsibility | How it works | After success |
|------|----------------|--------------|---------------|
| `eda` | Load data, detect task type, compute statistics, identify missing values and outliers | Generates and executes Python code, saves stdout to `eda_output.txt` | `eda_validator` |
| `eda_validator` | Validate EDA output quality and completeness using an LLM as a judge | Reads `eda_output.txt`, asks LLM to evaluate, returns `valid` + `feedback` | valid: Feature Engineering; invalid: retry EDA |
| `feature_engineering` | Create new features: date parts, group aggregates, frequency encodings, ratios | Generates code that reads train/test, applies transformations, saves processed CSVs | `fe_validator` |
| `fe_validator` | Verify processed files exist, are non-empty, contain only numeric columns, and no traceback occurred | Checks file existence, size, dtypes, and return code | valid: Train; invalid: retry FE |
| `train` | Benchmark multiple model families (RandomForest, XGBoost, LightGBM, CatBoost, etc.) with 5-fold CV | Generates code that runs cross-validation and saves `exploration_metrics.json` with mean +/- std per model | `train_validator` |
| `train_validator` | Ensure exploration metrics were produced | Validates that `exploration_metrics.json` exists and contains data | valid: Tune; invalid: retry Train or early stop |
| `tune` | Select best model from exploration, run Optuna HPO (40 trials, hold-out split), attempt ensemble (Voting + Stacking from top-3) | Generates code that optimises hyperparameters and saves final `model.joblib` bundle | `tune_validator` |
| `tune_validator` | Verify the model bundle was created and execution succeeded | Checks `model.joblib` exists and return code is zero | valid: Submission; invalid: retry Tune or early stop |
| `submission` | Build submission CSV using the tuned model and send it to Kaggle | Loads model bundle, applies to test data, saves CSV, calls Kaggle API | `report` |
| `report` | Generate a final Markdown benchmark report summarising all stages | Reads all stage reports and metrics, uses LLM to format a structured report | END |

### Automation

- Data is downloaded from Kaggle automatically on pipeline start via the Kaggle API
- The target column is auto-detected as the column present in `train.csv` but absent in `test.csv`
- After tuning, predictions are submitted to Kaggle without any manual step

### Validation & Retry

Each agent node has a corresponding validator connected through conditional edges in the graph:

- When validation fails, the validator collects feedback (stderr, missing files, dtype errors, etc.) and passes it back to the agent. The agent regenerates code using this feedback on the next attempt.
- The EDA validator uses an LLM to judge output quality and completeness. All other validators (FE, Train, Tune) use heuristic checks: return code, file existence, content validation.
- Each stage allows up to 3 attempts (configurable in `state.py`).
- For critical stages (train, tune), exhausting all attempts triggers an early stop — the pipeline jumps directly to `report` instead of crashing, and the failed stage is marked in the benchmark report.

### Security & Guardrails

Two types of input are validated before the pipeline proceeds:

1. Downloaded CSV files from Kaggle are scanned for formula injection (`=`, `+`, `-`, `@` cell prefixes) and prompt injection patterns.
2. All LLM-generated code is analysed via AST before execution. Dangerous imports (e.g. `subprocess`, `socket`), calls (e.g. `eval`, `exec`, `os.system`), and OS-level operations are blocked. Code that fails the check is never executed. Each subprocess also has a configurable timeout (30-1800 s).

## Retrieval-Augmented Generation (RAG)

The EDA and Feature Engineering agents can use context from public Kaggle notebooks and markdown docs placed in the `knowledge/` directory. Before generating code, the agent runs a semantic search over a FAISS index of this knowledge base and appends the top-3 most relevant excerpts to the LLM prompt.

The FAISS index is built once and cached to disk. A SHA-256 hash of the knowledge files is stored alongside the index — if any file is added, removed, or modified, the index is rebuilt automatically on the next run.

The pipeline works normally with an empty `knowledge/` directory — RAG is silently disabled.


## Monitoring

Each pipeline run creates a timestamped directory in `artifacts/`. It contains:

- `info.log` — full execution log with timestamps and log levels
- `reports/benchmark_report.md` — final pipeline report (model comparison, metrics, stage summaries, token usage per stage and total)
- Per-stage subdirectories (`eda/`, `feature_engineering/`, `train/`, `tune/`, `submission/`) with generated code attempts, outputs, and stage reports (including token counts, duration, attempt history)

Console output provides a structured overview of the pipeline progress with stage separators and timing.


## Project Structure

```text
watafa-agentic-ml/
├── data/                        # Kaggle competition data (auto-downloaded)
├── knowledge/                   # RAG knowledge base (.ipynb, .md files)
├── artifacts/                   # Timestamped pipeline run outputs
├── src/
│   ├── agents/
│   │   ├── eda.py
│   │   ├── feature_engineering.py
│   │   ├── train.py
│   │   ├── tune.py
│   │   ├── submission.py
│   │   └── report.py
│   ├── utils/
│   │   ├── code_utils.py        # Code extraction & subprocess execution
│   │   ├── guardrails.py        # AST code validation & CSV injection checks
│   │   ├── io_utils.py          # File I/O helpers
│   │   ├── kaggle_utils.py      # Kaggle API integration
│   │   ├── llm_utils.py         # LLM client configuration
│   │   ├── metrics_utils.py     # Stage metrics tracking
│   │   └── rag.py               # FAISS vector store
│   ├── logger/
│   │   ├── logger.py            # Logging setup
│   │   └── logger_config.json
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

   The pipeline uses the following models (strongly recommended to keep the defaults, other models may produce unexpected results):
   - `minimax/minimax-m2.7` (LLM, via OpenRouter) — cost-effective model that ranks #2 in coding on the OpenRouter leaderboard
   - `openai/text-embedding-3-small` (embeddings for RAG, via OpenRouter) — best price-to-quality ratio among OpenAI embedding models

4. Run the pipeline:

   ```bash
   python run.py
   ```

5. After the run completes, check the benchmark report:

   ```
   artifacts/<timestamp>/reports/benchmark_report.md
   ```


## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
