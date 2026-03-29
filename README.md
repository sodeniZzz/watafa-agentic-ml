
# <center>Workflow Agents for Tabular AutoFit & Analysis </center>

<p align="center">
  <a href="#about">About</a> •
  <a href="#pipeline-architecture">Pipeline Architecture</a> •
  <a href="#retrieval-augmented-generation-rag">Retrieval-Augmented Generation (RAG)</a> •
  <a href="#data-collection--submission">Data Collection & Submission</a> •
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

### Multi‑task & Multi‑model Support

The pipeline automatically detects the problem type from the EDA report and adapts its behaviour:

- **Regression** – uses MSE, MAE, R² metrics; trains models like RandomForest, XGBoost, LightGBM, CatBoost, Ridge, etc.
- **Binary classification** – uses Accuracy, F1, ROC‑AUC, Precision, Recall; includes LogisticRegression, SVC, and tree‑based classifiers.
- **Multiclass classification** – uses Macro/Weighted F1.

During the **train (exploration) phase**, all supported models are evaluated with **5‑fold cross‑validation** using default (sensible) hyperparameters. The results are ranked, and the best model (by primary metric) is automatically selected for the subsequent **tuning phase** (Optuna). This allows fair comparison between different architectures without manual intervention.

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

- **Sandboxed execution** – All generated code runs in an isolated subprocess with a timeout (30–1800s) and captured stdout/stderr.
- **Path whitelisting** – Code only accesses explicitly provided file paths; other system paths are blocked.
- **Dangerous module blacklist** – Imports like `os.system`, `subprocess`, `eval`, `exec`, `open` (outside allowed dirs) are rejected before execution.
- **Resource limits** – Timeout prevents infinite loops; no direct memory/CPU limits, but the subprocess environment is minimal.


## Retrieval-Augmented Generation (RAG)

### Knowledge Base Contents
The RAG module indexes **public Kaggle notebooks** from **similar competitions** (e.g., previous real estate prediction challenges, regression tasks with tabular data). Each notebook is converted to text, chunked, embedded, and stored in a **FAISS vector database** .

### When & How It Is Used
The RAG context is injected **only into the EDA and Feature Engineering prompts** . When the agent generates code for these stages, it first performs a semantic search over the knowledge base using the column names and task description as a query. The top‑3 most relevant notebook excerpts are appended to the prompt, providing the LLM with proven examples of similar data transformations, feature creation, and EDA patterns .

### Technical Implementation
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (384‑dim, fast & compact)
- **Vector Store**: FAISS index built offline from `knowledge/` directory (`.ipynb`, `.md` files)
- **Retrieval**: Cosine similarity search, top‑k = 3


## Data Collection & Submission

### Automatic Data Download
The pipeline automatically downloads competition data via the **Kaggle API** . Upon initialization, the agent uses the official `kaggle` Python package to authenticate (using `KAGGLE_USERNAME` and `KAGGLE_API_KEY` from `.env`) and downloads `train.csv`, `test.csv`, and `sample_submission.csv` directly from the competition page . No manual data preparation is required.

### Automatic Submission
After the tuned model generates predictions, the pipeline automatically submits the resulting `submission.csv` to Kaggle using `api.competition_submit()` . The submission message includes a timestamp and pipeline version for traceability. The CLI command `kaggle competitions submit -f submission.csv -m "..."` is executed in a subprocess, and the return code is logged. This enables fully unattended participation in Kaggle competitions .


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
