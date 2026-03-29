# Workflow Agents for Tabular AutoFit & Analysis

<p align="center">
  <a href="#about">About</a> •
  <a href="#project-structure">Project Structure</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#license">License</a>
</p>

 > [!NOTE]
 >
 > This repository is a **student project** for the 2026 HSE/MTS LLM-based Intelligent Agent Systems course.


## About

LLM-driven multi-agent pipeline that automates Kaggle tabular competitions end-to-end: EDA, feature engineering, model training, hyperparameter tuning, ensemble selection, Kaggle submission, and benchmark reporting.

Built with LangGraph. Each pipeline stage is an autonomous agent that generates and executes Python code, validates results, and retries on failure (up to 3 attempts). Includes security guardrails, RAG-powered context from notebooks/docs, and automatic early stopping when critical stages fail.


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
│   │   ├── tune.py              # Optuna tuning + ensemble
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

3. Run the pipeline:

   ```bash
   python run.py
   ```


## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
