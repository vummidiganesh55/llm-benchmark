LLM Benchmarking Platform








A production-oriented platform for benchmarking, evaluating, comparing, and monitoring Large Language Models across quality, latency, throughput, reliability, cost, semantic similarity, and failure behavior.

📌 Description

LLM Benchmarking Platform is an experiment-driven evaluation system for comparing multiple LLM providers and local models using a consistent benchmark dataset and reproducible evaluation pipeline.

The platform combines:

Benchmark execution

Multiple LLM provider adapters

Exact and fuzzy evaluation

Semantic evaluation

LLM-as-a-Judge support

Retry and fallback handling

Failure analysis

Statistical analysis

Regression detection

Quality × latency × cost analysis

Experiment tracking

Concurrency benchmarking

JSON / CSV / HTML report export

Model leaderboard

FastAPI backend

Streamlit dashboard

Docker

Automated pytest CI with GitHub Actions

The project is designed to demonstrate LLM engineering, evaluation engineering, backend engineering, observability, testing, and production-oriented system design.

🎯 Problem Statement

LLM applications cannot be evaluated reliably using accuracy alone.

Different models can have different:

Accuracy

Response latency

P50/P95 latency

Token throughput

Error rates

Retry behavior

Fallback behavior

Cost

Semantic similarity

Category-level strengths and weaknesses

A practical evaluation platform therefore needs a repeatable benchmark pipeline that records these dimensions and makes model-to-model comparisons measurable.

💡 Solution

This project provides an end-to-end benchmarking workflow:

Dataset
   ↓
Benchmark Runner
   ↓
LLM Provider Layer
   ↓
Model Response
   ↓
Evaluation Engine
   ├── Exact Match
   ├── Fuzzy Match
   ├── Semantic Similarity
   └── LLM-as-a-Judge
   ↓
Metrics Engine
   ├── Quality
   ├── Latency
   ├── Throughput
   ├── Reliability
   └── Cost
   ↓
Experiment Storage
   ↓
Analysis
   ├── Comparison
   ├── Statistics
   ├── Failure Analysis
   ├── Regression Detection
   └── Quality × Latency × Cost
   ↓
Reports + Leaderboard + Dashboard

🚀 Features

1. Multi-Provider LLM Architecture

Provider adapters allow the benchmark engine to work with local and cloud-backed models.

Current provider/model integrations include:

Ollama

Google Gemini

Groq

OpenAI adapter

Anthropic adapter

DeepSeek adapter

Mistral adapter

Kimi adapter

xAI/Grok adapter

Cohere adapter

The project also supports local Ollama models such as:

qwen2.5:3b

qwen2.5-coder:3b

llama3.2:latest

mistral:latest

and cloud/provider-backed configurations such as:

gemini-3.6-flash

openai/gpt-oss-20b

Provider availability and pricing depend on the configured API keys and provider plans.

2. 50-Question Benchmark Dataset

The benchmark dataset contains 50 questions distributed across five categories:

Category

Questions

Math

10

Knowledge

10

Reasoning

10

Coding

10

General

10

Total

50

Dataset versioning is maintained through:

datasets/metadata.json
datasets/benchmark.json

The original benchmark dataset is preserved while an enhanced version is maintained separately.

3. Evaluation Metrics

Quality

Accuracy

Exact Match

Fuzzy Match

Average Fuzzy Similarity

Semantic Match

Average Semantic Similarity

LLM-as-a-Judge correctness

LLM-as-a-Judge relevance

LLM-as-a-Judge reasoning quality

LLM-as-a-Judge confidence

Performance

Average latency

P50 latency

P95 latency

Provider latency

Tokens per second

Input tokens

Output tokens

Total tokens

Reliability

Error rate

Retry rate

Fallback rate

Fallback success rate

Average attempts

Total retries

Total attempts

Cost

Input cost

Output cost

Total cost

Cost per question

🏗️ Architecture

flowchart TD
    UI[Streamlit Dashboard]
    API[FastAPI Backend]

    API --> RUN[Benchmark Runner]
    API --> REPORT[Experiment Reports]
    API --> REG[Regression Detection]
    API --> CONC[Concurrency Benchmark]
    API --> LB[Model Leaderboard]

    RUN --> REGISTRY[Model Registry]
    REGISTRY --> PROVIDERS[LLM Provider Layer]

    PROVIDERS --> OLLAMA[Ollama]
    PROVIDERS --> CLOUD[Cloud Providers]

    RUN --> EVAL[Evaluation Engine]
    EVAL --> EXACT[Exact Match]
    EVAL --> FUZZY[Fuzzy Match]
    EVAL --> SEMANTIC[Semantic Evaluation]
    EVAL --> JUDGE[LLM-as-a-Judge]

    RUN --> METRICS[Metrics Engine]
    METRICS --> QUALITY[Quality]
    METRICS --> PERFORMANCE[Latency / Throughput]
    METRICS --> RELIABILITY[Reliability]
    METRICS --> COST[Cost]

    METRICS --> STORAGE[Experiment Storage]

    STORAGE --> ANALYSIS[Analysis Layer]
    ANALYSIS --> FAILURE[Failure Analysis]
    ANALYSIS --> STATS[Statistical Runs]
    ANALYSIS --> REGRESSION[Regression Detection]
    ANALYSIS --> QLC[Quality × Latency × Cost]

    STORAGE --> EXPORT[JSON / CSV / HTML Reports]

🔄 Workflow

1. Select model
       ↓
2. Load benchmark dataset
       ↓
3. Validate dataset
       ↓
4. Execute benchmark questions
       ↓
5. Capture response + latency + token metrics
       ↓
6. Evaluate responses
       ↓
7. Calculate aggregate metrics
       ↓
8. Persist experiment
       ↓
9. Analyze experiment
       ↓
10. Generate report
       ↓
11. Compare models
       ↓
12. Detect regressions
       ↓
13. Display results in Streamlit

🛠️ Tech Stack

Area

Technology

Language

Python

API

FastAPI

Dashboard

Streamlit

Validation

Pydantic

Testing

Pytest

LLM Orchestration

Provider abstraction

Local LLM Runtime

Ollama

Semantic Evaluation

Embedding-based evaluation

Data

JSON

Containerization

Docker / Docker Compose

CI

GitHub Actions

Version Control

Git / GitHub

📂 Project Structure

llm/
├── backend/
│   ├── api/
│   │   └── benchmark.py
│   ├── benchmark/
│   │   ├── runner.py
│   │   ├── evaluator.py
│   │   ├── semantic_evaluator.py
│   │   ├── metrics.py
│   │   ├── storage.py
│   │   ├── cost_calculator.py
│   │   ├── dataset_version.py
│   │   ├── llm_judge.py
│   │   ├── failure_analyzer.py
│   │   ├── comparison_engine.py
│   │   ├── statistics.py
│   │   ├── retry_handler.py
│   │   ├── embedding_cache.py
│   │   ├── quality_cost_latency.py
│   │   ├── regression_detector.py
│   │   ├── regression_service.py
│   │   ├── evaluation_harness.py
│   │   ├── experiment_report.py
│   │   ├── concurrency_runner.py
│   │   ├── report_exporter.py
│   │   └── leaderboard.py
│   ├── llm/
│   │   ├── base.py
│   │   ├── model_registry.py
│   │   ├── provider_factory.py
│   │   ├── ollama_provider.py
│   │   └── provider adapters...
│   ├── schemas/
│   │   └── benchmark.py
│   ├── observability/
│   │   └── logger.py
│   ├── main.py
│   └── run_benchmark.py
│
├── frontend/
│   └── streamlit_app.py
│
├── datasets/
│   ├── benchmark.json
│   ├── benchmark_enhanced.json
│   └── metadata.json
│
├── tests/
│   ├── test_benchmark.py
│   ├── test_evaluator.py
│   ├── test_metrics.py
│   ├── test_retry_handler.py
│   ├── test_regression_detector.py
│   ├── test_evaluation_harness.py
│   ├── test_experiment_report.py
│   ├── test_concurrency_runner.py
│   ├── test_report_exporter.py
│   ├── test_leaderboard.py
│   └── ...
│
├── .github/
│   └── workflows/
│       └── tests.yml
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── pytest.ini
├── .gitignore
└── README.md

📋 Prerequisites

Recommended environment:

Python 3.12+

Git

Docker Desktop (optional but recommended)

Ollama for local models

At least 8 GB RAM recommended for local LLM experimentation

API keys only when using cloud providers

⚙️ Installation

1. Clone the repository

git clone https://github.com/vummidiganesh55/llm-benchmark.git
cd llm-benchmark

2. Create a virtual environment

Windows

python -m venv .venv
.venv\Scripts\activate

Linux / macOS

python -m venv .venv
source .venv/bin/activate

3. Install dependencies

python -m pip install --upgrade pip
pip install -r requirements.txt

🔑 Environment Variables

Create a .env file in the project root.

Example:

# Local Ollama
OLLAMA_BASE_URL=http://localhost:11434

# Optional cloud providers
GOOGLE_API_KEY=
GROQ_API_KEY=
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
MISTRAL_API_KEY=
DEEPSEEK_API_KEY=
XAI_API_KEY=
COHERE_API_KEY=

Never commit .env or API keys.

The repository .gitignore excludes environment secrets and runtime-generated files.

▶️ Usage

Start the FastAPI backend

uvicorn backend.main:app --reload

Open:

http://127.0.0.1:8000/docs

Available benchmark endpoints include:

GET  /benchmark/models
POST /benchmark/run
POST /benchmark/regression
GET  /benchmark/report
POST /benchmark/concurrency
GET  /benchmark/leaderboard

Start the Streamlit dashboard

streamlit run frontend/streamlit_app.py

Open:

http://localhost:8501

Run the final 50-question benchmark

python -m backend.run_benchmark

The final benchmark runner validates the dataset, executes all 50 questions, calculates metrics, and writes the final JSON artifact under results/.

💡 Example

Example benchmark request:

{
  "model": "qwen2.5:3b",
  "limit": 5
}

Example leaderboard request:

GET /benchmark/leaderboard?sort_by=accuracy&descending=true&latest_only=true

Example regression request:

{
  "baseline_experiment_id": "20261002_172627_064433",
  "current_experiment_id": "20261002_172534_051571"
}

The recorded regression experiment detected:

Average latency regression

Tokens-per-second regression

while accuracy and error rate did not regress. The stored regression artifact reports two regressed metrics.

🧪 Testing

Run the complete test suite:

pytest -q

Current local verification:

243 passed, 1 warning

The warning is from a dependency deprecation notice related to google-genai/typing internals and does not currently cause test failure.

⚙️ GitHub Actions CI

The project includes:

.github/workflows/tests.yml

The CI pipeline:

Push / Pull Request
        ↓
Checkout repository
        ↓
Set up Python 3.12
        ↓
Install requirements
        ↓
Run pytest

Workflow:

name: LLM Benchmark CI

on:
  push:
    branches:
      - main
      - master
  pull_request:
    branches:
      - main
      - master

jobs:
  tests:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: "pip"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Run pytest
        run: pytest -q

CI verification status

The local test suite currently passes with:

243 passed, 1 warning

The GitHub Actions workflow is configured to execute the same pytest suite on pushes and pull requests. A successful remote GitHub Actions run should be verified from the repository's Actions tab after the latest push rather than claimed from the local test result.

📊 Evaluation

The platform evaluates LLMs across multiple dimensions rather than using a single score.

Quality

Accuracy
Exact Match
Fuzzy Match
Semantic Similarity
LLM-as-a-Judge

Performance

Average Latency
P50 Latency
P95 Latency
Tokens / Second

Reliability

Error Rate
Retry Rate
Fallback Rate
Fallback Success Rate

Cost

Input Cost
Output Cost
Total Cost
Cost / Question

Analysis

Model Comparison
Failure Analysis
Statistical Runs
Regression Detection
Quality × Latency × Cost
Experiment Reports
Concurrency Benchmarking
Leaderboard

📸 Screenshots / Demo

Important for GitHub: the screenshot assets should be committed to a repository path such as docs/screenshots/. The local outputs/ directory is intentionally ignored by .gitignore, so README image links should point to committed copies rather than the ignored outputs/ directory.

Recommended repository layout:

docs/
└── screenshots/
    ├── backend_ui.png
    ├── final_50Q_benchmark.png
    ├── streamlit_dashboard.png
    └── tests.png

FastAPI / Swagger API



The FastAPI Swagger UI exposes benchmark execution, regression detection, experiment reports, concurrency benchmarking, and model leaderboard endpoints.

Final 50Q Benchmark



Latest captured final 50-question run shown in the project artifacts:

Metric

Result

Questions

50

Successful

50

Failed

0

Accuracy

86.00%

Average Latency

1.303 s

P50 Latency

0.7354 s

P95 Latency

4.1467 s

Tokens / Second

68.4269

Total Tokens

4,458

The screenshot also shows exact match at 86%, fuzzy match at 88%, and average fuzzy similarity of 0.9168. Semantic evaluation and LLM-as-a-Judge were disabled for this captured final run.

Streamlit Dashboard



The dashboard screenshot shows:

30 experiments

6 models

659 questions

83.70% average accuracy

5.98 s average latency

Dataset version 1.0.0

Five benchmark categories

Experiment history

Tests



Current local test verification:

243 passed, 1 warning

Regression Detection

The regression artifact demonstrates detection of measurable changes between two experiments:

Average latency       → regression detected
Tokens / second       → regression detected
Accuracy              → no regression
Error rate            → no regression

Model Leaderboard

The captured leaderboard contains six models:

gemini-3.6-flash
llama3.2:latest
mistral:latest
openai/gpt-oss-20b
qwen2.5-coder:3b
qwen2.5:3b

The leaderboard supports sorting by:

Accuracy

Average latency

P95 latency

Tokens per second

Total cost

Error rate

⚡ Performance

The platform records both model quality and runtime performance.

Example captured benchmark metrics:

Accuracy
Average Latency
P50 Latency
P95 Latency
Tokens / Second
Input Tokens
Output Tokens
Total Tokens

A captured Qwen 2.5 3B experiment recorded:

Accuracy          : 100%
Average Latency   : 0.5589 s
P95 Latency       : 0.5752 s
Tokens / Second   : 95.7288
Error Rate        : 0%

These values belong to a 2-question experiment, not the final 50-question benchmark.

🔐 Security

Security considerations:

API keys are stored in environment variables.

.env is excluded from Git.

Local runtime results are excluded from Git where appropriate.

Generated screenshot/report outputs are excluded from Git unless intentionally copied into documentation assets.

Provider credentials should never be hard-coded.

Benchmark execution should be treated as untrusted workload when arbitrary user prompts/tools are introduced.

🔮 Future Improvements

Planned or possible extensions:

Distributed benchmark execution

Persistent experiment database

Advanced vector-store evaluation

More robust semantic evaluation

Human evaluation interface

Automated model selection

Prompt/version tracking

Dataset expansion

Scheduled benchmark runs

Hardware utilization metrics

GPU/CPU/RAM monitoring

Kubernetes deployment

Distributed worker queues

Advanced cost forecasting

Statistical significance testing

Online regression alerts

⚠️ Limitations

Local model latency depends heavily on hardware and model size.

Cloud provider quotas can affect benchmark completion.

Provider pricing changes over time.

Semantic evaluation quality depends on the embedding/evaluation model.

LLM-as-a-Judge results depend on the selected judge model.

A benchmark score is not a universal measure of production quality.

Historical experiments may have been executed under different runtime/provider conditions.

The captured final 50-question run does not include semantic evaluation or LLM-as-a-Judge scores.

🤝 Contributing

Contributions are welcome.

Suggested workflow:

git checkout -b feature/your-feature

Make changes and run:

pytest -q

Then:

git add .
git commit -m "Add your feature"
git push origin feature/your-feature

Open a pull request against main.

📄 License

This project can be released under the MIT License.

If you publish this repository under MIT, add a LICENSE file containing the standard MIT license text and your chosen copyright holder/year.

👨‍💻 Author

Vummidi Ganesh

AI / Machine Learning Engineer | LLM Engineering | Evaluation | Python

GitHub:
https://github.com/vummidiganesh55

Project:
https://github.com/vummidiganesh55/llm-benchmark

⭐ Project Highlights

✓ Multi-provider LLM benchmarking
✓ 50-question benchmark dataset
✓ Exact + fuzzy + semantic evaluation
✓ LLM-as-a-Judge
✓ Retry + fallback
✓ Failure analysis
✓ Statistical analysis
✓ Regression detection
✓ Experiment tracking
✓ Quality × Latency × Cost
✓ Concurrency benchmarking
✓ JSON / CSV / HTML reporting
✓ Model leaderboard
✓ FastAPI backend
✓ Streamlit dashboard
✓ Docker
✓ Pytest
✓ GitHub Actions CI

Built as an engineering-focused LLM evaluation platform rather than a single-model demo.
