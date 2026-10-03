# 🚀 LLM Benchmarking Platform

![Python](https://img.shields.io/badge/Python-3.12%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B)
![Pytest](https://img.shields.io/badge/Tests-243%20Passed-brightgreen)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED)
![GitHub Actions](https://img.shields.io/badge/CI-GitHub%20Actions-2088FF)
![License](https://img.shields.io/badge/License-MIT-green)

A production-oriented **LLM evaluation and benchmarking platform** for measuring, comparing, and analyzing Large Language Models across **quality, latency, throughput, reliability, cost, semantic similarity, failure behavior, and regression performance**.

The platform supports multiple LLM providers, reproducible benchmark experiments, automated evaluation, statistical analysis, regression detection, experiment reports, concurrency testing, model leaderboard generation, and an interactive Streamlit dashboard.

---

# 📌 Description

The **LLM Benchmarking Platform** is designed to answer a practical question:

> **How can different LLMs be evaluated consistently using measurable and reproducible metrics?**

Instead of evaluating an LLM only by manually looking at responses, this platform executes a standardized benchmark dataset and records quantitative measurements for every experiment.

The system provides:

- Multi-model benchmarking
- Multi-provider architecture
- Automated answer evaluation
- Latency measurement
- Token throughput measurement
- Reliability analysis
- Retry and fallback mechanisms
- Cost analysis
- Embedding caching
- Statistical analysis
- Regression detection
- Experiment tracking
- Evaluation harness
- Experiment reports
- JSON / CSV / HTML report export
- Model leaderboard
- Concurrency benchmarking
- Streamlit dashboard
- FastAPI REST API
- Docker support
- Automated testing
- GitHub Actions CI

---

# 🎯 Problem Statement

LLM applications are often evaluated using informal testing or a small number of manually selected examples.

This creates several problems:

- Model quality is difficult to compare consistently.
- Latency can vary significantly between models.
- Token throughput is often ignored.
- API failures and rate limits are not systematically measured.
- Regression between model versions can go unnoticed.
- Cost is difficult to compare across providers.
- Benchmark results may not be reproducible.
- There is often no centralized experiment history.
- Manual evaluation does not scale.

The goal of this project is to build an **experiment-driven LLM benchmarking system** that converts LLM evaluation into measurable engineering data.

---

# 💡 Solution

The platform uses a standardized benchmark dataset and executes experiments through a common provider interface.

Each benchmark records:

- Input question
- Expected answer
- Model response
- Evaluation result
- Latency
- Provider latency
- Token usage
- Retry information
- Fallback information
- Error information
- Category-level performance

The collected results are then processed by evaluation, statistics, reliability, cost, regression, reporting, and leaderboard components.

---

# 🚀 Features

## 🔹 Core Benchmarking

- 50-question benchmark dataset
- Multiple benchmark categories
- Model/provider abstraction
- Configurable benchmark execution
- Per-question result tracking
- Experiment IDs
- Persistent experiment storage

## 🔹 Evaluation

- Exact-match evaluation
- Fuzzy-match evaluation
- Keyword evaluation
- Semantic similarity evaluation
- LLM-as-a-Judge support
- Category-level evaluation
- Quality metrics

## 🔹 Performance Analysis

- Average latency
- P50 latency
- P95 latency
- Provider latency
- Tokens per second
- Token efficiency
- Throughput analysis

## 🔹 Reliability

- Error rate
- Retry tracking
- Retry rate
- Fallback support
- Fallback success rate
- Failure analysis
- Provider failure tracking

## 🔹 Cost Analysis

- Input token cost
- Output token cost
- Total cost
- Cost per question
- Quality × Latency × Cost analysis

## 🔹 Experiment Management

- Dataset versioning
- Reproducible experiment configuration
- Experiment tracking
- Experiment reports
- JSON reports
- CSV reports
- HTML reports

## 🔹 Model Comparison

- Model comparison engine
- Model leaderboard
- Latest experiment comparison
- Sortable benchmark metrics
- Accuracy comparison
- Latency comparison
- Throughput comparison
- Cost comparison
- Error-rate comparison

## 🔹 Advanced Evaluation

- Regression detection
- Evaluation harness
- Statistical analysis
- Concurrency benchmarking
- Embedding cache
- Dataset enhancement

## 🔹 Developer Experience

- FastAPI backend
- Swagger/OpenAPI documentation
- Streamlit dashboard
- Docker support
- Pytest test suite
- GitHub Actions CI

---

# 🏗️ Architecture

```text
                         ┌──────────────────────────┐
                         │       Streamlit UI       │
                         │      Dashboard           │
                         └────────────┬─────────────┘
                                      │
                                      ▼
                         ┌──────────────────────────┐
                         │       FastAPI API        │
                         │ /run /report /regression│
                         │ /leaderboard /concurrency│
                         └────────────┬─────────────┘
                                      │
                                      ▼
                         ┌──────────────────────────┐
                         │    Benchmark Runner      │
                         └────────────┬─────────────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    │                 │                 │
                    ▼                 ▼                 ▼
             ┌────────────┐    ┌────────────┐    ┌────────────┐
             │   Ollama   │    │ Cloud APIs │    │  Providers │
             │   Models   │    │            │    │   Factory  │
             └─────┬──────┘    └─────┬──────┘    └────────────┘
                   │                  │
                   └────────┬─────────┘
                            ▼
                  ┌─────────────────────┐
                  │ Evaluation Pipeline │
                  ├─────────────────────┤
                  │ Exact Match         │
                  │ Fuzzy Match         │
                  │ Keyword Match       │
                  │ Semantic Evaluation │
                  │ LLM-as-a-Judge      │
                  └──────────┬──────────┘
                             │
                             ▼
                ┌──────────────────────────┐
                │ Metrics & Analysis Layer │
                ├──────────────────────────┤
                │ Quality                  │
                │ Latency                  │
                │ Throughput               │
                │ Cost                     │
                │ Reliability              │
                │ Statistics               │
                │ Regression Detection     │
                └────────────┬─────────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Experiment Storage   │
                  └──────────┬───────────┘
                             │
                ┌────────────┼────────────┐
                ▼            ▼            ▼
          ┌──────────┐ ┌───────────┐ ┌────────────┐
          │ Reports  │ │Leaderboard│ │ Dashboard  │
          └──────────┘ └───────────┘ └────────────┘
````

---

# 🔄 Workflow

```text
Benchmark Dataset
       │
       ▼
Dataset Validation
       │
       ▼
Experiment Configuration
       │
       ▼
Benchmark Runner
       │
       ▼
LLM Provider
       │
       ▼
Model Response
       │
       ▼
Evaluation Pipeline
       │
       ├── Exact Match
       ├── Fuzzy Match
       ├── Keyword Match
       ├── Semantic Evaluation
       └── LLM-as-a-Judge
       │
       ▼
Metrics Calculation
       │
       ├── Quality
       ├── Latency
       ├── Throughput
       ├── Cost
       └── Reliability
       │
       ▼
Experiment Storage
       │
       ├── Regression Detection
       ├── Statistical Analysis
       ├── Reports
       ├── Leaderboard
       └── Dashboard
```

---

# 🛠️ Tech Stack

| Layer             | Technology                   |
| ----------------- | ---------------------------- |
| Language          | Python                       |
| API               | FastAPI                      |
| UI                | Streamlit                    |
| LLM Framework     | Custom Provider Architecture |
| Local LLM         | Ollama                       |
| Cloud Providers   | Provider adapters            |
| Validation        | Pydantic                     |
| Testing           | Pytest                       |
| Data Processing   | JSON / Python                |
| Embeddings        | Ollama embedding models      |
| Containerization  | Docker                       |
| CI/CD             | GitHub Actions               |
| API Documentation | Swagger / OpenAPI            |
| Version Control   | Git / GitHub                 |

---

# 📂 Project Structure

```text
llm/
│
├── backend/
│   ├── api/
│   │   └── benchmark.py
│   │
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
│   │   ├── dataset_enhancer.py
│   │   ├── regression_detector.py
│   │   ├── regression_service.py
│   │   ├── evaluation_harness.py
│   │   ├── experiment_report.py
│   │   ├── concurrency_runner.py
│   │   ├── report_exporter.py
│   │   ├── leaderboard.py
│   │   └── quality_cost_latency.py
│   │
│   ├── llm/
│   │   ├── base.py
│   │   ├── ollama_provider.py
│   │   ├── model_registry.py
│   │   ├── provider_factory.py
│   │   └── provider adapters
│   │
│   ├── schemas/
│   │   └── benchmark.py
│   │
│   ├── observability/
│   │   └── logger.py
│   │
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
│   ├── test_storage.py
│   ├── test_regression_detector.py
│   ├── test_evaluation_harness.py
│   ├── test_experiment_report.py
│   ├── test_concurrency_runner.py
│   ├── test_report_exporter.py
│   ├── test_leaderboard.py
│   └── ...
│
├── docs/
│   └── screenshots/
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
```

---

# 📋 Prerequisites

Before running the project, install:

* Python 3.12+
* Git
* Docker Desktop
* Ollama
* Required Ollama models

Example local model:

```bash
ollama pull qwen2.5:3b
```

Check installed models:

```bash
ollama list
```

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/vummidiganesh55/llm-benchmark.git
cd llm-benchmark
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 🔑 Environment Variables

Create a `.env` file in the project root if you want to use cloud providers.

Example:

```env
MISTRAL_API_KEY=your_mistral_api_key
GOOGLE_API_KEY=your_google_api_key
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
GROQ_API_KEY=your_groq_api_key
COHERE_API_KEY=your_cohere_api_key
XAI_API_KEY=your_xai_api_key
```

For local Ollama benchmarking, an API key is not required.

### Security

Never commit:

```text
.env
API keys
access tokens
private credentials
```

The repository `.gitignore` excludes environment files and generated runtime outputs.

---

# ▶️ Usage

## Start FastAPI Backend

```bash
uvicorn backend.main:app --reload
```

Open Swagger:

```text
http://127.0.0.1:8000/docs
```

Available benchmark endpoints include:

```text
GET  /benchmark/models
POST /benchmark/run
POST /benchmark/regression
GET  /benchmark/report
POST /benchmark/concurrency
GET  /benchmark/leaderboard
```

---

## Start Streamlit Dashboard

```bash
streamlit run frontend/streamlit_app.py
```

Open:

```text
http://localhost:8501
```

---

## Run the Final 50-Question Benchmark

```bash
python -m backend.run_benchmark
```

The runner validates the benchmark dataset, executes all 50 questions, calculates metrics, prints individual results, and stores the final JSON result under:

```text
results/final_50q_qwen2.5_3b.json
```

---

# 💡 Example

Example benchmark configuration:

```json
{
  "model": "qwen2.5:3b",
  "limit": 50
}
```

Example metrics produced by the platform:

```json
{
  "total": 50,
  "successful": 50,
  "failed": 0,
  "accuracy": 0.86,
  "average_latency": 1.303,
  "p50_latency": 0.7354,
  "p95_latency": 4.1467,
  "tokens_per_second": 68.4269
}
```

The exact benchmark result depends on the model, hardware, runtime state, dataset version, and experiment configuration.

---

# 🧪 Testing

The project uses **Pytest** for automated testing.

Run:

```bash
pytest -q
```

Current local verification:

```text
243 passed, 1 warning
```

The warning is a dependency deprecation warning related to a Google Generative AI dependency and Python 3.17 compatibility.

The test suite covers:

* Benchmark runner
* Evaluators
* Metrics
* Storage
* Dataset versioning
* Retry handling
* Cost calculation
* Embedding cache
* Comparison engine
* Statistical analysis
* Failure analysis
* LLM-as-a-Judge
* Model registry
* Regression detection
* Regression service
* Evaluation harness
* Experiment reports
* Concurrency runner
* Report exporter
* Model leaderboard

---

# 📊 Evaluation

The platform evaluates LLMs using multiple dimensions.

## Quality Metrics

* Accuracy
* Exact Match
* Fuzzy Match
* Keyword Match
* Average Fuzzy Similarity
* Semantic Match
* Average Semantic Similarity
* LLM-as-a-Judge scores

## Performance Metrics

* Average Latency
* P50 Latency
* P95 Latency
* Provider Latency
* Tokens per Second
* Throughput

## Reliability Metrics

* Error Rate
* Retry Rate
* Fallback Rate
* Fallback Success Rate
* Average Attempts
* Average Retries

## Cost Metrics

* Input Cost
* Output Cost
* Total Cost
* Cost per Question

## Advanced Analysis

* Statistical benchmark analysis
* Quality × Latency × Cost
* Regression detection
* Failure analysis
* Experiment comparison
* Concurrency analysis

---

# 🏆 Model Leaderboard

The platform provides a model leaderboard based on stored experiments.

Supported leaderboard metrics include:

```text
accuracy
average_latency
p95_latency
tokens_per_second
total_cost_usd
error_rate
```

The leaderboard exposes measurable results rather than assigning an overall winner.

The current stored leaderboard contains experiments across six model entries:

* Gemini 3.6 Flash
* Llama 3.2
* Mistral
* GPT-OSS 20B
* Qwen2.5-Coder 3B
* Qwen2.5 3B

The leaderboard artifact records **30 experiments across 6 models**. 

---

# 📈 Regression Detection

The platform compares a baseline experiment against a current experiment.

Example regression analysis:

```text
Baseline:
qwen2.5:3b
Experiment: 20261002_172627_064433

Current:
qwen2.5:3b
Experiment: 20261002_172534_051571
```

The recorded comparison detected changes in:

```text
Average Latency
Tokens Per Second
```

while accuracy and error rate did not trigger regression thresholds.   

This provides a foundation for detecting performance regressions between experiments.

---

# 🔬 Experiment Reports

The platform generates structured experiment reports containing:

```text
Experiment
Summary
Quality
Reliability
Performance
Cost
Category Performance
```

Reports can be exported as:

```text
JSON
CSV
HTML
```

---

# ⚡ Performance

A captured 50-question benchmark run recorded:

```text
Total Questions      : 50
Successful Questions : 50
Failed Questions     : 0
Accuracy             : 86%
Average Latency      : 1.303 seconds
P50 Latency          : 0.7354 seconds
P95 Latency          : 4.1467 seconds
Tokens / Second      : 68.4269
```

These measurements are experiment-specific and should not be interpreted as universal model performance.

The platform is designed to preserve experiment-level measurements so different models and runs can be analyzed under the same evaluation framework.

---

# 📸 Screenshots / Demo

## FastAPI Swagger API

![FastAPI Swagger API](docs/screenshots/backend_ui.png)

The Swagger interface exposes the benchmark API endpoints for:

* Model discovery
* Benchmark execution
* Regression detection
* Experiment reports
* Concurrency benchmarking
* Model leaderboard

---

## Final 50Q Benchmark

![Final 50Q Benchmark](docs/screenshots/final_50Q_benchmark.png)

Example captured benchmark output showing:

* 50 questions
* Successful execution
* Accuracy
* Latency
* Token throughput
* Evaluation metrics

---

## Streamlit Dashboard

![Streamlit Dashboard](docs/screenshots/streamlit_dashboard.png)

The dashboard provides multiple analytical views:

* Overview
* Model Comparison
* Quality
* Categories
* Performance
* Cost
* Reliability
* Failures
* Statistics
* Experiments
* Question Explorer

---

## Automated Tests

![Automated Tests](docs/screenshots/tests.png)

Current local test verification:

```text
243 passed, 1 warning
```

---

# 🔄 GitHub Actions CI

The project includes a GitHub Actions workflow:

```text
.github/workflows/tests.yml
```

The CI pipeline:

```text
Git Push / Pull Request
        │
        ▼
Checkout Repository
        │
        ▼
Setup Python 3.12
        │
        ▼
Install Dependencies
        │
        ▼
Run Pytest
        │
        ▼
Test Result
```

The workflow automatically runs the test suite for pushes and pull requests targeting the main branches.

Local verification before the latest repository update:

```text
243 passed, 1 warning
```

---

# 🐳 Docker

The project includes Docker support.

Build:

```bash
docker compose build
```

Start:

```bash
docker compose up
```

The architecture supports running:

* FastAPI backend
* Streamlit frontend
* Ollama on the host/local environment

Docker provides a reproducible application environment while keeping local model execution separate.

---

# 🔐 Security

Security considerations implemented in the project include:

* API keys stored through environment variables
* `.env` excluded from Git
* Generated runtime outputs excluded from Git
* No credentials hard-coded into the benchmark configuration
* Provider-specific API adapters
* Validation of benchmark request schemas
* Controlled benchmark execution
* Error handling around provider failures

### Never commit

```text
.env
API keys
Private credentials
Access tokens
Local secrets
```

---

# 🔮 Future Improvements

Potential future enhancements include:

* Distributed benchmarking
* GPU utilization monitoring
* Memory usage benchmarking
* Streaming latency measurement
* More standardized benchmark datasets
* Human evaluation interface
* Advanced LLM-as-a-Judge calibration
* Persistent database-backed experiment storage
* Authentication and authorization
* Cloud deployment
* Kubernetes deployment
* Distributed worker execution
* Advanced observability
* Prometheus metrics
* Grafana dashboards
* Scheduled benchmark execution
* Automated performance regression alerts

---

# ⚠️ Limitations

Current limitations include:

* Benchmark results depend on local hardware and runtime conditions.
* Local Ollama models have different inference characteristics from cloud APIs.
* Cloud provider quotas and rate limits can affect experiments.
* Some provider integrations require API credentials.
* Cost measurements depend on configured provider pricing.
* LLM-as-a-Judge requires a configured judge model/provider.
* Semantic evaluation depends on the configured embedding model.
* Benchmark datasets are relatively small and should not be treated as comprehensive evaluations of general model capability.
* A single benchmark run should not be interpreted as universal model performance.

---

# 🤝 Contributing

Contributions are welcome.

## Development workflow

```bash
git clone https://github.com/vummidiganesh55/llm-benchmark.git

cd llm-benchmark

python -m venv .venv

pip install -r requirements.txt

pytest -q
```

For a new feature:

```text
1. Create a feature branch
2. Implement the change
3. Add tests
4. Run the complete test suite
5. Update documentation
6. Create a pull request
```

---

# 📄 License

This project is licensed under the MIT License.

See the `LICENSE` file for details.

---

# 👨‍💻 Author

## Vummidi Ganesh

**AI / Machine Learning Engineer**

Interested in:

* Artificial Intelligence
* Machine Learning
* Large Language Models
* NLP
* RAG
* AI Agents
* LLM Evaluation
* MLOps
* Generative AI

GitHub:

[https://github.com/vummidiganesh55](https://github.com/vummidiganesh55)

---

# ⭐ Project Highlights

This project demonstrates practical experience with:

```text
Python
        ↓
LLM Provider Architecture
        ↓
Benchmark Engineering
        ↓
Evaluation Pipelines
        ↓
Quality / Latency / Cost Analysis
        ↓
Reliability Engineering
        ↓
Regression Detection
        ↓
Experiment Tracking
        ↓
Report Generation
        ↓
Model Leaderboards
        ↓
FastAPI
        ↓
Streamlit
        ↓
Pytest
        ↓
Docker
        ↓
GitHub Actions
```

The project is designed as an **engineering-focused LLM evaluation platform**, emphasizing measurable experiments, reproducibility, automated testing, and production-oriented evaluation workflows.

````

### One important thing before you paste it

For the screenshots to render on GitHub, the repository needs these files:

```text
docs/screenshots/backend_ui.png
docs/screenshots/final_50Q_benchmark.png
docs/screenshots/streamlit_dashboard.png
docs/screenshots/tests.png
````

Your `outputs/` folder can remain ignored.

Also, I intentionally used your **captured 86% / 50-question run** in the Performance section rather than silently substituting another benchmark result. The experiment/leaderboard artifacts contain some historical records with inconsistent question-count fields, so the README avoids presenting those records as a clean 50-question comparison. 
