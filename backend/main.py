from fastapi import FastAPI

from backend.api.benchmark import router as benchmark_router


app = FastAPI(
    title="LLM Benchmark Platform",
    description="Platform for benchmarking and evaluating LLMs",
    version="0.1.0"
)


app.include_router(
    benchmark_router,
    prefix="/benchmark",
    tags=["Benchmark"]
)


@app.get("/")
def root():

    return {
        "status": "running",
        "service": "LLM Benchmark Platform",
        "version": "0.1.0"
    }