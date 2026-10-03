from pydantic import BaseModel, Field


class BenchmarkRequest(BaseModel):
    model: str
    limit: int | None = Field(
        default=None,
        description="Optional number of benchmark questions to run.",
    )


class RegressionRequest(BaseModel):
    baseline_experiment_id: str
    current_experiment_id: str
class ConcurrencyRequest(BaseModel):
    model: str
    limit: int = Field(
        default=5,
        description="Number of benchmark questions to run.",
    )
    concurrency: int = Field(
        default=2,
        description="Maximum number of concurrent requests.",
    )