from pydantic import BaseModel, Field


class DeidentifyRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=50000, description="Raw clinical note text")


class PHISpan(BaseModel):
    start: int
    end: int
    label: str
    text: str
    source: str = Field(description="'ner' or 'regex'")
    confidence: float


class DeidentifyResponse(BaseModel):
    redacted_text: str
    phi_spans: list[PHISpan]
    phi_count: int
    sources: dict[str, int] = Field(description="Counts by detection source")


class HealthResponse(BaseModel):
    status: str
    model: str
    device: str
