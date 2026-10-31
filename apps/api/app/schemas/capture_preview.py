from pydantic import BaseModel, Field

from app.schemas.quality import CoveragePrediction


class CapturePreviewResponse(BaseModel):
    coverage_prediction: CoveragePrediction | None = None
    messages: list[str] = Field(default_factory=list)
