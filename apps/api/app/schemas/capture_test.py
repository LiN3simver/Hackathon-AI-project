from typing import Any

from pydantic import BaseModel, Field

from app.schemas.inspection import InspectionAnalysisTask
from app.schemas.quality import ImageMetrics, QualityResponse


class CaptureTestAnalysis(BaseModel):
    task: InspectionAnalysisTask
    result: dict[str, Any] = Field(default_factory=dict)
    needs_review: bool = False
    error_message: str = ""


class CaptureTestResponse(QualityResponse):
    quality_skipped: bool = False
    # Test flow gave up after TEST_RECOGNITION_LIMIT_SECONDS; nothing was judged.
    recognition_abandoned: bool = False
    score: int | None = Field(default=None, ge=0, le=100)
    metrics: ImageMetrics | None = None
    analyses: list[CaptureTestAnalysis] = Field(default_factory=list)
