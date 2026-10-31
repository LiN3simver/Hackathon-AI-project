from pydantic import BaseModel, Field


class PlatePrediction(BaseModel):
    plate_detected: bool
    plate_text: str | None = None
    normalized_plate: str | None = None
    confidence: float = Field(ge=0, le=1)
    is_confident: bool
    bounding_box: list[float] | None = Field(default=None, min_length=4, max_length=4)
    image_usable: bool
    expected_plate: str | None = None
    matches_expected: bool | None = None
    message: str
    model_version: str
    latency_ms: float = Field(ge=0)
    model_uncertain: bool = False
    timed_out: bool = False
    fast_matched: bool = False
    matched_characters: str | None = None


class PlateModelInfo(BaseModel):
    name: str
    version: str
    engine: str
    status: str
    available: bool
    message: str
    python: str
    model_files: list[str]
    source: str
    license: str
    requires_training: bool
    human_confirmation: bool


class PlateRecognitionResponse(BaseModel):
    request_id: str
    filename: str
    prediction: PlatePrediction
