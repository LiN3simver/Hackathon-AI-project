from datetime import datetime
from enum import Enum

from pydantic import BaseModel


class ReadinessLevel(str, Enum):
    READY = "ready"
    WARNING = "warning"
    ERROR = "error"


class ReadinessCheck(BaseModel):
    key: str
    label: str
    status: ReadinessLevel
    required: bool
    message: str
    detail: str | None = None


class ReadinessReport(BaseModel):
    overall: str
    checked_at: datetime
    checks: list[ReadinessCheck]
    ready_count: int
    total_count: int
