from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class AuthAudience(str, Enum):
    USER = "user"
    OPS = "ops"


class AuthRole(str, Enum):
    USER = "user"
    STAFF = "staff"
    ADMIN = "admin"


class UserLoginRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    rental_id: str = Field(min_length=1, max_length=80)
    password: str = Field(min_length=1, max_length=120)


class OpsLoginRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    username: str = Field(min_length=1, max_length=80)
    password: str = Field(min_length=1, max_length=120)


class AuthPrincipal(BaseModel):
    subject: str
    audience: AuthAudience
    role: AuthRole
    display_name: str
    customer_id: str | None = None
    rental_id: str | None = None
    vehicle_id: str | None = None
    vehicle_label: str | None = None
    license_plate: str | None = None
    session_id: str | None = None


class AuthSession(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_at: datetime
    principal: AuthPrincipal
