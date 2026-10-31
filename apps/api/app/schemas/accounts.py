from datetime import date, datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.services.license_plate import normalize_plate


class ManagedAccountKind(str, Enum):
    RENTAL = "rental"
    STAFF = "staff"
    ADMIN = "admin"


class ManagedAccount(BaseModel):
    subject: str
    kind: ManagedAccountKind
    username: str
    display_name: str
    customer_id: str | None = None
    rental_date: date | None = None
    return_date: date | None = None
    rental_id: str | None = None
    vehicle_id: str | None = None
    vehicle_label: str | None = None
    license_plate: str | None = None
    created_by: str
    created_by_name: str
    created_at: datetime
    updated_at: datetime
    is_active: bool
    is_current: bool


class AccountManagementResponse(BaseModel):
    accounts: list[ManagedAccount]
    active_account_count: int
    active_session_count: int
    active_device_count: int
    active_device_window_minutes: int = 5
    active_inspection_count: int = 0
    queued_analysis_job_count: int = 0
    processing_analysis_job_count: int = 0
    analysis_queue_capacity: int = 0
    analysis_queue_at_capacity: bool = False
    generated_at: datetime


class RentalAccountCreate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    rental_id: str = Field(min_length=1, max_length=80)
    customer_id: str | None = Field(default=None, min_length=1, max_length=80)
    display_name: str | None = Field(default=None, min_length=1, max_length=120)
    rental_date: date | None = None
    return_date: date | None = None
    vehicle_id: str | None = Field(default=None, min_length=1, max_length=40)
    vehicle_label: str = Field(min_length=1, max_length=120)
    license_plate: str = Field(min_length=5, max_length=10)
    password: str = Field(min_length=4, max_length=4, pattern=r"^[0-9]{4}$")

    @model_validator(mode="after")
    def validate_dates(self) -> "RentalAccountCreate":
        if self.rental_date and self.return_date and self.return_date < self.rental_date:
            raise ValueError("還車日期不能早於租車日期")
        return self

    @field_validator("license_plate")
    @classmethod
    def validate_license_plate(cls, value: str) -> str:
        normalized = normalize_plate(value)
        if normalized is None:
            raise ValueError("請輸入有效車牌")
        return normalized


class RentalAccountUpdate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    customer_id: str | None = Field(default=None, min_length=1, max_length=80)
    display_name: str | None = Field(default=None, min_length=1, max_length=120)
    rental_date: date | None = None
    return_date: date | None = None
    vehicle_id: str | None = Field(default=None, min_length=1, max_length=40)
    vehicle_label: str = Field(min_length=1, max_length=120)
    license_plate: str = Field(min_length=5, max_length=10)
    password: str | None = Field(default=None, min_length=4, max_length=4, pattern=r"^[0-9]{4}$")

    @model_validator(mode="after")
    def validate_dates(self) -> "RentalAccountUpdate":
        if self.rental_date and self.return_date and self.return_date < self.rental_date:
            raise ValueError("還車日期不能早於租車日期")
        return self

    @field_validator("license_plate")
    @classmethod
    def validate_license_plate(cls, value: str) -> str:
        normalized = normalize_plate(value)
        if normalized is None:
            raise ValueError("請輸入有效車牌")
        return normalized


class OpsAccountCreate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    username: str = Field(min_length=1, max_length=80)
    display_name: str = Field(min_length=1, max_length=120)
    kind: ManagedAccountKind
    password: str = Field(min_length=8, max_length=120)

    @model_validator(mode="after")
    def validate_ops_kind(self) -> "OpsAccountCreate":
        if self.kind == ManagedAccountKind.RENTAL:
            raise ValueError("營運帳號角色只能是管理者或營運人員")
        return self


class OpsAccountUpdate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    display_name: str = Field(min_length=1, max_length=120)
    kind: ManagedAccountKind
    password: str | None = Field(default=None, min_length=8, max_length=120)

    @model_validator(mode="after")
    def validate_ops_kind(self) -> "OpsAccountUpdate":
        if self.kind == ManagedAccountKind.RENTAL:
            raise ValueError("營運帳號角色只能是管理者或營運人員")
        return self


class DeleteAccountRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    confirmation_identifier: str = Field(min_length=1, max_length=80)
    admin_password: str = Field(min_length=1, max_length=120)


class DeleteAccountResponse(BaseModel):
    deleted_subject: str
    message: str
    management: AccountManagementResponse
