from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.accounts import RentalAccountCreate
from app.schemas.history import CleanlinessHistoryRecordInput, DamageHistoryRecordInput, LitterHistoryRecordInput


class RagSeedInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rental: RentalAccountCreate
    data_kind: Literal["synthetic", "vehicle_history"] = "synthetic"
    description: str = Field(default="", max_length=1000)
    damage: DamageHistoryRecordInput | None = None
    litter: LitterHistoryRecordInput | None = None
    cleanliness: CleanlinessHistoryRecordInput | None = None

    @model_validator(mode="after")
    def validate_records(self) -> "RagSeedInput":
        if self.data_kind == "synthetic" and not any((self.damage, self.litter, self.cleanliness)):
            raise ValueError("請選擇車損、留下垃圾或造成髒汙，至少一種")
        if not all((self.rental.customer_id, self.rental.display_name,
                    self.rental.rental_date, self.rental.return_date)):
            raise ValueError("請填寫使用者編號、姓名與借還車日期")
        for record in (self.damage, self.litter, self.cleanliness):
            if record is None:
                continue
            if record.rental_id != self.rental.rental_id:
                raise ValueError("事件與租單編號必須相同")
            if getattr(record, "damage_case_id", None) or getattr(record, "cleanliness_alert_id", None):
                raise ValueError("模擬資料不能連結既有辨識案件")
        return self


class VehicleHistoryBatchInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    records: list[RagSeedInput] = Field(min_length=1, max_length=50)

    @model_validator(mode="after")
    def validate_batch(self) -> "VehicleHistoryBatchInput":
        if any(record.data_kind != "vehicle_history" for record in self.records):
            raise ValueError("請使用同車牌租借歷程資料類型")
        if len({record.rental.license_plate for record in self.records}) != 1:
            raise ValueError("每批必須使用同一車牌")
        if len({record.rental.rental_id for record in self.records}) != len(self.records):
            raise ValueError("同一批租單編號不可重複")
        return self
