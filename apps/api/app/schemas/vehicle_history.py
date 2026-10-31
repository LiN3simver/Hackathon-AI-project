from datetime import date

from pydantic import BaseModel, Field

from app.schemas.history import RentalEvent


class VehicleRentalHistory(BaseModel):
    rental_id: str
    renter_id: str
    renter_name: str
    rental_date: date | None = None
    return_date: date | None = None
    uploaded: bool = False


class VehicleHistorySummary(BaseModel):
    license_plate: str
    vehicle_label: str
    rental_count: int
    renter_count: int
    event_count: int
    latest_return_date: date | None = None


class VehicleHistoryProfile(VehicleHistorySummary):
    rentals: list[VehicleRentalHistory] = Field(default_factory=list)
    events: list[RentalEvent] = Field(default_factory=list)
    statistics: dict = Field(default_factory=dict)
