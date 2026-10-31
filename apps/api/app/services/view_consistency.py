from __future__ import annotations

from app.schemas.quality import ExpectedView

FRONT_VIEWS = {ExpectedView.LEFT_FRONT, ExpectedView.RIGHT_FRONT}
REAR_VIEWS = {ExpectedView.LEFT_REAR, ExpectedView.RIGHT_REAR}


def vehicle_end(view: ExpectedView) -> str | None:
    if view in FRONT_VIEWS:
        return "front"
    if view in REAR_VIEWS:
        return "rear"
    return None


def same_vehicle_end(first: ExpectedView, second: ExpectedView) -> bool:
    first_end = vehicle_end(first)
    return first_end is not None and first_end == vehicle_end(second)
