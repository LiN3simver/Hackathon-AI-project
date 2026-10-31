import json
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from pydantic import ValidationError

from app.core.config import get_settings
from app.schemas.auth import AuthPrincipal
from app.schemas.history import RentalEvent
from app.schemas.rag_seed import VehicleHistoryBatchInput
from app.schemas.vehicle_history import VehicleHistoryProfile, VehicleHistorySummary
from app.services.account_store import AccountConflictError, get_account_store
from app.services.auth import require_admin, require_ops
from app.services.inspection_storage import get_inspection_storage
from app.services.operations_store import get_operations_store
from app.services.rag_seed import create_rag_records
from app.services.upload_validation import read_validated_image
from app.services.vehicle_history import get_vehicle_history, list_vehicle_history

router = APIRouter(prefix="/history/vehicles", tags=["vehicle rental history"])


@router.get("", response_model=list[VehicleHistorySummary], dependencies=[Depends(require_ops)])
def list_vehicles(search: Annotated[str | None, Query(max_length=80)] = None):
    return list_vehicle_history(get_operations_store(), search)


@router.post("/import", response_model=list[RentalEvent], status_code=201)
async def import_history(
    actor: Annotated[AuthPrincipal, Depends(require_admin)], payload: Annotated[str, Form()],
    pickups: Annotated[list[UploadFile] | None, File()] = None,
    returns: Annotated[list[UploadFile] | None, File()] = None,
    evidence: Annotated[list[UploadFile] | None, File()] = None,
    photo_counts: Annotated[str | None, Form()] = None,
):
    try:
        request = VehicleHistoryBatchInput.model_validate_json(payload)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    pickups, returns, evidence = pickups or [], returns or [], evidence or []
    counts = [{"pickup": 1, "returned": 1, "evidence": 0} for _ in request.records]
    if photo_counts is not None:
        try:
            counts = json.loads(photo_counts)
            if isinstance(counts, list):
                for pair in counts:
                    if isinstance(pair, dict):
                        pair.setdefault("evidence", 0)
            if (not isinstance(counts, list) or len(counts) != len(request.records)
                    or any(not isinstance(pair, dict) or any(
                        type(pair.get(slot)) is not int or pair[slot] < 0
                        for slot in ("pickup", "returned", "evidence"))
                        or (pair["evidence"] == 0 and (pair["pickup"] == 0 or pair["returned"] == 0))
                        for pair in counts)):
                raise ValueError
        except (ValueError, TypeError):
            raise HTTPException(
                status_code=422, detail="請提供有效照片數量；可直接上傳歷程資料夾，或分別上傳借車與還車照片",
            ) from None
    if (sum(pair["pickup"] for pair in counts) != len(pickups)
            or sum(pair["returned"] for pair in counts) != len(returns)
            or sum(pair["evidence"] for pair in counts) != len(evidence)):
        raise HTTPException(status_code=422, detail="資料夾照片數量與租單不符，請重新選擇資料夾")
    photos = []
    additional_photos = []
    pickup_offset = return_offset = evidence_offset = 0
    for pair in counts:
        pickup_group = pickups[pickup_offset:pickup_offset + pair["pickup"]]
        return_group = returns[return_offset:return_offset + pair["returned"]]
        evidence_group = evidence[evidence_offset:evidence_offset + pair["evidence"]]
        photos.append(([await read_validated_image(image, get_settings().max_upload_mb) for image in pickup_group],
                       [await read_validated_image(image, get_settings().max_upload_mb) for image in return_group]))
        additional_photos.append([
            await read_validated_image(image, get_settings().max_upload_mb) for image in evidence_group
        ])
        pickup_offset += pair["pickup"]
        return_offset += pair["returned"]
        evidence_offset += pair["evidence"]
    get_account_store()
    try:
        return create_rag_records(get_operations_store(), get_inspection_storage(), request.records, actor,
                                  photos, additional_photos=additional_photos)
    except AccountConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/{plate}", response_model=VehicleHistoryProfile, dependencies=[Depends(require_ops)])
def vehicle_history(plate: str):
    profile = get_vehicle_history(get_operations_store(), plate)
    if profile is None:
        raise HTTPException(status_code=404, detail="找不到車牌租借歷程")
    return profile
