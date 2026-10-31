import json
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import ValidationError

from app.core.config import get_settings
from app.schemas.auth import AuthPrincipal
from app.schemas.history import RentalEvent
from app.schemas.rag_seed import RagSeedInput
from app.services.account_store import AccountConflictError, get_account_store
from app.services.auth import require_admin, require_ops
from app.services.inspection_storage import get_inspection_storage
from app.services.operations_store import get_operations_store
from app.services.rag_seed import create_rag_seed
from app.services.upload_validation import read_validated_image

router = APIRouter(prefix="/admin/rag-seeds", tags=["RAG synthetic data"])


@router.post("", response_model=list[RentalEvent], status_code=201)
async def create_seed(
    actor: Annotated[AuthPrincipal, Depends(require_admin)],
    payload: Annotated[str, Form()],
    pickup: Annotated[UploadFile | None, File()] = None,
    returned: Annotated[UploadFile | None, File()] = None,
) -> list[RentalEvent]:
    try:
        request = RagSeedInput.model_validate_json(payload)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if request.data_kind != "synthetic":
        raise HTTPException(status_code=422, detail="此入口僅接受模擬資料，租借歷程請使用歷程上傳區")
    limit = get_settings().max_upload_mb
    if request.damage and (pickup is None or returned is None):
        raise HTTPException(status_code=422, detail="車損案件需上傳借車與還車照片")
    if pickup is None and returned is None:
        raise HTTPException(status_code=422, detail="請上傳至少一張事件照片")
    pickup_bytes = await read_validated_image(pickup, limit) if pickup is not None else None
    return_bytes = await read_validated_image(returned, limit) if returned is not None else None
    get_account_store()  # Ensure the shared account schema exists before the transaction.
    try:
        return create_rag_seed(get_operations_store(), get_inspection_storage(), request,
                               actor, pickup_bytes, return_bytes)
    except AccountConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("", response_model=list[RentalEvent])
def list_seeds(_: Annotated[AuthPrincipal, Depends(require_admin)]) -> list[RentalEvent]:
    store = get_operations_store()
    with store._connect() as connection:
        rows = connection.execute(
            "SELECT * FROM rental_events WHERE json_extract(metadata_json, '$.synthetic') = 1 "
            "AND json_extract(metadata_json, '$.purpose') = 'rag' ORDER BY created_at DESC LIMIT 100"
        ).fetchall()
    return [store._rental_event(row) for row in rows]


@router.get("/{event_id}/images/{slot}")
def seed_image(event_id: str, slot: str,
               _: Annotated[AuthPrincipal, Depends(require_ops)]) -> FileResponse:
    with get_operations_store()._connect() as connection:
        row = connection.execute("SELECT metadata_json FROM rental_events WHERE id = ?",
                                 (event_id,)).fetchone()
    metadata = json.loads(row["metadata_json"]) if row else {}
    key = metadata.get("evidence_photos", {}).get(slot)
    if (not (metadata.get("synthetic") or metadata.get("history_uploaded"))
            or metadata.get("purpose") != "rag" or not key):
        raise HTTPException(status_code=404, detail="找不到歷程照片")
    path = get_inspection_storage().resolve(key)
    if path is None:
        raise HTTPException(status_code=404, detail="找不到歷程照片")
    return FileResponse(path, headers={"Cache-Control": "private, no-store"})
