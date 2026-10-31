import json
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Response

from app.core.config import get_settings
from app.schemas.auth import AuthPrincipal, AuthRole
from app.schemas.operations import (
    Alert,
    AlertStatus,
    ClearDamageReviewDuplicatesResponse,
    ClearOperationsActivityResponse,
    ClearTestDataResponse,
    CreateWorkOrderRequest,
    DeleteInspectionPhotosResponse,
    InspectionPhotoRecord,
    NotificationDeliveryStatus,
    NotificationDispatchResponse,
    NotificationOutboxItem,
    OperationsActivity,
    RecognitionProgress,
    ReviewCleanlinessAlertRequest,
    UpdateWorkOrderRequest,
    WorkOrder,
    WorkOrderStatus,
)
from app.services.auth import require_admin, require_ops
from app.services.email_notifications import (
    dispatch_notification_outbox,
    reconcile_power_automate_outbox,
)
from app.services.history_export import build_xlsx
from app.services.inspection_storage import get_inspection_storage
from app.services.operations_store import OPERATIONS_ACTIVITY_RETENTION_DAYS, get_operations_store

router = APIRouter(
    prefix="/operations",
    tags=["operations"],
    dependencies=[Depends(require_ops)],
)


@router.delete("/damage-review/duplicates", response_model=ClearDamageReviewDuplicatesResponse)
def clear_damage_review_duplicates() -> ClearDamageReviewDuplicatesResponse:
    if get_settings().environment == "production":
        raise HTTPException(status_code=403, detail="正式環境禁止清除車損覆核測試資料")
    deleted = get_operations_store().clear_damage_review_duplicates()
    return ClearDamageReviewDuplicatesResponse(
        **deleted,
        message="已清除與人工真值重複的車損覆核案件及關聯分析工作。",
    )


@router.delete("/damage-review/pending", response_model=ClearDamageReviewDuplicatesResponse)
def clear_pending_damage_review() -> ClearDamageReviewDuplicatesResponse:
    if get_settings().environment == "production":
        raise HTTPException(status_code=403, detail="正式環境禁止清除車損覆核測試資料")
    deleted = get_operations_store().clear_pending_damage_review()
    return ClearDamageReviewDuplicatesResponse(
        **deleted,
        message="已清除待覆核車損案件；學習樣本、原始照片、警示、工單與人工真值均保留。",
    )


@router.delete("/test-data", response_model=ClearTestDataResponse)
def clear_test_data() -> ClearTestDataResponse:
    if get_settings().environment == "production":
        raise HTTPException(status_code=403, detail="正式環境禁止清除營運資料")
    deleted = get_operations_store().clear_test_data()
    return ClearTestDataResponse(
        **deleted,
        message="測試警示、工單、Email 通知與新車損人工覆核案件已清除；操作紀錄永久保留。",
    )


@router.get("/alerts", response_model=list[Alert])
def list_alerts(status: Annotated[AlertStatus | None, Query()] = None) -> list[Alert]:
    return get_operations_store().list_alerts(status.value if status else None)


@router.get("/activity", response_model=list[OperationsActivity])
def list_activity(
    principal: Annotated[AuthPrincipal, Depends(require_ops)],
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
    include_archived: bool = False,
    before_created_at: datetime | None = None,
    before_id: Annotated[str | None, Query(max_length=100)] = None,
    entity_type: Annotated[str | None, Query(max_length=40)] = None,
) -> list[OperationsActivity]:
    is_admin = principal.role == AuthRole.ADMIN
    if not is_admin and entity_type == "learning_sample":
        raise HTTPException(status_code=403, detail="只有後台管理員可以查看訓練資料紀錄")
    if (before_created_at is None) != (before_id is None):
        raise HTTPException(status_code=422, detail="分頁游標的時間與 ID 必須一起提供")
    return get_operations_store().list_operations_activity(
        limit=limit, offset=offset, entity_type=entity_type, include_archived=include_archived,
        # Admins audit everyone; staff only see their own non-private operations.
        actor=None if is_admin else principal.subject, exclude_learning_samples=not is_admin,
        before_created_at=before_created_at.isoformat() if before_created_at else None, before_id=before_id,
    )


@router.get("/activity/export")
def export_activity(principal: Annotated[AuthPrincipal, Depends(require_ops)]) -> Response:
    is_admin = principal.role == AuthRole.ADMIN
    records = get_operations_store().list_operations_activity(
        limit=None, include_archived=True,
        actor=None if is_admin else principal.subject, exclude_learning_samples=not is_admin,
    )
    details = [json.dumps(item.metadata, ensure_ascii=False) for item in records]
    # Excel limits each cell to 32,767 characters and each sheet to 1,048,576 rows.
    parts = max(1, max((len(detail) + 29999) // 30000 for detail in details)) if details else 1
    headers = ["紀錄 ID", "時間（UTC）", "操作人", "類型", "資料 ID", "動作", "摘要"]
    headers += [f"詳細資料 JSON（{i + 1}）" for i in range(parts)]
    rows = [[item.id, item.created_at, item.actor, item.entity_type, item.entity_id, item.action,
             item.summary, *[detail[i * 30000:(i + 1) * 30000] for i in range(parts)]]
            for item, detail in zip(records, details, strict=True)]
    chunks = [rows[i:i + 1_048_575] for i in range(0, len(rows), 1_048_575)] or [[]]
    content = build_xlsx([(f"操作紀錄{index + 1}", headers, chunk, [28, 30, 24, 24, 30, 24, 60] + [80] * parts)
                          for index, chunk in enumerate(chunks)])
    return Response(content, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    headers={"Content-Disposition": 'attachment; filename="operations-history.xlsx"',
                             "Cache-Control": "no-store"})


@router.delete("/activity/expired", response_model=ClearOperationsActivityResponse)
def clear_expired_activity(
    _: Annotated[AuthPrincipal, Depends(require_admin)],
) -> ClearOperationsActivityResponse:
    days = OPERATIONS_ACTIVITY_RETENTION_DAYS
    deleted = get_operations_store().clear_operations_activity_older_than(days=days)
    return ClearOperationsActivityResponse(
        deleted_activity_events=deleted, older_than_days=days,
        message=f"已封存並清除 {days} 天前的 {deleted} 筆營運操作紀錄；仍可匯出查核。",
    )


@router.get("/notifications", response_model=list[NotificationOutboxItem])
def list_notifications(
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    delivery_status: Annotated[
        NotificationDeliveryStatus | None,
        Query(alias="status"),
    ] = None,
) -> list[NotificationOutboxItem]:
    store = get_operations_store()
    reconcile_power_automate_outbox(store=store)
    return store.list_notification_outbox(
        limit=limit,
        status=delivery_status,
    )


@router.post("/notifications/dispatch", response_model=NotificationDispatchResponse)
def dispatch_notifications(
    _: Annotated[AuthPrincipal, Depends(require_admin)],
) -> NotificationDispatchResponse:
    return dispatch_notification_outbox(store=get_operations_store(), limit=50)


@router.post(
    "/notifications/{notification_id}/retry",
    response_model=NotificationOutboxItem,
)
def retry_notification(
    notification_id: str,
    background_tasks: BackgroundTasks,
    _: Annotated[AuthPrincipal, Depends(require_admin)],
) -> NotificationOutboxItem:
    store = get_operations_store()
    try:
        notification = store.retry_notification(notification_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if notification is None:
        raise HTTPException(status_code=404, detail="找不到 Email 通知")
    background_tasks.add_task(dispatch_notification_outbox, store=store)
    return notification


@router.get("/inspection-photos", response_model=list[InspectionPhotoRecord])
def list_inspection_photos(
    limit: Annotated[int, Query(ge=1, le=500)] = 200,
) -> list[InspectionPhotoRecord]:
    return get_operations_store().list_inspection_photo_records(limit=limit)


@router.get("/recognition-progress", response_model=list[RecognitionProgress])
def list_recognition_progress(
    limit: Annotated[int, Query(ge=1, le=100)] = 5,
) -> list[RecognitionProgress]:
    return get_operations_store().list_recognition_progress(limit=limit)


@router.delete(
    "/inspection-photos/{photo_id}",
    response_model=DeleteInspectionPhotosResponse,
)
def delete_inspection_photo(
    photo_id: str,
    principal: Annotated[AuthPrincipal, Depends(require_admin)],
) -> DeleteInspectionPhotosResponse:
    object_key = get_operations_store().delete_inspection_photo_record(
        photo_id, actor=principal.subject
    )
    if object_key is False:
        raise HTTPException(status_code=404, detail="找不到租還車照片紀錄")
    deleted_images = 0
    if isinstance(object_key, str):
        path = get_inspection_storage().resolve(object_key)
        if path is not None:
            get_inspection_storage().delete(object_key)
            deleted_images = 1
    return DeleteInspectionPhotosResponse(
        deleted_records=1,
        deleted_images=deleted_images,
        message="租還車照片紀錄已刪除。",
    )


@router.delete(
    "/inspection-photos",
    response_model=DeleteInspectionPhotosResponse,
)
def clear_inspection_photos(
    principal: Annotated[AuthPrincipal, Depends(require_admin)],
) -> DeleteInspectionPhotosResponse:
    store = get_operations_store()
    deleted_records, object_keys = store.clear_inspection_photo_records(actor=principal.subject)
    storage = get_inspection_storage()
    deleted_images = 0
    for object_key in object_keys:
        path = storage.resolve(object_key)
        if path is not None:
            storage.delete(object_key)
            deleted_images += 1
    return DeleteInspectionPhotosResponse(
        deleted_records=deleted_records,
        deleted_images=deleted_images,
        message="全部租還車照片紀錄已清除。",
    )


@router.patch("/alerts/{alert_id}/acknowledge", response_model=Alert)
def acknowledge_alert(
    alert_id: str,
    principal: Annotated[AuthPrincipal, Depends(require_ops)],
) -> Alert:
    try:
        alert = get_operations_store().set_alert_status(
            alert_id, AlertStatus.ACKNOWLEDGED.value, actor=principal.subject
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if alert is None:
        raise HTTPException(status_code=404, detail="找不到警示")
    return alert


@router.patch("/alerts/{alert_id}/resolve", response_model=Alert)
def resolve_alert(
    alert_id: str,
    principal: Annotated[AuthPrincipal, Depends(require_ops)],
) -> Alert:
    try:
        alert = get_operations_store().set_alert_status(
            alert_id, AlertStatus.RESOLVED.value, actor=principal.subject
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if alert is None:
        raise HTTPException(status_code=404, detail="找不到警示")
    return alert


@router.patch("/alerts/{alert_id}/cancel", response_model=Alert)
def cancel_alert(
    alert_id: str,
    principal: Annotated[AuthPrincipal, Depends(require_ops)],
) -> Alert:
    store = get_operations_store()
    active_order = next(
        (
            order
            for order in store.list_work_orders()
            if order.alert_id == alert_id
            and order.status not in {WorkOrderStatus.COMPLETED, WorkOrderStatus.CANCELLED}
        ),
        None,
    )
    if active_order is not None:
        raise HTTPException(status_code=409, detail="請先取消進行中的整備工單")
    try:
        alert = store.set_alert_status(
            alert_id, AlertStatus.CANCELLED.value, actor=principal.subject
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if alert is None:
        raise HTTPException(status_code=404, detail="找不到警示")
    return alert


@router.post("/alerts/{alert_id}/cleanliness-review", response_model=Alert)
def review_cleanliness_alert(
    alert_id: str,
    request: ReviewCleanlinessAlertRequest,
    principal: Annotated[AuthPrincipal, Depends(require_ops)],
    background_tasks: BackgroundTasks,
) -> Alert:
    store = get_operations_store()
    try:
        alert = store.review_cleanliness_alert(
            alert_id,
            decision=request.decision,
            reviewer_notes=request.reviewer_notes,
            actor=principal.subject,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if alert is None:
        raise HTTPException(status_code=404, detail="找不到警示")
    background_tasks.add_task(dispatch_notification_outbox, store=store)
    return alert


@router.post("/alerts/{alert_id}/work-order", response_model=WorkOrder, status_code=201)
def create_work_order(
    alert_id: str,
    request: CreateWorkOrderRequest,
    principal: Annotated[AuthPrincipal, Depends(require_ops)],
) -> WorkOrder:
    store = get_operations_store()
    alert = store.get_alert(alert_id)
    if alert is None:
        raise HTTPException(status_code=404, detail="找不到警示")
    if alert.status in {AlertStatus.RESOLVED, AlertStatus.CANCELLED}:
        raise HTTPException(status_code=409, detail="已結束的警示不能建立工單")
    actor = principal.subject
    if alert.status == AlertStatus.OPEN:
        store.set_alert_status(alert_id, AlertStatus.ACKNOWLEDGED.value, actor=actor)
    return store.create_work_order(
        alert, request.assignee, request.notes, actor=actor
    )


@router.get("/work-orders", response_model=list[WorkOrder])
def list_work_orders(status: Annotated[WorkOrderStatus | None, Query()] = None) -> list[WorkOrder]:
    return get_operations_store().list_work_orders(status.value if status else None)


@router.patch("/work-orders/{work_order_id}", response_model=WorkOrder)
def update_work_order(
    work_order_id: str,
    request: UpdateWorkOrderRequest,
    principal: Annotated[AuthPrincipal, Depends(require_ops)],
) -> WorkOrder:
    if request.status == WorkOrderStatus.CANCELLED and principal.role != AuthRole.ADMIN:
        raise HTTPException(status_code=403, detail="只有後台管理員可以取消或刪除工單")
    try:
        work_order = get_operations_store().update_work_order(
            work_order_id,
            status=request.status.value if request.status else None,
            assignee=request.assignee,
            notes=request.notes,
            actor=principal.subject,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if work_order is None:
        raise HTTPException(status_code=404, detail="找不到工單")
    return work_order


@router.delete("/work-orders/{work_order_id}", status_code=204)
def delete_work_order(
    work_order_id: str,
    principal: Annotated[AuthPrincipal, Depends(require_admin)],
) -> Response:
    if not get_operations_store().delete_work_order(work_order_id, actor=principal.subject):
        raise HTTPException(status_code=404, detail="找不到工單")
    return Response(status_code=204)
