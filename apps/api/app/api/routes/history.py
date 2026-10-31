from typing import Annotated
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import PlainTextResponse, Response, StreamingResponse

from app.schemas.auth import AuthPrincipal
from app.schemas.history import (
    AutomationSummary,
    BlacklistUpdate,
    CleanlinessHistoryRecordInput,
    DamageHistoryRecordInput,
    HistoryEventDetail,
    HistoryFeedback,
    HistoryFeedbackInput,
    HistoryModelList,
    HistoryModelOption,
    HistoryQueryRequest,
    HistoryQueryResponse,
    HistoryUserSuggestion,
    LitterHistoryRecordInput,
    RentalEvent,
    UserHistoryListItem,
    UserHistoryProfile,
)
from app.services.auth import require_admin, require_ops
from app.services.history_query import query_history
from app.services.history_records import delete_history_record, save_damage_record, update_blacklist
from app.services.operations_store import get_operations_store

router = APIRouter(
    prefix="/history",
    tags=["user rental history"],
    dependencies=[Depends(require_ops)],
)


@router.get("/automation", response_model=AutomationSummary)
def get_automation_summary(
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> AutomationSummary:
    return get_operations_store().automation_summary(limit=limit)


@router.get("/examples", response_model=list[str])
def history_example_questions() -> list[str]:
    from app.services.history_capabilities import example_questions

    return example_questions(get_operations_store())


@router.get("/models", response_model=HistoryModelList)
def history_models() -> HistoryModelList:
    """The composer's 「Model」 picker: text models only, the default first."""
    from app.core.config import get_settings
    from app.services import history_llm_models

    settings = get_settings()
    options = history_llm_models.list_options(settings)
    ready = history_llm_models.available(options, settings)
    return HistoryModelList(default_id=history_llm_models.DEFAULT_ID, options=[
        HistoryModelOption(id=option.id, label=option.label, provider=option.provider, tier=option.tier,
                           description=option.description, available=ready[option.id])
        for option in options])


@router.get("/user-suggestions", response_model=list[HistoryUserSuggestion])
def suggest_history_users(
    search: Annotated[str, Query(min_length=1, max_length=80)],
    limit: Annotated[int, Query(ge=1, le=12)] = 8,
) -> list[HistoryUserSuggestion]:
    from app.services.history_intent import catalogue

    needle = search.strip().casefold()
    if not needle:
        return []
    store = get_operations_store()
    users, _ = catalogue(store)
    current_names = {}
    with store._connect() as connection:
        for row in connection.execute("SELECT customer_id, rental_id, display_name FROM auth_accounts "
                                      "WHERE kind='rental' ORDER BY updated_at DESC, created_at DESC"):
            uid = row["customer_id"] or row["rental_id"]
            if row["display_name"]:
                current_names.setdefault(uid, row["display_name"])
    matches = sorted((uid for uid in users if needle in uid.casefold()),
                     key=lambda uid: (not uid.casefold().startswith(needle), uid.casefold()))
    return [HistoryUserSuggestion(user_id=uid, display_name=current_names.get(uid) or next(
        (name for name in sorted(users[uid]) if name != uid), uid)) for uid in matches[:limit]]


@router.get("/users", response_model=list[UserHistoryListItem])
def list_history_users(
    search: Annotated[str | None, Query(max_length=80)] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
) -> list[UserHistoryListItem]:
    return get_operations_store().list_history_users(search=search, limit=limit)


@router.get("/users/{user_id}", response_model=UserHistoryProfile)
def get_user_history(user_id: str) -> UserHistoryProfile:
    profile = get_operations_store().get_user_history_profile(user_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="找不到使用者或租單歷程")
    return profile


@router.post("/query", response_model=HistoryQueryResponse)
def ask_user_history(request: HistoryQueryRequest) -> HistoryQueryResponse:
    return query_history(request)


@router.post("/query/stream")
async def ask_user_history_stream(request: HistoryQueryRequest) -> StreamingResponse:
    """Server-sent events: the deterministic answer card first, then verified summary sentences."""
    from app.core.config import get_settings
    from app.services.history_answer import answer_events

    return StreamingResponse(
        answer_events(request, get_operations_store(), get_settings()),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/query/export")
def export_user_history(request: HistoryQueryRequest, http: Request) -> Response:
    """Every record behind an answer as .xlsx; the chat itself shows at most 10 photo groups."""
    from app.services.history_export import export_history

    filename, content = export_history(request, get_operations_store(), str(http.base_url).rstrip("/") + "/api/v1")
    return Response(content, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    headers={"Content-Disposition": f"attachment; filename*=UTF-8''{quote(filename)}"})


@router.get("/events/{event_id}/detail", response_model=HistoryEventDetail)
def get_event_detail(event_id: str) -> HistoryEventDetail:
    """What the photos show for one record: AI comparison when a damage case exists, else the record text."""
    from app.services.history_event_detail import build_event_detail
    from app.services.history_evidence import Directory

    store = get_operations_store()
    events = store.list_rental_events(user_id=None, limit=None, event_ids=[event_id])
    if not events:
        raise HTTPException(status_code=404, detail="找不到這筆紀錄")
    directory = Directory(store)
    return build_event_detail(store, events[0], directory.renter_name(events[0]), directory.plate(events[0]))


@router.post("/feedback", response_model=HistoryFeedback, status_code=201)
def create_feedback(
    request: HistoryFeedbackInput,
    actor: Annotated[AuthPrincipal, Depends(require_ops)],
) -> HistoryFeedback:
    from app.services.history_feedback import save_feedback

    return save_feedback(get_operations_store(), request, actor.subject)


@router.get("/feedback", response_model=list[HistoryFeedback])
def list_feedback(
    _: Annotated[AuthPrincipal, Depends(require_admin)],
    rating: Annotated[int | None, Query(ge=-1, le=1)] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
) -> list[HistoryFeedback]:
    from app.services.history_feedback import list_feedback as stored

    return stored(get_operations_store(), rating=rating, limit=limit)


@router.get("/feedback/export", response_class=PlainTextResponse)
def export_feedback(_: Annotated[AuthPrincipal, Depends(require_admin)]) -> PlainTextResponse:
    """Export private answer feedback for an authorized evaluation workflow."""
    from app.services.history_feedback import export_golden

    return PlainTextResponse(export_golden(get_operations_store()), media_type="application/x-ndjson",
                             headers={"Content-Disposition": 'attachment; filename="history_feedback.jsonl"'})


@router.get("/regressions/export", response_class=PlainTextResponse)
def export_regressions(_: Annotated[AuthPrincipal, Depends(require_admin)]) -> PlainTextResponse:
    """Export refused summaries and their private evidence for authorized evaluation."""
    from app.services.history_regressions import export_cases

    return PlainTextResponse(export_cases(get_operations_store()), media_type="application/x-ndjson",
                             headers={"Content-Disposition": 'attachment; filename="history_regressions.jsonl"'})


@router.post("/damage-records", response_model=RentalEvent)
@router.post("/compensation-records", response_model=RentalEvent)
def create_damage_record(
    request: DamageHistoryRecordInput,
    actor: Annotated[AuthPrincipal, Depends(require_admin)],
) -> RentalEvent:
    try:
        return save_damage_record(get_operations_store(), request, actor.subject)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/damage-records", response_model=list[RentalEvent])
@router.get("/compensation-records", response_model=list[RentalEvent])
def list_damage_records(user_id: Annotated[str, Query(min_length=1, max_length=80)]):
    store = get_operations_store()
    if store.get_user_history_profile(user_id) is None:
        raise HTTPException(status_code=404, detail="找不到使用者")
    return store.list_rental_events(
        user_id=user_id, limit=None,
        event_types=["damage_liability_recorded"],
    )


@router.put("/damage-records/{event_id}", response_model=RentalEvent)
@router.put("/compensation-records/{event_id}", response_model=RentalEvent)
def edit_damage_record(
    event_id: str, request: DamageHistoryRecordInput,
    actor: Annotated[AuthPrincipal, Depends(require_admin)],
) -> RentalEvent:
    try:
        return save_damage_record(get_operations_store(), request, actor.subject, event_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.put("/users/{user_id}/blacklist", response_model=UserHistoryProfile)
def set_user_blacklist(
    user_id: str, request: BlacklistUpdate,
    actor: Annotated[AuthPrincipal, Depends(require_admin)],
) -> UserHistoryProfile:
    store = get_operations_store()
    try:
        update_blacklist(store, user_id, request, actor.subject)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return store.get_user_history_profile(user_id)


@router.delete("/damage-records/{event_id}", status_code=204)
@router.delete("/compensation-records/{event_id}", status_code=204)
def remove_damage_record(
    event_id: str, actor: Annotated[AuthPrincipal, Depends(require_admin)],
) -> Response:
    try:
        delete_history_record(get_operations_store(), event_id, "damage_liability_recorded", actor.subject)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return Response(status_code=204)


@router.delete("/litter-records/{event_id}", status_code=204)
def remove_litter_record(
    event_id: str, actor: Annotated[AuthPrincipal, Depends(require_admin)],
) -> Response:
    try:
        delete_history_record(get_operations_store(), event_id, "litter_recorded", actor.subject)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return Response(status_code=204)


@router.put("/cleanliness-records/{event_id}", response_model=RentalEvent)
def edit_cleanliness_record(
    event_id: str, request: CleanlinessHistoryRecordInput,
    actor: Annotated[AuthPrincipal, Depends(require_admin)],
) -> RentalEvent:
    try:
        return save_damage_record(get_operations_store(), request, actor.subject, event_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete("/cleanliness-records/{event_id}", status_code=204)
def remove_cleanliness_record(
    event_id: str, actor: Annotated[AuthPrincipal, Depends(require_admin)],
) -> Response:
    try:
        delete_history_record(get_operations_store(), event_id, "cleanliness_confirmed", actor.subject)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return Response(status_code=204)


@router.get("/litter-records", response_model=list[RentalEvent])
def list_litter_records(user_id: Annotated[str, Query(min_length=1, max_length=80)]):
    store = get_operations_store()
    if store.get_user_history_profile(user_id) is None:
        raise HTTPException(status_code=404, detail="找不到使用者")
    return store.list_rental_events(user_id=user_id, limit=None, event_types=["litter_recorded"])


@router.post("/litter-records", response_model=RentalEvent)
def create_litter_record(
    request: LitterHistoryRecordInput,
    actor: Annotated[AuthPrincipal, Depends(require_admin)],
) -> RentalEvent:
    try:
        return save_damage_record(get_operations_store(), request, actor.subject)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.put("/litter-records/{event_id}", response_model=RentalEvent)
def edit_litter_record(
    event_id: str, request: LitterHistoryRecordInput,
    actor: Annotated[AuthPrincipal, Depends(require_admin)],
) -> RentalEvent:
    try:
        return save_damage_record(get_operations_store(), request, actor.subject, event_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
