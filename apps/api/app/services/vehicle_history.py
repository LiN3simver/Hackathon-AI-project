"""Plate-scoped RAG with immutable borrower and vehicle snapshots on imports."""

import json

from app.schemas.history import HistoryCitation, HistoryQueryRequest, HistoryQueryResponse
from app.schemas.vehicle_history import VehicleHistoryProfile, VehicleHistorySummary, VehicleRentalHistory
from app.services.history_records import compensation_statistics, is_litter_event, litter_statistics
from app.services.license_plate import normalize_plate
from app.services.operations_store import OperationsStore


def get_vehicle_history(store: OperationsStore, plate: str) -> VehicleHistoryProfile | None:
    plate = normalize_plate(plate)
    if plate is None:
        return None
    with store._connect() as connection:
        accounts = {
            row["rental_id"]: dict(row)
            for row in connection.execute(
                "SELECT * FROM auth_accounts WHERE kind = 'rental' AND rental_id IS NOT NULL"
            ).fetchall()
        }
        snapshots = {}
        for row in connection.execute(
            "SELECT rental_id, metadata_json FROM rental_events "
            "WHERE json_extract(metadata_json, '$.license_plate') IS NOT NULL ORDER BY created_at, id"
        ).fetchall():
            snapshots.setdefault(row["rental_id"], json.loads(row["metadata_json"]))
    rentals = {}
    label = plate
    for rental_id in set(accounts) | set(snapshots):
        account = accounts.get(rental_id, {})
        snapshot = snapshots.get(rental_id, {})
        rental_plate = snapshot.get("license_plate") or account.get("license_plate")
        if normalize_plate(rental_plate or "") != plate:
            continue
        label = snapshot.get("vehicle_label") or account.get("vehicle_label") or plate
        rentals[rental_id] = VehicleRentalHistory(
            rental_id=rental_id,
            renter_id=snapshot.get("renter_id") or account.get("customer_id") or rental_id,
            renter_name=snapshot.get("renter_name") or account.get("display_name") or rental_id,
            rental_date=snapshot.get("rental_date") or account.get("rental_date"),
            return_date=snapshot.get("return_date") or account.get("return_date"),
            uploaded=bool(snapshot.get("history_uploaded")),
        )
    if not rentals:
        return None
    events = []
    for event in store.list_rental_events(user_id=None, limit=None):
        rental = rentals.get(event.rental_id)
        if rental is None:
            continue
        metadata = dict(
            event.metadata,
            license_plate=plate,
            renter_name=rental.renter_name,
            renter_id=rental.renter_id,
            rental_date=str(rental.rental_date or ""),
            return_date=str(rental.return_date or ""),
        )
        events.append(event.model_copy(update={"metadata": metadata}))
    ordered = sorted(rentals.values(), key=lambda rental: (str(rental.rental_date or ""), rental.rental_id))
    return VehicleHistoryProfile(
        license_plate=plate,
        vehicle_label=label,
        rentals=ordered,
        events=events,
        rental_count=len(rentals),
        renter_count=len({rental.renter_id for rental in ordered}),
        event_count=len(events),
        latest_return_date=max(
            (rental.return_date for rental in ordered if rental.return_date),
            default=None,
        ),
        statistics={**compensation_statistics(events), **litter_statistics(events)},
    )


def list_vehicle_history(store: OperationsStore, search: str | None = None) -> list[VehicleHistorySummary]:
    with store._connect() as connection:
        plates = {
            row[0]
            for row in connection.execute(
                "SELECT license_plate FROM auth_accounts WHERE kind = 'rental' AND license_plate IS NOT NULL "
                "UNION SELECT json_extract(metadata_json, '$.license_plate') FROM rental_events "
                "WHERE json_extract(metadata_json, '$.license_plate') IS NOT NULL"
            ).fetchall()
        }
    result = []
    for plate in plates:
        profile = get_vehicle_history(store, plate)
        if profile and (not search or search.lower() in f"{profile.license_plate} {profile.vehicle_label}".lower()):
            result.append(VehicleHistorySummary.model_validate(profile.model_dump()))
    return sorted(result, key=lambda item: (str(item.latest_return_date or ""), item.license_plate), reverse=True)


def query_vehicle_history(store: OperationsStore, request: HistoryQueryRequest) -> HistoryQueryResponse:
    from app.core.config import get_settings
    from app.services.history_query import _event_types, _period_from_question, _rank_events

    plate = request.license_plate
    profile = get_vehicle_history(store, plate)
    if profile is None:
        return HistoryQueryResponse(
            question=request.question,
            license_plate=plate,
            query_mode="sql_plus_retrieval",
            generated_by="template",
            answer="找不到這個車牌的租借歷程。",
            warnings=["vehicle_not_found"],
        )
    try:
        since, until, period = _period_from_question(request.question)
    except ValueError:
        return HistoryQueryResponse(
            question=request.question,
            license_plate=plate,
            query_mode="sql_plus_retrieval",
            generated_by="template",
            answer="日期格式不正確，請使用例如 2026年1月。",
            warnings=["invalid_time_filter"],
        )
    start, end = since.date() if since else None, until.date() if until else None
    if request.date_from is not None or request.date_to is not None:
        start, end = request.date_from, request.date_to
        from datetime import timedelta

        period = f"{start or '起始'} 至 {(end - timedelta(days=1)) if end else '目前'}"
    types, labels = _event_types(request.question)
    rentals = [
        rental
        for rental in profile.rentals
        if (
            not start
            or (rental.return_date or rental.rental_date) is not None
            and (rental.return_date or rental.rental_date) >= start
        )
        and (
            not end
            or (rental.rental_date or rental.return_date) is not None
            and (rental.rental_date or rental.return_date) < end
        )
    ]
    from app.services.history_intent import mentions

    named_renters = set(request.renter_ids) or {
        rental.renter_id
        for rental in rentals
        if any(
            len(identifier) >= 2 and mentions(request.question, identifier)
            for identifier in (rental.renter_name, rental.renter_id)
        )
    }
    if named_renters:
        rentals = [rental for rental in rentals if rental.renter_id in named_renters]
    if request.rental_ids:
        rentals = [rental for rental in rentals if rental.rental_id in request.rental_ids]
    if (
        not types
        and any(term in request.question for term in ("最近一次", "最後一次", "上一次", "最近是誰", "最近誰"))
        and rentals
    ):
        rentals = [
            max(
                rentals,
                key=lambda rental: (str(rental.return_date or ""), str(rental.rental_date or ""), rental.rental_id),
            )
        ]
    rental_ids = {rental.rental_id for rental in rentals}
    events = [
        event
        for event in profile.events
        if event.rental_id in rental_ids
        and (not types or event.event_type in types)
        and (
            event.event_type == "rental_history_uploaded"
            or (not start or event.occurred_on and event.occurred_on >= start)
            and (not end or event.occurred_on and event.occurred_on < end)
        )
    ]
    from app.services.history_query import _DISPUTE_TERMS

    if any(term in request.question.lower() for term in _DISPUTE_TERMS):
        events = [
            event
            for event in events
            if event.disputed or event.event_type in {"dispute_opened", "dispute_resolved", "decision_reversed"}
        ]
    if labels == ["垃圾"]:
        events = [event for event in events if is_litter_event(event)]
    statistics = {
        **compensation_statistics(events),
        **litter_statistics(events),
        "rental_count": len(rentals),
        "renter_count": len({rental.renter_id for rental in rentals}),
        "matched_event_count": len(events),
    }
    matched = _rank_events(request.question, events, get_settings().history_rag_max_events)
    scope = period or "目前保存期間"
    answer = f"車牌 {plate} 在{scope}有 {len(rentals)} 次租借，涉及 {statistics['renter_count']} 位承租人。"
    if "黑名單" in request.question:
        profiles = [
            store.get_user_history_profile(uid, event_limit=1)
            for uid in sorted({rental.renter_id for rental in rentals})
        ]
        listed = [user for user in profiles if user and user.blacklisted]
        statistics["blacklisted_user_count"] = len(listed)
        answer += f"\n這些承租人中，目前有 {len(listed)} 人列入黑名單（目前狀態）。"
        for user in listed:
            answer += f"\n{user.display_name}：{user.blacklist_reason}"
        if any(user is None for user in profiles):
            answer += "\n部分歷史承租人的目前帳號已不存在，無法確認其黑名單狀態。"
    elif not labels:
        for rental in rentals[:20]:
            answer += (
                f"\n{rental.rental_date or '日期未填'} 借車 → {rental.return_date or '日期未填'} 還車："
                f"{rental.renter_name}（{rental.renter_id}），租單 {rental.rental_id}。"
            )
        if len(rentals) > 20:
            answer += "\n此處列出前 20 次；可在對話中指定月份或承租人繼續查詢。"
    if "車損" in labels or "賠償" in labels:
        answer += (
            f"\n已確認責任車損 {statistics['caused_damage_count']} 次，"
            f"應賠 NT$ {statistics['compensation_amount']}，已賠 NT$ {statistics['paid_amount']}，"
            f"未賠 NT$ {statistics['outstanding_amount']}。"
        )
    if "垃圾" in labels or "整潔" in labels:
        answer += (
            f"\n已確認留下垃圾 {statistics['litter_count']} 次，"
            f"待確認 {statistics['pending_litter_count']} 次、爭議中 {statistics['disputed_litter_count']} 次。"
        )
    if labels:
        for event in matched[:10]:
            answer += (
                f"\n{event.occurred_on}｜當時承租人：{event.metadata['renter_name']}"
                f"（{event.metadata['renter_id']}）｜{event.title}"
                f"{'（爭議中）' if event.disputed else ''}：{event.summary}"
            )
        if not events:
            answer += "\n沒有找到符合條件的事件。"
    response = HistoryQueryResponse(
        question=request.question,
        license_plate=plate,
        query_mode="sql_plus_retrieval",
        answer=answer,
        generated_by="template",
        matched_events=matched,
        statistics=statistics,
        interpreted_filters={
            "license_plate": plate,
            "period": period,
            "categories": labels,
            "renter_ids": sorted(named_renters),
            "since": str(start) if start else None,
            "until": str(end) if end else None,
        },
        citations=[
            HistoryCitation(
                event_id=event.id,
                rental_id=event.rental_id,
                source_type=event.source_type,
                source_id=event.source_id,
                created_at=event.created_at,
            )
            for event in matched
        ],
    )
    response._all_events = events
    response._rental_rows = [
        {
            "rental_id": rental.rental_id,
            "display_name": rental.renter_name,
            "renter_name": rental.renter_name,
            "customer_id": rental.renter_id,
            "license_plate": plate,
            "rental_date": rental.rental_date,
            "return_date": rental.return_date,
        }
        for rental in rentals
    ]
    return response
