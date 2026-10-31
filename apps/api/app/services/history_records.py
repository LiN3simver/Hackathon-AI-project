"""Human-entered responsibility, compensation and blacklist records."""

import json
from decimal import Decimal
from uuid import uuid4

from app.schemas.history import (
    BlacklistUpdate,
    CleanlinessHistoryRecordInput,
    DamageHistoryRecordInput,
    LitterHistoryRecordInput,
    RentalEvent,
)
from app.services.operations_store import OperationsStore, _now


def compensation_statistics(events: list[RentalEvent]) -> dict:
    records = {
        (event.source_type, event.source_id): event
        for event in events
        if event.event_type == "damage_liability_recorded"
        and event.verified
        and not event.disputed
    }
    damage_records = list(records.values())
    damage_amount = sum(int(event.metadata.get("compensation_cents", 0)) for event in damage_records)
    damage_paid = sum(int(event.metadata.get("paid_cents", 0)) for event in damage_records)
    for event in events:
        if (event.event_type in {"litter_recorded", "cleanliness_confirmed"}
                and event.metadata.get("claim_enabled") and event.verified and not event.disputed):
            records[(event.source_type, event.source_id)] = event
    amount = sum(int(event.metadata.get("compensation_cents", 0)) for event in records.values())
    paid = sum(int(event.metadata.get("paid_cents", 0)) for event in records.values())
    statistics = {
        "compensation_count": sum(
            int(event.metadata.get("compensation_cents", 0)) > 0 for event in records.values()
        ),
        "compensation_amount": f"{Decimal(amount) / 100:.2f}",
        "paid_amount": f"{Decimal(paid) / 100:.2f}",
        "outstanding_amount": f"{Decimal(amount - paid) / 100:.2f}",
    }
    statistics.update(
        caused_damage_count=len(damage_records),
        damage_compensation_count=sum(int(event.metadata.get("compensation_cents", 0)) > 0 for event in damage_records),
        damage_compensation_amount=f"{Decimal(damage_amount) / 100:.2f}",
        damage_paid_amount=f"{Decimal(damage_paid) / 100:.2f}",
        damage_outstanding_amount=f"{Decimal(damage_amount - damage_paid) / 100:.2f}",
    )
    return statistics


def is_litter_event(event: RentalEvent) -> bool:
    return event.event_type == "litter_recorded" or (
        event.event_type in {"cleanliness_confirmed", "cleanliness_alert_created"}
        and event.metadata.get("litter_present") is True
        and not event.metadata.get("litter_dismissed")
    )


def confirmed_dirty_count(events: list[RentalEvent]) -> int:
    grouped: dict[str, list[RentalEvent]] = {}
    for event in events:
        if event.event_type in {"cleanliness_confirmed", "cleanliness_alert_created"}:
            key = str(event.metadata.get("cleanliness_alert_id") or event.source_id)
            grouped.setdefault(key, []).append(event)
    return sum(
        not any(event.disputed or is_litter_event(event) or event.metadata.get("litter_dismissed") for event in records)
        and any(event.event_type == "cleanliness_confirmed" and event.verified for event in records)
        for records in grouped.values()
    )


def litter_statistics(events: list[RentalEvent]) -> dict:
    grouped: dict[tuple[str, str], list[RentalEvent]] = {}
    for event in events:
        if not is_litter_event(event):
            continue
        alert_id = event.metadata.get("cleanliness_alert_id")
        key = ("alert", str(alert_id or event.source_id)) if (
            alert_id or event.event_type != "litter_recorded"
        ) else ("manual", event.source_id)
        grouped.setdefault(key, []).append(event)
    disputed = {key for key, records in grouped.items() if any(event.disputed for event in records)}
    confirmed = {key for key, records in grouped.items()
                 if key not in disputed and any(event.verified for event in records)}
    pending = set(grouped) - confirmed - disputed
    return dict(litter_count=len(confirmed), dirty_count=confirmed_dirty_count(events),
                pending_litter_count=len(pending),
                disputed_litter_count=len(disputed), has_litter_history=bool(confirmed))


def save_damage_record(
    store: OperationsStore,
    request: DamageHistoryRecordInput | LitterHistoryRecordInput | CleanlinessHistoryRecordInput,
    actor: str,
    event_id: str | None = None,
) -> RentalEvent:
    is_litter = isinstance(request, LitterHistoryRecordInput)
    is_cleanliness = isinstance(request, CleanlinessHistoryRecordInput)
    event_type = (
        "cleanliness_confirmed" if is_cleanliness
        else "litter_recorded" if is_litter else "damage_liability_recorded"
    )
    source_type = (
        "cleanliness_history_record" if is_cleanliness
        else "litter_history_record" if is_litter else "damage_history_record"
    )
    case_id = request.damage_case_id if isinstance(request, DamageHistoryRecordInput) else None
    alert_id = request.cleanliness_alert_id if is_litter else None
    with store._connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        account = connection.execute(
            "SELECT * FROM auth_accounts WHERE kind = 'rental' AND rental_id = ?",
            (request.rental_id,),
        ).fetchone()
        if account is None:
            raise ValueError("找不到租單，請先建立測試租單")
        current = None
        if event_id:
            current = connection.execute(
                "SELECT * FROM rental_events WHERE id = ? "
                "AND event_type = ? AND source_type = ?",
                (event_id, event_type, source_type),
            ).fetchone()
            if current is None:
                raise LookupError("找不到車損或垃圾紀錄")
            old = json.loads(current["metadata_json"])
            if (current["rental_id"] != request.rental_id
                    or current["event_type"] != event_type
                    or old.get("damage_case_id") != case_id
                    or old.get("cleanliness_alert_id") != alert_id):
                raise ValueError("修改紀錄時不能變更租單、紀錄類型或關聯案件")
        if case_id:
            case = connection.execute(
                "SELECT * FROM damage_cases WHERE id = ? AND rental_id = ?",
                (case_id, request.rental_id),
            ).fetchone()
            if case is None or case["review_decision"] != "confirmed_damage":
                raise ValueError("需連結同一租單且已人工確認車損的案件")
        if alert_id:
            alert = connection.execute(
                "SELECT * FROM alerts WHERE id = ? AND rental_id = ? AND type = 'cleanliness'",
                (alert_id, request.rental_id),
            ).fetchone()
            if alert is None or alert["review_decision"] != "confirmed":
                raise ValueError("需連結同一租單且已人工確認的垃圾警示")
            if not any(item.get("category") == "trash" and item.get("visible") is True
                       for item in json.loads(alert["evidence_json"])):
                raise ValueError("此警示沒有可見垃圾證據，不能登錄為垃圾紀錄")
        source_id = (current["source_id"] if current else
                     case_id or alert_id
                     or f"CHR-{uuid4().hex[:12].upper()}")
        if not current and connection.execute(
            "SELECT 1 FROM rental_events WHERE source_type = ? "
            "AND source_id = ?", (source_type, source_id),
        ).fetchone():
            raise ValueError("此警示已有垃圾紀錄，請修改原紀錄" if is_litter
                             else "此案件已有賠償紀錄，請修改原紀錄")
        metadata = {
            **(json.loads(current["metadata_json"]) if current else {}),
            "occurred_on": request.occurred_on.isoformat() if request.occurred_on else None,
            "damage_case_id": case_id,
            "cleanliness_alert_id": alert_id,
            "record_type": "cleanliness" if is_cleanliness else "litter" if is_litter else "damage",
            "responsibility_confirmed": request.responsibility_confirmed,
            "recorded_by": actor,
        }
        if is_cleanliness:
            metadata["litter_present"] = False
            title = "已確認造成髒汙" if request.responsibility_confirmed else "待確認造成髒汙"
        elif is_litter:
            metadata["litter_present"] = True
            title = "已確認使用者留下垃圾" if request.responsibility_confirmed else "待確認垃圾紀錄"
        else:
            metadata.update(compensation_cents=int(request.compensation_amount * 100),
                            paid_cents=int(request.paid_amount * 100), currency="TWD",
                            damage_location=request.damage_location, damage_type=request.damage_type,
                            present_at_pickup=request.present_at_pickup)
            title = "已確認使用者責任車損" if request.responsibility_confirmed else "待確認責任車損"
        if is_litter or is_cleanliness:
            metadata["claim_enabled"] = request.claim_enabled
            if request.claim_enabled:
                metadata.update(compensation_cents=int(request.compensation_amount * 100),
                                paid_cents=int(request.paid_amount * 100), currency="TWD")
            else:
                for key in ("compensation_cents", "paid_cents", "currency"):
                    metadata.pop(key, None)
        if current:
            if metadata.get("synthetic"):
                title = f"[模擬資料] {title}"
            connection.execute(
                """UPDATE rental_events SET title = ?, summary = ?, verified = ?,
                disputed = ?, metadata_json = ? WHERE id = ?""",
                (title, request.description, int(request.responsibility_confirmed),
                 int(request.disputed), json.dumps(metadata, ensure_ascii=False), event_id),
            )
        else:
            store._insert_rental_event(
                connection, rental_id=request.rental_id, vehicle_id=account["vehicle_id"],
                event_type=event_type, source_type=source_type,
                source_id=source_id, title=title, summary=request.description,
                verified=request.responsibility_confirmed, disputed=request.disputed,
                metadata=metadata,
            )
            event_id = connection.execute(
                "SELECT id FROM rental_events WHERE source_type = ? "
                "AND source_id = ?", (source_type, source_id),
            ).fetchone()["id"]
        store._record_activity(
            connection, entity_type=source_type, entity_id=event_id,
            action="updated" if current else "created", actor=actor,
            summary=title, metadata={"before": dict(current) if current else None,
                                     "after": request.model_dump(mode="json")},
        )
    user_id = account["customer_id"] or request.rental_id
    apply_blacklist_rule(store, user_id)
    return next(event for event in store.list_rental_events(user_id=user_id, limit=None) if event.id == event_id)


def delete_history_record(
    store: OperationsStore, event_id: str, event_type: str, actor: str,
) -> None:
    with store._connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        current = connection.execute(
            "SELECT * FROM rental_events WHERE id = ? AND event_type = ? "
            "AND (? != 'cleanliness_confirmed' OR source_type = 'cleanliness_history_record')",
            (event_id, event_type, event_type),
        ).fetchone()
        if current is None:
            raise LookupError("找不到車損或垃圾紀錄")
        connection.execute("DELETE FROM rental_events WHERE id = ?", (event_id,))
        store._record_activity(
            connection, entity_type=current["source_type"], entity_id=event_id,
            action="deleted", actor=actor, summary="刪除車損或垃圾紀錄",
            metadata={"before": dict(current)},
        )


BLACKLIST_OFFENCES = 2
BLACKLIST_RULE_ACTOR = "system:blacklist-rule"


def apply_blacklist_rule(store: OperationsStore, user_id: str) -> bool:
    """List a renter once confirmed damage, litter and dirt reach BLACKLIST_OFFENCES.

    Pending and disputed records do not count. The rule only adds: it never removes anyone,
    and a renter a person has taken off the list stays off.
    """
    profile = store.get_user_history_profile(user_id)
    if profile is None or profile.blacklisted:
        return False
    offences = profile.caused_damage_count + profile.litter_count + profile.dirty_count
    if offences < BLACKLIST_OFFENCES:
        return False
    with store._connect() as connection:
        if connection.execute("SELECT 1 FROM history_blacklist WHERE user_id = ?", (profile.user_id,)).fetchone():
            return False
    reason = (f"累積 {offences} 次已確認不良紀錄（車損 {profile.caused_damage_count} 次、"
              f"留下垃圾 {profile.litter_count} 次、髒污 {profile.dirty_count} 次），"
              f"達 {BLACKLIST_OFFENCES} 次自動列入")
    update_blacklist(store, profile.user_id, BlacklistUpdate(blacklisted=True, reason=reason),
                     BLACKLIST_RULE_ACTOR)
    return True


def update_blacklist(
    store: OperationsStore, user_id: str, request: BlacklistUpdate, actor: str,
) -> None:
    profile = store.get_user_history_profile(user_id)
    if profile is None:
        raise LookupError("找不到使用者")
    user_id = profile.user_id
    with store._connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        before = connection.execute(
            "SELECT * FROM history_blacklist WHERE user_id = ?", (user_id,),
        ).fetchone()
        connection.execute(
            """INSERT INTO history_blacklist (user_id, blacklisted, reason, updated_by, updated_at)
            VALUES (?, ?, ?, ?, ?) ON CONFLICT(user_id) DO UPDATE SET
            blacklisted = excluded.blacklisted, reason = excluded.reason,
            updated_by = excluded.updated_by, updated_at = excluded.updated_at""",
            (user_id, int(request.blacklisted), request.reason, actor, _now()),
        )
        store._record_activity(
            connection, entity_type="history_blacklist", entity_id=user_id,
            action="added" if request.blacklisted else "removed", actor=actor,
            summary=request.reason, metadata={"before": dict(before) if before else None,
                                               "after": request.model_dump()},
        )
