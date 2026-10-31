"""Private RAG evidence imports; never invoke inference, training or notification jobs."""

import sqlite3
from uuid import uuid4

from app.schemas.auth import AuthPrincipal
from app.schemas.history import RentalEvent
from app.schemas.rag_seed import RagSeedInput
from app.services.account_store import AccountConflictError, _password_hash
from app.services.history_records import apply_blacklist_rule
from app.services.inspection_storage import InspectionImageStorage
from app.services.operations_store import OperationsStore, _now


def create_rag_seed(store: OperationsStore, storage: InspectionImageStorage,
                    request: RagSeedInput, actor: AuthPrincipal,
                    pickup: bytes | None, returned: bytes | None) -> list[RentalEvent]:
    return create_rag_records(store, storage, [request], actor, [(pickup, returned)])


def create_rag_records(store: OperationsStore, storage: InspectionImageStorage,
                       requests: list[RagSeedInput], actor: AuthPrincipal,
                       photos: list[tuple[bytes | list[bytes] | None, bytes | list[bytes] | None]],
                       additional_photos: list[list[bytes]] | None = None) -> list[RentalEvent]:
    if len(requests) != len(photos):
        raise ValueError("租單與借還車照片組數必須相同")
    if additional_photos is None:
        additional_photos = [[] for _ in requests]
    if len(requests) != len(additional_photos):
        raise ValueError("租單與歷程照片組數必須相同")
    saved_keys: list[str] = []
    source_ids: list[str] = []
    try:
        with store._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            for request, (pickup, returned), evidence in zip(requests, photos, additional_photos, strict=True):
                if ((request.damage or request.data_kind == "vehicle_history")
                        and not (request.data_kind == "vehicle_history" and evidence)
                        and (not pickup or not returned)):
                    raise ValueError("車損或租借歷程需上傳借車與還車照片")
                if not pickup and not returned and not evidence:
                    raise ValueError("請上傳至少一張事件照片")
                rental = request.rental
                seed_id = f"RAG-{uuid4().hex}"
                keys = {}
                for slot, content in (("pickup", pickup), ("return", returned), ("history", evidence)):
                    if content is None:
                        continue
                    images = content if isinstance(content, list) else [content]
                    for index, image in enumerate(images):
                        image_slot = slot if index == 0 else f"{slot}-{index + 1}"
                        keys[image_slot] = storage.save(seed_id, image_slot, image)
                        saved_keys.append(keys[image_slot])
                now = _now()
                connection.execute(
                    """INSERT INTO auth_accounts
                    (subject, kind, username, display_name, customer_id, password_hash, rental_id,
                     vehicle_id, vehicle_label, license_plate, created_by, created_by_name,
                     created_at, updated_at, rental_date, return_date)
                    VALUES (?, 'rental', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (f"rental:{rental.rental_id}", rental.rental_id, rental.display_name,
                     rental.customer_id, _password_hash(rental.password), rental.rental_id,
                     rental.vehicle_id or rental.license_plate, rental.vehicle_label,
                     rental.license_plate, actor.subject, actor.display_name, now, now,
                     rental.rental_date.isoformat(), rental.return_date.isoformat()),
                )
                snapshot = dict(
                    synthetic=request.data_kind == "synthetic", purpose="rag", recognition_skipped=True,
                    history_uploaded=request.data_kind == "vehicle_history", recorded_by=actor.subject,
                    license_plate=rental.license_plate, renter_name=rental.display_name,
                    renter_id=rental.customer_id, rental_date=rental.rental_date.isoformat(),
                    return_date=rental.return_date.isoformat(), vehicle_label=rental.vehicle_label,
                    occurred_on=rental.return_date.isoformat(), evidence_photos=keys,
                )
                if request.data_kind == "vehicle_history":
                    source_id = f"{seed_id}-rental"
                    source_ids.append(source_id)
                    summary = (f"{rental.display_name}於 {rental.rental_date} 借車、"
                               f"{rental.return_date} 還車，車牌 {rental.license_plate}。"
                               f"{request.description}")
                    store._insert_rental_event(
                        connection, rental_id=rental.rental_id,
                        vehicle_id=rental.vehicle_id or rental.license_plate,
                        event_type="rental_history_uploaded", source_type="vehicle_rental_history",
                        source_id=source_id, title="租借歷程（直接上傳）", summary=summary,
                        verified=True, metadata=snapshot,
                    )
                for kind, record in (("damage", request.damage), ("litter", request.litter),
                                     ("cleanliness", request.cleanliness)):
                    if record is None:
                        continue
                    source_id = f"{seed_id}-{kind}"
                    source_ids.append(source_id)
                    metadata = dict(snapshot, record_type=kind,
                                    responsibility_confirmed=record.responsibility_confirmed,
                                    occurred_on=(record.occurred_on or rental.return_date).isoformat())
                    if kind == "damage" or record.claim_enabled:
                        metadata.update(compensation_cents=int(record.compensation_amount * 100),
                                        paid_cents=int(record.paid_amount * 100), currency="TWD")
                    if kind != "damage":
                        metadata["claim_enabled"] = record.claim_enabled
                    if kind == "litter":
                        metadata["litter_present"] = True
                    elif kind == "cleanliness":
                        metadata["litter_present"] = False
                    title = {"damage": "車損責任", "litter": "留下垃圾", "cleanliness": "造成髒汙"}[kind]
                    title = ("已確認" if record.responsibility_confirmed else "待確認") + title
                    if request.data_kind == "synthetic":
                        title = f"[模擬資料] {title}"
                    store._insert_rental_event(
                        connection, rental_id=rental.rental_id,
                        vehicle_id=rental.vehicle_id or rental.license_plate,
                        event_type={"damage": "damage_liability_recorded", "litter": "litter_recorded",
                                    "cleanliness": "cleanliness_confirmed"}[kind],
                        source_type=f"{kind}_history_record", source_id=source_id,
                        title=title, summary=record.description, verified=record.responsibility_confirmed,
                        disputed=record.disputed, metadata=metadata,
                    )
                store._record_activity(
                    connection, entity_type="rag_seed" if snapshot["synthetic"] else "vehicle_rental_history",
                    entity_id=seed_id, action="created", actor=actor.subject,
                    summary=f"建立 RAG 租借歷程 {rental.rental_id}",
                    metadata={"rental_id": rental.rental_id, "synthetic": snapshot["synthetic"],
                              "license_plate": rental.license_plate, "recognition_skipped": True},
                )
            placeholders = ",".join("?" for _ in source_ids)
            rows = connection.execute(
                f"SELECT * FROM rental_events WHERE source_id IN ({placeholders}) ORDER BY created_at, id",
                source_ids,
            ).fetchall()
            events = [store._rental_event(row) for row in rows]
    except Exception as exc:
        for key in saved_keys:
            storage.delete(key)
        if isinstance(exc, sqlite3.IntegrityError):
            raise AccountConflictError("租單編號已存在，請使用新的編號；本批尚未建立") from exc
        raise
    # After the commit, so the profile counts this batch's records too.
    for user_id in dict.fromkeys(request.rental.customer_id for request in requests
                                 if request.damage or request.litter or request.cleanliness):
        apply_blacklist_rule(store, user_id)
    return events
