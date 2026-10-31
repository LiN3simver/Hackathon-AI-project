"""What a damage record's photos show, and whether the mark existed before this rental.

Sources, best first: the damage-pair comparison stored with a reviewed damage case;
explicit fields on a manual or seeded record; otherwise a deterministic reading of
the record text. Everything here is advisory and never changes liability.
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout

from app.schemas.history import (
    DamageRegion,
    HistoryEventDetail,
    PriorDamageCheck,
    PriorRental,
    RentalEvent,
)
from app.services.inspection_storage import get_inspection_storage
from app.services.license_plate import normalize_plate

logger = logging.getLogger("api")

DAMAGE_EVENT_TYPES = ("damage_liability_recorded", "damage_confirmed", "damage_alert_created",
                      "damage_observation_alert_created", "damage_reviewed")
VIEW_LABELS = {"left_front": "左前方", "right_front": "右前方", "left_rear": "左後方", "right_rear": "右後方",
               "interior_front": "車內前座", "interior_rear": "車內後座", "damage_detail": "損傷特寫"}
_LOCATION = re.compile(
    r"(左前|右前|左後|右後|左側|右側|前方|後方|左|右|前|後)?\s*"
    r"(車門|葉子板|輪框|輪圈|大燈|頭燈|尾燈|後照鏡|照後鏡|保險桿|保桿|引擎蓋|車頂|後車廂|行李箱|擋風玻璃|車窗|車頭|車尾|側裙|門把|車身)")
_TYPES = (("刮痕|刮傷|擦傷|刮", "scratch", "刮痕"), ("凹陷|凹痕|凹", "dent", "凹陷"), ("破裂|裂痕|裂", "crack", "裂痕"),
          ("破損|斷裂|脫落|撞壞|撞毀", "broken_part", "破損"), ("掉漆|擦漆", "paint_transfer", "掉漆"),
          ("變形", "deformation", "變形"))
TYPE_LABELS = {code: label for _, code, label in _TYPES} | {"panel_misalignment": "板件錯位", "other": "其他損傷"}
_PICKUP_ABSENT = re.compile(r"借車(?:時|照片|照|前)?[^。，,；;]{0,8}(?:無|沒有|未見|未有|看不到)")
_PICKUP_PRESENT = re.compile(r"借車(?:時|照片|照|前)?[^。，,；;]{0,8}(?:已有|就有|已經有|原本就)|原有損傷|舊傷")
_RETURN_PRESENT = re.compile(r"還車(?:時|照片|照|後)?[^。，,；;]{0,6}(?:新增|出現|發現|多了|有)")


def damage_case_id_for(event: RentalEvent) -> str | None:
    if event.metadata.get("damage_case_id"):
        return str(event.metadata["damage_case_id"])
    return event.source_id if event.source_type == "damage_case_review" else None


def photo_source_for(event: RentalEvent) -> str | None:
    if event.metadata.get("purpose") == "rag" and event.metadata.get("evidence_photos"):
        return "rag_seed"
    return "damage_case" if damage_case_id_for(event) else None


def parse_damage_text(text: str, metadata: dict | None = None) -> dict:
    """Location, type and pickup/return presence from a record; explicit fields win over the text."""
    metadata = metadata or {}
    location_match = _LOCATION.search(text or "")
    location = "".join(part for part in location_match.groups() if part) if location_match else None
    damage_type = next((code for pattern, code, _ in _TYPES if re.search(pattern, text or "")), None)
    at_pickup = False if _PICKUP_ABSENT.search(text or "") else True if _PICKUP_PRESENT.search(text or "") else None
    at_return = True if _RETURN_PRESENT.search(text or "") else None
    explicit = any(metadata.get(key) is not None for key in ("damage_location", "damage_type", "present_at_pickup"))
    return {
        "location": metadata.get("damage_location") or location,
        "damage_type": metadata.get("damage_type") or damage_type,
        "present_at_pickup": metadata.get("present_at_pickup") if metadata.get("present_at_pickup") is not None
        else at_pickup,
        "present_at_return": True if metadata.get("present_at_pickup") is False and at_return is None else at_return,
        "parsed": explicit or any(value is not None for value in (location, damage_type, at_pickup, at_return)),
    }


def responsibility_of(event: RentalEvent) -> str:
    if event.event_type in {"damage_liability_recorded", "litter_recorded"}:
        if event.disputed:
            return "disputed"
        return "confirmed" if event.verified and event.metadata.get("responsibility_confirmed", True) else "pending"
    if event.disputed:
        return "disputed"
    if event.event_type == "cleanliness_confirmed" and event.verified:
        return "confirmed"
    if event.event_type in {"damage_confirmed", "cleanliness_confirmed"}:
        return "pending"
    if event.event_type.endswith("alert_created"):
        return "pending" if event.verified else "ai_only"
    return "not_applicable"


_RESPONSIBILITY_SUFFIX = {"confirmed": "責任已確認", "pending": "責任尚待確認", "disputed": "爭議中，不計入",
                          "ai_only": "AI 偵測，尚未人工確認", "not_applicable": ""}


def conclusion_for(at_pickup: bool | None, at_return: bool | None, responsibility: str) -> str:
    if at_pickup is False and at_return:
        text = "借車時沒有、還車時出現，屬於本次租借期間新增的損傷"
    elif at_pickup:
        text = "借車時已有此損傷，不應算在本次承租人"
    elif at_return:
        text = "還車時發現此損傷，但借車照片未記錄是否已存在"
    else:
        text = "現有照片與紀錄不足以判斷損傷何時出現"
    suffix = _RESPONSIBILITY_SUFFIX.get(responsibility, "")
    return f"{text}（{suffix}）" if suffix else text


def build_event_detail(store, event: RentalEvent, renter_name: str, plate: str | None) -> HistoryEventDetail:
    parsed = parse_damage_text(f"{event.title}。{event.summary}", event.metadata)
    responsibility = responsibility_of(event)
    case_id = damage_case_id_for(event)
    case = store.get_damage_case(case_id) if case_id else None
    detail = dict(
        event_id=event.id, renter_name=renter_name, license_plate=plate, occurred_on=event.occurred_on,
        title=event.title, summary=event.summary, responsibility=responsibility, disputed=event.disputed,
        location=parsed["location"], damage_type=parsed["damage_type"], present_at_pickup=parsed["present_at_pickup"],
        present_at_return=parsed["present_at_return"], photo_source=photo_source_for(event), damage_case_id=case_id,
        photo_slots=(list(event.metadata["evidence_photos"]) if event.metadata.get("evidence_photos")
                     else ["pickup", "return"]),
        source="record_text" if parsed["parsed"] else "none",
    )
    if case is not None:
        comparison = case.comparison
        verification = comparison.verification
        specialist = comparison.specialist
        candidates = comparison.candidates
        if verification and verification.confirmed_region_ids:
            candidates = [item for item in candidates if item.id in verification.confirmed_region_ids] or candidates
        if not candidates and specialist:
            candidates = specialist.regions
        decision = case.review_decision.value
        ai_new = verification is not None and verification.decision.value == "confirmed_new_damage"
        if decision == "confirmed_damage" or ai_new:
            detail.update(present_at_pickup=False, present_at_return=True)
        elif verification is not None and verification.decision.value == "no_new_damage":
            detail.update(present_at_return=False)
        detail.update(
            source="damage_case", view_label=VIEW_LABELS.get(case.expected_view.value, case.expected_view.value),
            ai_summary=(verification.summary if verification else comparison.summary)[:160],
            ai_score=comparison.risk_score, pickup_match_score=specialist.pickup_match_score if specialist else None,
            review_decision=decision, reviewer_notes=case.reviewer_notes or None,
            regions=[DamageRegion(description=item.description[:120], score=item.change_score,
                                  bounding_box=item.bounding_box)
                     for item in sorted(candidates, key=lambda item: item.change_score, reverse=True)[:3]],
        )
        detail["location"] = detail["location"] or detail["view_label"]
    detail["conclusion"] = conclusion_for(detail["present_at_pickup"], detail["present_at_return"], responsibility)
    return HistoryEventDetail(**detail)


def describe(detail: HistoryEventDetail) -> str:
    """One evidence-pack line for the model."""
    def presence(value, yes, no):
        return yes if value else no if value is False else "未記錄"

    parts = [f"部位：{detail.location or '未記錄'}", f"類型：{TYPE_LABELS.get(detail.damage_type or '', '未記錄')}",
             f"借車照片：{presence(detail.present_at_pickup, '已有此損傷', '沒有此損傷')}",
             f"還車照片：{presence(detail.present_at_return, '出現此損傷', '沒有此損傷')}",
             "依據：" + {"damage_case": "AI 照片比對與人工覆核", "record_text": "紀錄文字", "none": "無照片判讀資料"}[
                 detail.source]]
    if detail.ai_summary:
        parts.append(f"AI 說明：{detail.ai_summary}")
    if detail.reviewer_notes:
        parts.append(f"覆核備註：{detail.reviewer_notes[:80]}")
    return "｜".join(parts)


# ---------------------------------------------------------------- prior damage


def _photo_bytes(store, rental_id: str, slot: str, events: list[RentalEvent]) -> bytes | None:
    storage = get_inspection_storage()
    for event in events:
        key = (event.metadata.get("evidence_photos") or {}).get(slot)
        if event.rental_id == rental_id and event.metadata.get("purpose") == "rag" and key:
            path = storage.resolve(key)
            if path is not None:
                return path.read_bytes()
    for case in store.list_damage_cases():
        if case.rental_id == rental_id:
            path = store.get_damage_case_image_file(case.id, slot)
            if path is not None:
                return path.read_bytes()
    return None


# Long-lived, so a timed-out comparison (often the first model load) finishes in the background
# and leaves the model loaded for the next question instead of blocking this answer.
_VISUAL_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="prior-damage")


def _visual_compare(earlier: bytes, current: bytes):
    from app.services.damage_pair_specialist import get_damage_pair_specialist

    specialist = get_damage_pair_specialist()
    if not specialist.available():
        return None
    return specialist.predict(earlier, current)


def check_prior_damage(store, focus: RentalEvent, rental_rows: dict[str, dict], names: dict[str, str],
                       *, visual: bool = True, timeout: float = 10.0) -> PriorDamageCheck:
    """Compare the focus record with earlier rentals of the same car: records first, then photos."""
    row = rental_rows.get(focus.rental_id, {})
    plate = normalize_plate(row.get("license_plate") or focus.metadata.get("license_plate") or "")
    parsed = parse_damage_text(f"{focus.title}。{focus.summary}", focus.metadata)
    start = str(row.get("rental_date") or focus.occurred_on or "")
    earlier = sorted(
        (other for rid, other in rental_rows.items()
         if rid != focus.rental_id and plate and normalize_plate(other.get("license_plate") or "") == plate
         and str(other.get("return_date") or other.get("rental_date") or "") <= start),
        key=lambda other: str(other.get("return_date") or other.get("rental_date") or ""), reverse=True)[:3]
    damage_events = [event for event in store.list_rental_events(user_id=None, limit=None,
                                                                 event_types=list(DAMAGE_EVENT_TYPES))
                     if event.rental_id in {other["rental_id"] for other in earlier}]
    photo_events = store.list_rental_events(user_id=None, limit=None, event_ids=None) if visual else []
    current_return = _photo_bytes(store, focus.rental_id, "return", photo_events) if visual else None
    rentals: list[PriorRental] = []
    visual_hits, visual_clear, checked, timed_out = [], [], False, False
    for other in earlier:
        rid = other["rental_id"]
        own = [event for event in damage_events if event.rental_id == rid
               and not (event.event_type == "damage_reviewed" and event.outcome in {"dismissed", "retry_photos"})]
        locations = {parse_damage_text(f"{event.title}。{event.summary}", event.metadata)["location"] for event in own}
        same = bool(parsed["location"]) and parsed["location"] in locations
        score = None
        earlier_return = _photo_bytes(store, rid, "return", photo_events) if current_return and not timed_out else None
        if earlier_return and current_return:
            try:
                result = _VISUAL_POOL.submit(_visual_compare, earlier_return, current_return).result(timeout=timeout)
            except FutureTimeout:
                logger.warning("prior_damage_visual_timeout rental=%s", rid)
                result, timed_out = None, True
            except Exception:  # the comparison is optional evidence
                logger.exception("prior_damage_visual_failed rental=%s", rid)
                result = None
            if result is not None:
                checked = True
                score = result.pickup_match_score
                (visual_hits if not result.detected or score >= result.threshold else visual_clear).append(other)
        rentals.append(PriorRental(
            renter_name=names.get(rid) or other.get("display_name") or "使用者",
            rental_date=other.get("rental_date") or None, return_date=other.get("return_date") or None,
            records=[f"{event.title}：{event.summary[:60]}" for event in own][:3], same_location=same,
            visual_score=score))
    location = parsed["location"] or "此部位"
    if any(item.same_location for item in rentals):
        first = next(item for item in rentals if item.same_location)
        verdict = "already_present"
        text = (f"較早的租借（{first.renter_name}，{first.return_date or first.rental_date}）已記錄{location}的損傷，"
                "可能不是本次造成，建議人工複查")
    elif visual_hits:
        first = rentals[earlier.index(visual_hits[0])]
        verdict = "already_present"
        text = f"影像比對顯示 {first.renter_name} 還車時已有相似痕跡，可能不是本次造成，建議人工複查"
    elif parsed["present_at_pickup"] is False or (checked and not visual_hits):
        verdict = "new_damage"
        text = f"借車照片與同車較早的租借都沒有{location}損傷的紀錄，判斷為本次租借期間新增"
    elif not earlier:
        verdict = "unknown"
        text = "找不到同車較早的租借紀錄，無法追溯"
    else:
        verdict = "unknown"
        text = "較早的租借沒有足夠紀錄或照片可以比對"
    return PriorDamageCheck(
        focus_event_id=focus.id, renter_name=names.get(focus.rental_id), license_plate=plate,
        location=parsed["location"], damage_type=parsed["damage_type"], verdict=verdict,
        verdict_text=text + "；僅供人工判斷，不會自動變更責任。", visual_checked=checked, earlier_rentals=rentals)
