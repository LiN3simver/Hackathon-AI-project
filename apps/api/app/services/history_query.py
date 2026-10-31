from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from threading import BoundedSemaphore

from app.core.config import get_settings
from app.schemas.history import (
    HistoryCitation,
    HistoryQueryRequest,
    HistoryQueryResponse,
    RentalEvent,
)
from app.services.history_records import compensation_statistics, is_litter_event, litter_statistics
from app.services.operations_store import OperationsStore, get_operations_store

_DAMAGE_TERMS = {"損壞", "損毀", "弄壞", "車損", "刮傷", "凹痕", "damage"}
_CLEAN_TERMS = {"髒污", "弄髒", "清潔", "污漬", "clean", "dirty"}
_LITTER_TERMS = {"垃圾", "trash", "litter"}
_COMPENSATION_TERMS = {"賠償", "賠款", "應賠", "已賠", "未賠", "欠款"}
_DISPUTE_TERMS = {"爭議", "申訴", "撤銷", "dispute", "appeal"}
_generation_slot = BoundedSemaphore(value=1)
_TAIPEI = timezone(timedelta(hours=8))


def _period_from_question(
    question: str,
) -> tuple[datetime | None, datetime | None, str | None]:
    """Inclusive start / exclusive end in Taiwan calendar dates."""
    month_names = {
        "一": 1,
        "二": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
        "十": 10,
        "十一": 11,
        "十二": 12,
    }
    day_match = re.search(r"(?<![A-Za-z0-9_-])(\d{4})[-/](\d{1,2})[-/](\d{1,2})(?!\d)", question)
    if day_match:
        start = datetime(*(int(value) for value in day_match.groups()), tzinfo=_TAIPEI)
        return start, start + timedelta(days=1), start.strftime("%Y年%m月%d日")
    match = re.search(
        r"(?<![A-Za-z0-9_-])(\d{4})\s*(?:年\s*)?(\d{1,2}|十二|十一|十|[一二三四五六七八九])\s*月"
        r"|(?<![A-Za-z0-9_-])(\d{4})[-/](\d{1,2})(?![-/\d])",
        question,
    )
    if match:
        year = int(match.group(1) or match.group(3))
        raw_month = match.group(2) or match.group(4)
        month = month_names.get(raw_month) or int(raw_month)
        start = datetime(year, month, 1, tzinfo=_TAIPEI)
        end = datetime(year + (month == 12), month % 12 + 1, 1, tzinfo=_TAIPEI)
        return start, end, f"{year}年{month}月"
    year_match = re.search(r"(?<![A-Za-z0-9_-])((?:19|20)\d{2})(?:\s*年)?(?![\d_-])", question)
    if year_match:
        year = int(year_match.group(1))
        return (datetime(year, 1, 1, tzinfo=_TAIPEI), datetime(year + 1, 1, 1, tzinfo=_TAIPEI), f"{year}年")
    return _relative_period(question)


def _relative_period(question: str) -> tuple[datetime | None, datetime | None, str | None]:
    start, label = _since_from_question(question)
    return start, None, label


def _tokens(value: str) -> set[str]:
    lowered = value.lower()
    latin = set(re.findall(r"[a-z0-9_-]{2,}", lowered))
    chinese_runs = re.findall(r"[\u4e00-\u9fff]+", lowered)
    chinese: set[str] = set()
    for run in chinese_runs:
        chinese.add(run)
        chinese.update(run[index : index + 2] for index in range(max(0, len(run) - 1)))
    return latin | chinese


def _since_from_question(question: str) -> tuple[datetime | None, str | None]:
    now = datetime.now(_TAIPEI)
    match = re.search(r"近\s*(\d+|一|兩|二|三)\s*(天|日|個月|月|年)", question)
    if match:
        written_numbers = {"一": 1, "兩": 2, "二": 2, "三": 3}
        amount = written_numbers.get(match.group(1)) or int(match.group(1))
        amount = max(1, min(3650, amount))
        unit = match.group(2)
        if unit in {"天", "日"}:
            since = now - timedelta(days=amount)
        elif unit in {"個月", "月"}:
            since = now - timedelta(days=amount * 30)
        else:
            since = now - timedelta(days=amount * 365)
        return since, match.group(0)
    if "今年" in question:
        return datetime(now.year, 1, 1, tzinfo=_TAIPEI), "今年"
    return None, None


def _event_types(question: str) -> tuple[list[str] | None, list[str]]:
    lowered = question.lower()
    types: set[str] = set()
    labels: list[str] = []
    if any(term in lowered for term in _DAMAGE_TERMS):
        types.update(
            {
                "damage_confirmed",
                "damage_reviewed",
                "damage_alert_created",
                "damage_observation_alert_created",
                "automation_decision",
                "damage_liability_recorded",
            }
        )
        labels.append("車損")
    if any(term in lowered for term in _CLEAN_TERMS):
        types.update(
            {
                "cleanliness_confirmed",
                "cleanliness_alert_created",
                "manual_review_completed",
                "automation_decision",
                "litter_recorded",
            }
        )
        labels.append("整潔")
    if any(term in lowered for term in _LITTER_TERMS):
        types.update({"litter_recorded", "cleanliness_confirmed", "cleanliness_alert_created"})
        labels.append("垃圾")
    if any(term in lowered for term in _COMPENSATION_TERMS) and not any(
        term in lowered for term in _DAMAGE_TERMS | _CLEAN_TERMS
    ):
        types.update({"damage_liability_recorded", "litter_recorded", "cleanliness_confirmed"})
        labels.append("賠償")
    if any(term in lowered for term in _DISPUTE_TERMS):
        types.update(
            {"dispute_opened", "dispute_resolved", "decision_reversed", "damage_liability_recorded", "litter_recorded"}
        )
        labels.append("爭議")
    return (sorted(types) if types else None), labels


def _rank_events(question: str, events: list[RentalEvent], limit: int) -> list[RentalEvent]:
    query_tokens = _tokens(question)
    ranked: list[tuple[float, RentalEvent]] = []
    for event in events:
        searchable = " ".join(
            [
                event.event_type,
                event.title,
                event.summary,
                event.outcome or "",
                event.rental_id,
                event.vehicle_id,
                json.dumps(event.metadata, ensure_ascii=False),
            ]
        )
        overlap = len(query_tokens & _tokens(searchable))
        score = float(overlap * 3)
        if event.verified:
            score += 1.5
        if event.event_type.endswith("confirmed"):
            score += 2
        if event.event_type == "automation_decision":
            score += 0.25
        ranked.append((score, event))
    ranked.sort(key=lambda item: (item[0], item[1].created_at), reverse=True)
    positive = [event for score, event in ranked if score > 0]
    return (positive or [event for _, event in ranked])[:limit]


def _template_answer(events: list[RentalEvent], period: str | None) -> str:
    rental_ids = sorted({event.rental_id for event in events})
    confirmed_damage = [event for event in events if event.event_type == "damage_confirmed" and event.verified]
    confirmed_cleanliness = [
        event for event in events if event.event_type == "cleanliness_confirmed" and event.verified
    ]
    completed_review_ids = {event.source_id for event in events if event.event_type == "manual_review_completed"}
    reviewed_alert_ids = {
        event.source_id for event in events if event.event_type in {"cleanliness_confirmed", "cleanliness_reviewed"}
    }
    pending = [
        event
        for event in events
        if event.event_type == "automation_decision"
        and event.outcome == "manual_review"
        and event.source_id not in completed_review_ids
    ]
    unverified_alerts = [
        event
        for event in events
        if "alert_created" in event.event_type and not event.verified and event.source_id not in reviewed_alert_ids
    ]
    scope = period or "目前保存期間"
    if not events:
        return f"我沒有找到這位使用者在{scope}內的相關紀錄。"
    answer = (
        f"我在{scope}內找到 {len(events)} 筆紀錄，包含 {len(rental_ids)} 次租車。"
        f"已確認的車損有 {len(confirmed_damage)} 次，"
        f"已確認的髒污有 {len(confirmed_cleanliness)} 次，"
        f"還有 {len(pending)} 次需要工作人員再看一看。"
    )
    if unverified_alerts:
        answer += f"另外，電腦覺得有 {len(unverified_alerts)} 次可能有問題，但還沒有人確認，所以不會算成不好的紀錄。"
    answer += "這只是幫忙整理，不會自己決定扣錢或停止租車資格。"
    return answer


def _resolve_user_id(store: OperationsStore, request: HistoryQueryRequest) -> str | None:
    if request.user_id:
        profile = store.get_user_history_profile(request.user_id, event_limit=1)
        if profile:
            return profile.user_id
        from app.services.history_intent import catalogue

        return request.user_id if request.user_id in catalogue(store)[0] else None
    users = store.list_history_users(limit=None)
    candidates = []
    for user in users:
        identifiers = [user.user_id, *user.rental_ids, *user.vehicle_ids]
        if user.display_name != user.user_id:
            identifiers.append(user.display_name)
        if any(identifier and identifier.lower() in request.question.lower() for identifier in identifiers):
            candidates.append(user.user_id)
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        return None
    return users[0].user_id if len(users) == 1 else None


def _query_resolved_history(
    request: HistoryQueryRequest,
    *,
    store: OperationsStore | None = None,
) -> HistoryQueryResponse:
    store = store or get_operations_store()
    settings = get_settings()
    if request.license_plate:
        from app.services.vehicle_history import query_vehicle_history

        return query_vehicle_history(store, request)
    if "清潔費" in request.question:
        return HistoryQueryResponse(
            question=request.question,
            user_id=request.user_id,
            query_mode="sql_plus_retrieval",
            answer="目前提供垃圾紀錄查詢，可以問「他曾經留下垃圾嗎？」或「2026一月的垃圾紀錄？」。",
            generated_by="template",
        )
    user_id = None if request.all_users else _resolve_user_id(store, request)
    if user_id is None and not request.all_users:
        return HistoryQueryResponse(
            question=request.question,
            query_mode="sql_plus_retrieval",
            answer="請先選一位使用者，再問我問題。",
            generated_by="template",
            warnings=["user_not_resolved"],
        )
    try:
        since, until, period = _period_from_question(request.question)
    except ValueError:
        return HistoryQueryResponse(
            question=request.question,
            user_id=user_id,
            query_mode="sql_plus_retrieval",
            answer="日期格式不正確，請使用例如 2026年1月 或 2026-01-15。",
            generated_by="template",
            warnings=["invalid_time_filter"],
        )
    if request.date_from is not None or request.date_to is not None:
        since = datetime.combine(request.date_from, datetime.min.time(), tzinfo=_TAIPEI) if request.date_from else None
        until = datetime.combine(request.date_to, datetime.min.time(), tzinfo=_TAIPEI) if request.date_to else None
        period = (
            f"{request.date_from or '起始'} 至 {(request.date_to - timedelta(days=1)) if request.date_to else '目前'}"
        )
    event_types, category_labels = _event_types(request.question)
    events = store.list_rental_events(
        user_id=user_id,
        limit=None,
        since=since.isoformat() if since else None,
        until=until.isoformat() if until else None,
        event_types=event_types,
    )
    if request.user_ids:
        from app.services.history_intent import catalogue

        _, rentals = catalogue(store)
        allowed = {rid for rid, (uid, _) in rentals.items() if uid in request.user_ids}
        events = [event for event in events if event.rental_id in allowed]
    elif request.all_users:
        from app.services.history_intent import catalogue

        _, rentals = catalogue(store)
        events = [event for event in events if event.rental_id in rentals]
    if request.rental_ids:
        events = [event for event in events if event.rental_id in request.rental_ids]
    lowered = request.question.lower()
    if "垃圾" in category_labels and not any(label in category_labels for label in {"車損", "整潔", "賠償"}):
        events = [event for event in events if is_litter_event(event)]
    if any(term in lowered for term in _DISPUTE_TERMS):
        events = [
            event
            for event in events
            if event.disputed or event.event_type in {"dispute_opened", "dispute_resolved", "decision_reversed"}
        ]
    if "借車" in lowered and "還車" not in lowered:
        events = [event for event in events if event.stage in {None, "pickup"}]
    elif "還車" in lowered and "借車" not in lowered:
        events = [event for event in events if event.stage in {None, "return"}]
    warnings: list[str] = []
    statistics = {**compensation_statistics(events), **litter_statistics(events)}
    physical_damage = {
        event.metadata.get("damage_case_id") or event.source_id
        for event in events
        if event.verified and event.event_type in {"damage_confirmed", "damage_liability_recorded"}
    }
    statistics["confirmed_damage_event_count"] = len(physical_damage)
    cleanliness = {
        event.metadata.get("cleanliness_alert_id") or event.source_id
        for event in events
        if event.verified and event.event_type in {"cleanliness_confirmed", "litter_recorded"}
    }
    statistics["confirmed_cleanliness_event_count"] = len(cleanliness)
    statistics["matched_event_count"] = len(events)
    if user_id:
        profile = store.get_user_history_profile(user_id)
        if profile:
            statistics.update(blacklisted=profile.blacklisted, blacklist_reason=profile.blacklist_reason)
    matched = _rank_events(request.question, events, settings.history_rag_max_events)
    cited = matched
    generated_by = "template"
    answer = _template_answer(events, period)
    if request.all_users:
        answer = answer.replace("這位使用者", "全部使用者")
    structured_query = any(
        term in lowered for term in _DAMAGE_TERMS | _CLEAN_TERMS | _LITTER_TERMS | _COMPENSATION_TERMS | {"黑名單"}
    )
    if structured_query:
        scope = "全部使用者" if request.all_users else "這位使用者"
        answer = f"{scope}在{period or '目前保存期間'}內，"
        show_damage = (
            "車損" in category_labels or "賠償" in category_labels or not category_labels and "黑名單" in lowered
        )
        show_litter = "垃圾" in category_labels or not category_labels and "黑名單" in lowered
        if show_damage:
            answer += (
                f"有 {len(physical_damage)} 次已確認車損事件，"
                f"其中已確認使用者責任且無爭議的車損有 {statistics['caused_damage_count']} 次。"
                f"車損應賠 NT$ {statistics['damage_compensation_amount']}，"
                f"已賠 NT$ {statistics['damage_paid_amount']}，"
                f"未賠 NT$ {statistics['damage_outstanding_amount']}。"
            )
        if "整潔" in category_labels and "垃圾" not in category_labels:
            answer += f"有 {len(cleanliness)} 次已確認髒污事件。"
        if show_litter:
            answer += (
                f"已確認留下垃圾的紀錄有 {statistics['litter_count']} 次，"
                f"尚待確認 {statistics['pending_litter_count']} 次，"
                f"爭議中 {statistics['disputed_litter_count']} 次。"
            )
        if show_damage:
            answer += (
                f"有賠償金額的紀錄 {statistics['compensation_count']} 筆，"
                f"應賠 NT$ {statistics['compensation_amount']}，"
                f"已賠 NT$ {statistics['paid_amount']}，"
                f"未賠 NT$ {statistics['outstanding_amount']}。"
            )
        if user_id and (show_damage or "黑名單" in lowered) and "blacklisted" not in statistics:
            answer += "目前帳號已不存在，無法確認黑名單狀態。"
        elif user_id and (show_damage or "黑名單" in lowered):
            answer += (
                f"目前已列入黑名單，原因：{statistics['blacklist_reason']}。"
                if statistics.get("blacklisted")
                else "目前未列入黑名單。"
            )
        if request.all_users and "黑名單" in lowered:
            blacklisted = [
                user
                for user in store.list_history_users(limit=None)
                if user.blacklisted and (not request.user_ids or user.user_id in request.user_ids)
            ]
            statistics["blacklisted_user_count"] = len(blacklisted)
            answer += f"目前黑名單共 {len(blacklisted)} 位使用者。"
            for user in blacklisted[:20]:
                answer += f"\n{user.display_name}：{user.blacklist_reason}"
        details = [
            event
            for event in matched
            if event.event_type
            in {"damage_confirmed", "damage_liability_recorded", "cleanliness_confirmed", "litter_recorded"}
        ]
        if "垃圾" in category_labels and not show_damage:
            details = [event for event in matched if is_litter_event(event)]
        names = (
            {user.user_id: user.display_name for user in store.list_history_users(limit=None)}
            if request.all_users and details
            else {}
        )
        for event in details[:10]:
            renter_prefix = (
                (event.metadata.get("renter_name") or names.get(event.user_id, "使用者")) + "｜"
                if request.all_users
                else ""
            )
            answer += (
                f"\n{event.occurred_on}｜"
                f"{renter_prefix}"
                f"{event.title}"
                f"{'（爭議中）' if event.disputed else ''}：{event.summary}"
            )
        if not events:
            answer += "沒有找到符合條件的事件。"
        if show_damage:
            answer += "\n車損賠償統計只計入已確認責任且無爭議的紀錄。"
        if show_litter:
            answer += "\n尚待確認或爭議中的垃圾紀錄，不計入已確認次數。"
        if "黑名單" in lowered:
            answer += "\n黑名單狀態為目前狀態。"
    response = HistoryQueryResponse(
        question=request.question,
        user_id=user_id,
        query_mode="sql_plus_retrieval",
        answer=answer,
        generated_by=generated_by,
        interpreted_filters={
            "period": period,
            "since": since.isoformat() if since else None,
            "until": until.isoformat() if until else None,
            "all_users": request.all_users,
            "date_basis": "事件日期／租還車日期／台灣紀錄日期",
            "categories": category_labels,
            "event_types": event_types or [],
        },
        citations=[
            HistoryCitation(
                event_id=event.id,
                rental_id=event.rental_id,
                source_type=event.source_type,
                source_id=event.source_id,
                created_at=event.created_at,
            )
            for event in cited
        ],
        matched_events=matched,
        warnings=warnings,
        statistics=statistics,
    )
    response._all_events = events
    return response


def query_history_data(request: HistoryQueryRequest, *, store: OperationsStore | None = None) -> HistoryQueryResponse:
    """Single conversational entry point, shared by user and plate history queries."""
    store = store or get_operations_store()
    # Preserve callers that already supply a validated explicit scope.
    if request.user_id or request.all_users or request.license_plate:
        return _query_resolved_history(request, store=store)
    from app.services.history_intent import resolve

    result = resolve(request, store, use_model=False)
    if isinstance(result, HistoryQueryResponse):
        return result
    context, question, users, rentals = result
    common = dict(
        question=question, date_from=context.date_from, date_to=context.date_to, rental_ids=context.rental_ids
    )
    if (
        not context.license_plates
        and context.topics == ["history"]
        and any(term in request.question for term in ("最近一次", "最後一次", "上一次"))
    ):
        from app.services.history_intent import rental_details

        choices = [
            (rid, row)
            for rid, row in rental_details(store).items()
            if rid in rentals
            and (not context.user_ids or rentals[rid][0] in context.user_ids)
            and (not context.rental_ids or rid in context.rental_ids)
            and (
                not context.date_from
                or str(row.get("return_date") or row.get("rental_date") or "") >= str(context.date_from)
            )
            and (
                not context.date_to
                or str(row.get("rental_date") or row.get("return_date") or "9999") < str(context.date_to)
            )
        ]
        if choices:
            common["rental_ids"] = [
                max(
                    choices,
                    key=lambda item: (str(item[1].get("return_date") or item[1].get("rental_date") or ""), item[0]),
                )[0]
            ]
    if context.license_plates:
        responses = [
            _query_resolved_history(
                HistoryQueryRequest(**common, license_plate=plate, renter_ids=context.user_ids), store=store
            )
            for plate in context.license_plates
        ]
        response = responses[0]
        if len(responses) > 1:
            response.answer = "\n\n".join(item.answer for item in responses)
            response.matched_events = [event for item in responses for event in item.matched_events]
            response._all_events = [event for item in responses for event in item._all_events]
            response._rental_rows = [row for item in responses for row in item._rental_rows]
            response.citations = [citation for item in responses for citation in item.citations]
            response.license_plate = None
            response.statistics = {"vehicles": {item.license_plate: item.statistics for item in responses}}
    else:
        ids = context.user_ids or sorted(users)
        response = _query_resolved_history(
            HistoryQueryRequest(
                **common, user_id=ids[0] if len(ids) == 1 else None, all_users=len(ids) != 1, user_ids=context.user_ids
            ),
            store=store,
        )
        if context.user_ids:
            names = [next((name for name in sorted(users[uid]) if name != uid), uid) for uid in ids]
            response.answer = response.answer.replace("這位使用者", "、".join(names)).replace(
                "全部使用者", "、".join(names)
            )
        if len(context.user_ids) > 1:
            parts = []
            for uid in ids:
                item = _query_resolved_history(HistoryQueryRequest(**common, user_id=uid, user_ids=[uid]), store=store)
                name = next((name for name in sorted(users[uid]) if name != uid), uid)
                parts.append(f"{name}：{item.answer}")
            response.answer = "\n\n".join(parts)
    if not context.license_plates and context.topics == ["history"]:
        from app.services.history_intent import rental_details

        details = rental_details(store)
        selected = [
            row
            for rid, row in details.items()
            if rid in rentals
            and (not context.user_ids or rentals[rid][0] in context.user_ids)
            and (not common["rental_ids"] or rid in common["rental_ids"])
            and (
                not context.date_from
                or str(row.get("return_date") or row.get("rental_date") or "") >= str(context.date_from)
            )
            and (
                not context.date_to
                or str(row.get("rental_date") or row.get("return_date") or "9999") < str(context.date_to)
            )
        ]
        selected.sort(key=lambda row: str(row.get("rental_date") or ""), reverse=True)
        if any(term in request.question for term in ("最近一次", "最後一次", "上一次")):
            selected = selected[:1]
        response.statistics["rental_count"] = len(selected)
        response._rental_rows = selected
        lines = [f"找到 {len(selected)} 次租借紀錄。"]
        for row in selected[:20]:
            lines.append(
                f"{row.get('display_name') or row.get('customer_id')}｜{row.get('license_plate') or '車牌未填'}｜"
                f"{row.get('rental_date') or '日期未填'} 借車 → {row.get('return_date') or '日期未填'} 還車｜"
                f"租單 {row['rental_id']}"
            )
        if len(selected) > 20:
            lines.append("此處列出最近 20 次，可指定月份繼續查詢。")
        response.answer = "\n".join(lines)
        if any(term in request.question for term in ("最近一次", "最後一次", "上一次")):
            selected_ids = {row["rental_id"] for row in selected}
            response.matched_events = [event for event in response.matched_events if event.rental_id in selected_ids]
            response.citations = [citation for citation in response.citations if citation.rental_id in selected_ids]
    response.question = request.question
    response.context = context
    response.interpreted_filters.update(
        scope=context.scope,
        user_ids=context.user_ids,
        license_plates=context.license_plates,
        rental_ids=context.rental_ids,
        topics=context.topics,
    )
    return response


def query_history(request: HistoryQueryRequest, *, store: OperationsStore | None = None) -> HistoryQueryResponse:
    """Non-streaming answer: the deterministic card, plus one verified summary when the model is free."""
    from app.services.history_answer import LLM_ERRORS, chosen_backend, complete_sync, prepare_answer

    store = store or get_operations_store()
    settings = get_settings()
    prepared = prepare_answer(request, store)
    response = prepared.response
    if not settings.history_rag_generation_enabled:
        response.warnings.append("llm_generation_disabled")
        return response
    if not prepared.needs_llm:
        return response
    if not _generation_slot.acquire(timeout=settings.history_llm_queue_seconds):
        response.warnings.append("ollama_busy_fallback")
        return response
    try:
        return complete_sync(prepared, settings, chosen_backend(request, response, settings))
    except LLM_ERRORS as exc:
        response.warnings.append(f"ollama_fallback:{type(exc).__name__}")
        return response
    finally:
        _generation_slot.release()
