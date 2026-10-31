"""Deterministic evidence: intent-aware ranking, the answer card, and the pack the model may rephrase.

Everything a user reads first is computed here, before (and without) any language
model: counts, amounts, who, when, and which records support it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal

from app.schemas.history import (
    CardMetric,
    ChipOption,
    EvidenceRow,
    HistoryAnswerCard,
    HistoryChatContext,
    HistoryEventDetail,
    HistoryQueryRequest,
    HistoryQueryResponse,
    PersonRow,
    RentalEvent,
    RiskRow,
    ScopeChip,
)
from app.services.history_event_detail import (
    TYPE_LABELS,
    build_event_detail,
    check_prior_damage,
    damage_case_id_for,
    describe,
    parse_damage_text,
    photo_source_for,
    responsibility_of,
)
from app.services.history_intent import TOPICS, catalogue, classify_intent, mentions, rental_details
from app.services.history_llm import DENIES_DAMAGE, EvidencePack
from app.services.history_records import compensation_statistics, is_litter_event, litter_statistics
from app.services.license_plate import normalize_plate

CARD_ROWS = 20
# Photo pairs shown inline in the chat; the rest are in the Excel export.
PHOTO_GROUPS = 10
PACK_ROWS = 8
# Process events that drown out what people ask about; shown only when the question is about reviews.
NOISE = {"automation_decision", "photo_accepted", "photo_retry_required", "inspection_started", "capture_completed",
         "manual_review_completed"}
REVIEW_WORDS = ("審核", "覆核", "重拍", "AI 判斷", "AI判斷", "自動判斷", "照片品質", "流程")
TOPIC_LABELS = {key: label for key, (label, _) in TOPICS.items()} | {"history": "租借歷程"}
KIND_LABELS = {"damage": "車損", "litter": "垃圾", "cleanliness": "髒污", "dispute": "爭議", "rental": "租借",
               "review": "AI 審核", "blacklist": "黑名單", "other": "其他"}
RESPONSIBILITY_LABELS = {"confirmed": "已確認", "pending": "待確認", "disputed": "爭議中", "ai_only": "AI 偵測未確認",
                         "not_applicable": "不適用"}
WANTED_KINDS = {"damage": {"damage"}, "compensation": {"damage"}, "litter": {"litter"},
                "cleanliness": {"cleanliness", "litter"}, "dispute": {"damage", "litter", "dispute"},
                "blacklist": {"damage", "litter"}}
RISK_LEVELS = {"high": ("高", 1.0), "watch": ("觀察", 0.6), "low": ("低", 0.2), "insufficient_data": ("樣本不足", 0.1)}


# ---------------------------------------------------------------- directory


class Directory:
    """Names and plates for people and rentals, read fresh for every question."""

    def __init__(self, store) -> None:
        self.users, self.rentals = catalogue(store)
        self.details = rental_details(store)

    def name(self, uid: str | None) -> str:
        # A known person whose name was entered the same as their number is shown by that number.
        names = self.users.get(uid or "", ())
        return next((name for name in sorted(names) if name != uid), uid if names else "使用者")

    def renter_id(self, event: RentalEvent) -> str:
        return (event.metadata.get("renter_id") or self.rentals.get(event.rental_id, (None, None))[0]
                or event.user_id)

    def renter_name(self, event: RentalEvent) -> str:
        return event.metadata.get("renter_name") or self.name(self.renter_id(event))

    def rental_dates(self, event: RentalEvent) -> tuple[date | None, date | None]:
        """Pickup and return dates of the event's rental, from the event snapshot or the rental account."""
        row = self.details.get(event.rental_id or "", {})
        return (_as_date(event.metadata.get("rental_date") or row.get("rental_date")),
                _as_date(event.metadata.get("return_date") or row.get("return_date")))

    def plate(self, event: RentalEvent) -> str | None:
        raw = event.metadata.get("license_plate") or self.rentals.get(event.rental_id, (None, ""))[1] or ""
        return normalize_plate(raw)

    def rental_names(self) -> dict[str, str]:
        return {rid: (row.get("display_name") or row.get("renter_name") or self.name(self.rentals.get(rid, ("",))[0]))
                for rid, row in self.details.items()}

    def known_names(self) -> set[str]:
        return {name for uid, names in self.users.items() for name in names if name != uid and len(name) >= 2}

    def ids(self) -> set[str]:
        return set(self.users) | set(self.rentals)


# ---------------------------------------------------------------- formatting


def money(value) -> str:
    amount = Decimal(str(value or 0))
    text = f"{amount:,.0f}" if amount == amount.to_integral_value() else f"{amount:,.2f}"
    return f"NT${text}"


def cents(value) -> str:
    return f"{Decimal(int(value or 0)) / 100:.2f}"


def period_label(context: HistoryChatContext | None, response: HistoryQueryResponse) -> str:
    if context is None:
        return response.interpreted_filters.get("period") or "目前保存期間"
    start, end = context.date_from, context.date_to
    if not start and not end:
        return "全部期間"
    if start and end and start.day == 1 and end.day == 1:
        months = (end.year - start.year) * 12 + end.month - start.month
        if months == 1:
            return f"{start.year}年{start.month}月"
        if months == 12 and start.month == 1:
            return f"{start.year}年"
    if start and end:
        return f"{start.isoformat()}～{(end - timedelta(days=1)).isoformat()}"
    if start and (start.month, start.day) == (1, 1) and start.year == date.today().year:
        return f"{start.year}年"  # 「今年」 has no end date: it runs to today
    return f"{start.isoformat()} 起" if start else f"{(end - timedelta(days=1)).isoformat()} 以前"


def _topics_from(response: HistoryQueryResponse) -> list[str]:
    if response.context and response.context.topics:
        return list(response.context.topics)
    labels = {"車損": "damage", "整潔": "cleanliness", "垃圾": "litter", "賠償": "compensation", "爭議": "dispute"}
    return [labels[label] for label in response.interpreted_filters.get("categories", []) if label in labels] or [
        "history"]


def _scope_of(response: HistoryQueryResponse) -> str:
    if response.context:
        return response.context.scope
    return "vehicle" if response.license_plate else "user" if response.user_id else "all"


def scope_label(response: HistoryQueryResponse, directory: Directory) -> str:
    context = response.context
    if context is None:
        if response.license_plate:
            return f"車牌 {response.license_plate}"
        return directory.name(response.user_id) if response.user_id else "全部使用者"
    parts = []
    if context.user_ids:
        parts.append("、".join(dict.fromkeys(directory.name(uid) for uid in context.user_ids)))
    if context.license_plates:
        parts.append("車牌 " + "、".join(context.license_plates))
    return "｜".join(parts) or "全部使用者"


# ---------------------------------------------------------------- rows and ranking


def row_kind(event: RentalEvent) -> str:
    kind = event.event_type
    if kind.startswith("damage_"):
        return "damage"
    if is_litter_event(event):
        return "litter"
    if kind.startswith("cleanliness") or kind == "manual_review_completed":
        return "cleanliness"
    if kind in {"dispute_opened", "dispute_resolved", "decision_reversed"}:
        return "dispute"
    if kind in {"rental_history_uploaded", "inspection_started", "capture_completed", "photo_accepted",
                "photo_retry_required"}:
        return "rental"
    return "review" if kind == "automation_decision" else "other"


def _counted(event: RentalEvent, responsibility: str) -> bool:
    return responsibility == "confirmed" and (event.event_type == "damage_liability_recorded"
                                              or row_kind(event) == "litter"
                                              or event.metadata.get("claim_enabled") is True)


def make_row(ref: int, event: RentalEvent, directory: Directory) -> EvidenceRow:
    kind = row_kind(event)
    responsibility = responsibility_of(event)
    parsed = parse_damage_text(f"{event.title}。{event.summary}", event.metadata) if kind == "damage" else {}
    liability = event.event_type == "damage_liability_recorded" or bool(event.metadata.get("claim_enabled"))
    pickup_on, returned_on = directory.rental_dates(event)
    return EvidenceRow(
        ref=ref, event_id=event.id, rental_id=event.rental_id, occurred_on=event.occurred_on,
        pickup_on=pickup_on, returned_on=returned_on,
        renter_name=directory.renter_name(event), license_plate=directory.plate(event), kind=kind,
        kind_label=KIND_LABELS[kind], title=event.title, summary=event.summary, responsibility=responsibility,
        counted=_counted(event, responsibility), present_at_pickup=parsed.get("present_at_pickup"),
        present_at_return=parsed.get("present_at_return"),
        compensation_amount=cents(event.metadata.get("compensation_cents")) if liability else None,
        paid_amount=cents(event.metadata.get("paid_cents")) if liability else None,
        photo_slots=(list(event.metadata["evidence_photos"]) if event.metadata.get("evidence_photos")
                     else ["pickup", "return"]),
        photo_source=photo_source_for(event), damage_case_id=damage_case_id_for(event),
    )


def _case_key(event: RentalEvent) -> tuple:
    kind = row_kind(event)
    if kind == "damage":
        return ("damage", damage_case_id_for(event) or f"{event.source_type}:{event.source_id}")
    if kind == "litter" or kind == "cleanliness":
        return ("clean", event.metadata.get("cleanliness_alert_id") or event.source_id)
    return ("event", event.id)


def _tier(event: RentalEvent, topics: list[str], reviews: bool) -> int | None:
    kind, responsibility = row_kind(event), responsibility_of(event)
    if not reviews and (event.event_type in NOISE or (
            event.event_type == "damage_reviewed" and event.outcome in {"dismissed", "retry_photos"})):
        return None
    if "dispute" in topics and (event.disputed or kind == "dispute"):
        return 0
    if "compensation" in topics and (
        event.event_type == "damage_liability_recorded" or event.metadata.get("claim_enabled")
    ):
        owed = int(event.metadata.get("compensation_cents", 0)) - int(event.metadata.get("paid_cents", 0))
        return 0 if responsibility == "confirmed" and owed > 0 else 1 if responsibility == "confirmed" else 3
    base = {"confirmed": 0, "pending": 2, "disputed": 3, "ai_only": 4}.get(responsibility, 5)
    if event.event_type == "damage_confirmed":
        base = 1
    if kind == "rental":
        base = 6
    if kind == "review":
        base = 7
    wanted = set().union(*(WANTED_KINDS.get(topic, set()) for topic in topics)) if topics != ["history"] else set()
    return base + (10 if wanted and kind not in wanted else 0)


def _bigrams(text: str) -> set[str]:
    from app.services.history_query import _tokens

    return _tokens(text)


def rank_for_intent(events: list[RentalEvent], question: str, topics: list[str],
                    directory: Directory) -> list[RentalEvent]:
    """One row per physical case, most relevant first; process noise only when asked about reviews."""
    reviews = any(word in question for word in REVIEW_WORDS)
    best: dict[tuple, tuple[int, RentalEvent]] = {}
    for event in events:
        tier = _tier(event, topics, reviews)
        if tier is None:
            continue
        key = _case_key(event)
        if key not in best or tier < best[key][0]:
            best[key] = (tier, event)
    query = _bigrams(question)

    def score(event: RentalEvent) -> float:
        plate = directory.plate(event)
        named = mentions(question, directory.renter_name(event)) or bool(
            plate and (mentions(question, plate) or mentions(question, plate.replace("-", ""))))
        return 5.0 * named + len(query & _bigrams(f"{event.title} {event.summary}"))

    # Best tier, then wording/name match, then newest first.
    ordered = sorted(best.values(), key=lambda item: (item[0], -score(item[1]),
                                                      -(item[1].occurred_on or date.min).toordinal()))
    return [event for _, event in ordered]


def person_label(person: PersonRow, *extra: str) -> str:
    """Disambiguate a display name using its account ID and associated vehicle."""
    parts = ([f"編號 {person.user_id}"] if person.user_id else []) + (
        [f"車牌 {'、'.join(person.license_plates)}"] if person.license_plates else []) + list(extra)
    return f"{person.name}（{'｜'.join(parts)}）" if parts else person.name


def people_breakdown(events: list[RentalEvent], directory: Directory) -> list[PersonRow]:
    groups: dict[str, list[RentalEvent]] = {}
    for event in events:
        if row_kind(event) in {"damage", "litter"}:
            groups.setdefault(directory.renter_id(event), []).append(event)
    people = []
    for uid, own in groups.items():
        money_stats, litter = compensation_statistics(own), litter_statistics(own)
        unconfirmed = {_case_key(event) for event in own if row_kind(event) == "damage"
                       and responsibility_of(event) in {"pending", "ai_only"}
                       and not (event.event_type == "damage_reviewed")}
        disputed = {_case_key(event) for event in own if event.disputed}
        if not (money_stats["caused_damage_count"] or litter["litter_count"] or unconfirmed or disputed
                or litter["pending_litter_count"]):
            continue
        plates = sorted({plate for event in own if (plate := directory.plate(event))})
        people.append(PersonRow(
            name=directory.renter_name(own[0]) if own[0].metadata.get("renter_name") else directory.name(uid),
            user_id=uid, license_plates=plates, caused_damage_count=money_stats["caused_damage_count"],
            unconfirmed_damage_count=len(unconfirmed - disputed), disputed_count=len(disputed),
            litter_count=litter["litter_count"], compensation_amount=money_stats["compensation_amount"],
            outstanding_amount=money_stats["outstanding_amount"]))
    people.sort(key=lambda person: (-person.caused_damage_count, -person.litter_count,
                                    -Decimal(person.outstanding_amount), person.name))
    return people


def scope_rentals(context: HistoryChatContext | None, directory: Directory) -> set[str]:
    rentals = set(directory.rentals)
    if context is None:
        return rentals
    if context.user_ids:
        rentals = {rid for rid in rentals if directory.rentals[rid][0] in context.user_ids}
    if context.license_plates:
        rentals = {rid for rid in rentals if directory.rentals[rid][1] in context.license_plates}
    if context.rental_ids:
        rentals &= set(context.rental_ids)
    return rentals


def focus_events(store, context: HistoryChatContext | None, directory: Directory) -> list[RentalEvent]:
    """Client-supplied focus ids, re-read and kept only when they belong to the validated scope."""
    if not context or not context.focus_event_ids:
        return []
    allowed = scope_rentals(context, directory)
    events = store.list_rental_events(user_id=None, limit=None, event_ids=list(context.focus_event_ids))
    order = {event_id: index for index, event_id in enumerate(context.focus_event_ids)}
    return sorted((event for event in events if event.rental_id in allowed), key=lambda event: order[event.id])


# ---------------------------------------------------------------- risk


def risk_rows(store, context: HistoryChatContext | None, directory: Directory) -> list[RiskRow]:
    if context and (context.user_ids or context.license_plates):
        uids = {directory.rentals[rid][0] for rid in scope_rentals(context, directory)}
    else:
        uids = set(directory.users)
    rows = []
    for uid in sorted(uids)[:60]:
        profile = store.get_user_history_profile(uid, event_limit=1)
        if profile is None:
            continue
        label, weight = RISK_LEVELS.get(profile.risk_level, (profile.risk_level, 0.1))
        if profile.risk_level in {"low", "insufficient_data"} and (profile.caused_damage_count or profile.litter_count):
            # Too few rentals to grade, but a confirmed incident still deserves a look.
            label, weight = f"{label}（有已確認事件）", 0.5
        rows.append(dict(profile=profile, label=label, weight=weight))
    if not rows:
        return []
    top_outstanding = max(Decimal(row["profile"].damage_outstanding_amount) for row in rows) or Decimal(1)
    top_damage = max(row["profile"].caused_damage_count for row in rows) or 1
    top_litter = max(row["profile"].litter_count for row in rows) or 1
    for row in rows:
        profile = row["profile"]
        row["score"] = round(min(1.0, 0.55 * row["weight"]
                                 + 0.2 * float(Decimal(profile.damage_outstanding_amount) / top_outstanding)
                                 + 0.15 * profile.caused_damage_count / top_damage
                                 + 0.1 * profile.litter_count / top_litter), 3)
    rows.sort(key=lambda row: (-row["weight"], -row["score"], -row["profile"].caused_damage_count,
                               row["profile"].display_name))
    plates = {uid: sorted({plate for rid, (owner, plate) in directory.rentals.items() if owner == uid and plate})
              for uid in uids}
    return [RiskRow(
        rank=index, name=row["profile"].display_name if row["profile"].display_name != row["profile"].user_id
        else directory.name(row["profile"].user_id), license_plates=plates.get(row["profile"].user_id, [])[:3],
        level=row["profile"].risk_level, level_label=row["label"], reasons=row["profile"].risk_reasons,
        rental_count=row["profile"].rental_count, caused_damage_count=row["profile"].caused_damage_count,
        litter_count=row["profile"].litter_count, disputed_count=row["profile"].disputed_count,
        outstanding_amount=f"{Decimal(row['profile'].damage_outstanding_amount):.2f}",
        blacklisted=row["profile"].blacklisted, score=row["score"],
    ) for index, row in enumerate(rows[:10], start=1)]


# ---------------------------------------------------------------- the evidence bundle


@dataclass
class Evidence:
    card: HistoryAnswerCard
    pack: EvidencePack | None
    events: dict[str, RentalEvent] = field(default_factory=dict)
    details: list[HistoryEventDetail] = field(default_factory=list)


def _stats(events: list[RentalEvent]) -> dict:
    stats = {**compensation_statistics(events), **litter_statistics(events)}
    disputed_cents = sum(int(event.metadata.get("compensation_cents", 0)) for event in events
                         if (event.event_type == "damage_liability_recorded"
                             or event.metadata.get("claim_enabled")) and event.disputed)
    unconfirmed = {_case_key(event) for event in events if row_kind(event) == "damage"
                   and responsibility_of(event) in {"pending", "disputed", "ai_only"}
                   and event.event_type != "damage_reviewed"}
    confirmed_cases = {_case_key(event) for event in events if event.event_type == "damage_liability_recorded"
                       and responsibility_of(event) == "confirmed"}
    stats.update(disputed_amount=cents(disputed_cents), unconfirmed_damage_count=len(unconfirmed - confirmed_cases),
                 confirmed_cleanliness_count=len({_case_key(event) for event in events
                                                  if event.event_type == "cleanliness_confirmed" and event.verified}))
    return stats


def build_evidence(request: HistoryQueryRequest, response: HistoryQueryResponse, store,
                   *, today: date, row_limit: int | None = CARD_ROWS) -> Evidence:
    """row_limit=None keeps every matching row (used by the Excel export)."""
    directory = Directory(store)
    context = response.context
    topics = _topics_from(response)
    scope = _scope_of(response)
    # Clarifications and input errors never reach retrieval, so they carry no interpreted filters.
    if response.clarification_options or not response.interpreted_filters:
        return Evidence(card=HistoryAnswerCard(intent="clarify", headline=response.answer), pack=None)
    intent = context.intent if context and context.intent else classify_intent(request.question, topics, scope)
    events = response._all_events or list(response.matched_events)
    stats = _stats(events)
    labels = dict(scope=scope_label(response, directory), period=period_label(context, response),
                  topic="、".join(TOPIC_LABELS.get(topic, topic) for topic in topics))
    builder = _Builder(request, response, store, directory, intent, topics, scope, events, stats, labels, today)
    builder.row_limit = row_limit
    return builder.build()


class _Builder:
    def __init__(self, request, response, store, directory, intent, topics, scope, events, stats, labels, today):
        self.request, self.response, self.store, self.directory = request, response, store, directory
        self.intent, self.topics, self.scope, self.events, self.stats = intent, topics, scope, events, stats
        self.labels, self.today = labels, today
        self.question = request.question
        self.lines: list[str] = []
        self.names: set[str] = set()
        self.confirmed: set[str] = set()
        self.plates: set[str] = set(response.context.license_plates if response.context else [])
        self.names.update(directory.name(uid) for uid in (response.context.user_ids if response.context else []))
        self.details: list[HistoryEventDetail] = []

    # ------------------------------------------------------------ entry

    def build(self) -> Evidence:
        handler = {"timeline": self._timeline, "event_detail": self._event_detail, "risk": self._risk,
                   "prior_damage": self._prior_damage, "blacklist": self._blacklist}.get(self.intent, self._events)
        card = handler()
        if (DENIES_DAMAGE.search(self.question)
                and any(row.kind == "damage" and (row.counted or row.present_at_return) for row in card.rows)):
            # 「為何是無損傷？」: correct the premise before anything else, with or without the model.
            card.headline = "紀錄中其實有車損：" + card.headline
        card.photo_limit = PHOTO_GROUPS
        card.photo_total = sum(1 for row in card.rows if row.photo_source)
        card.scope_label, card.period_label, card.topic_label = (self.labels["scope"], self.labels["period"],
                                                                 self.labels["topic"])
        card.chips = self._chips(card)
        card.followups = self._followups(card)
        events = {event.id: event for event in self.events}
        for row in card.rows:
            if row.renter_name and row.renter_name != "使用者":
                self.names.add(row.renter_name)
            if row.license_plate:
                self.plates.add(row.license_plate)
        pack = self._pack(card) if (card.rows or card.people or card.risk or card.prior_damage) else None
        return Evidence(card=card, pack=pack, events=events, details=self.details)

    # ------------------------------------------------------------ helpers

    row_limit: int | None = CARD_ROWS

    def _rows(self, events: list[RentalEvent], limit: int | None = -1) -> list[EvidenceRow]:
        stop = self.row_limit if limit == -1 else limit
        return [make_row(index, event, self.directory) for index, event in enumerate(events[:stop], start=1)]

    def _phrase(self) -> str:
        period = self.labels["period"]
        all_time = period in {"全部期間", "目前保存期間"}
        if self.labels["scope"] == "全部使用者":
            return "全部紀錄中" if all_time else period
        phrase = self.labels["scope"] if all_time else f"{self.labels['scope']}在{period}"
        # Keep whitespace between a vehicle identifier and the following count.
        return phrase + " " if phrase[-1:].isascii() and phrase[-1:].isalnum() else phrase

    def _metrics(self) -> list[CardMetric]:
        stats, metrics = self.stats, []
        if {"damage", "compensation", "dispute", "blacklist"} & set(self.topics):
            metrics.append(CardMetric(label="已確認責任車損", value=f"{stats['caused_damage_count']} 件",
                                      tone="bad" if stats["caused_damage_count"] else "good"))
            if stats["unconfirmed_damage_count"]:
                metrics.append(CardMetric(label="待確認／爭議", value=f"{stats['unconfirmed_damage_count']} 件",
                                          tone="warn"))
        if "compensation" in self.topics or (
                "damage" in self.topics and Decimal(self.stats["compensation_amount"]) > 0):
            metrics += [CardMetric(label="應賠", value=money(stats["compensation_amount"])),
                        CardMetric(label="已賠", value=money(stats["paid_amount"]), tone="good"),
                        CardMetric(label="未賠", value=money(stats["outstanding_amount"]),
                                   tone="bad" if Decimal(stats["outstanding_amount"]) > 0 else "good")]
        if {"litter", "cleanliness"} & set(self.topics):
            metrics.append(CardMetric(label="已確認垃圾", value=f"{stats['litter_count']} 次",
                                      tone="bad" if stats["litter_count"] else "good"))
            pending = stats["pending_litter_count"] + stats["disputed_litter_count"]
            if pending:
                metrics.append(CardMetric(label="垃圾待確認／爭議", value=f"{pending} 次", tone="warn"))
        return metrics

    def _topic_sentences(self, people: list[PersonRow]) -> list[str]:
        stats, phrase, out = self.stats, self._phrase(), []
        listing = self.intent == "list_users"
        if "compensation" in self.topics:
            sentence = (f"{phrase}已確認責任的索賠金額 {money(stats['compensation_amount'])}，"
                        f"已賠 {money(stats['paid_amount'])}，未賠 {money(stats['outstanding_amount'])}")
            if Decimal(stats["disputed_amount"]) > 0:
                sentence += f"；爭議中的 {money(stats['disputed_amount'])} 不計入"
            out.append(sentence + "。")
            if listing:
                owing = [person for person in people if Decimal(person.outstanding_amount) > 0]
                if owing:
                    out.append("尚未付清：" + "、".join(
                        person_label(person, f"未賠 {money(person.outstanding_amount)}")
                        for person in owing[:5]) + "。")
        elif "dispute" in self.topics:
            disputed = [person for person in people if person.disputed_count]
            count = len({_case_key(event) for event in self.events if event.disputed})
            names = "、".join(person_label(person) for person in disputed)
            out.append(f"{phrase}爭議中的紀錄有 {count} 件" + (f"：{names}" if names else "")
                       + "；爭議結案前不計入責任與賠償。")
        elif "damage" in self.topics:
            caused = [person for person in people if person.caused_damage_count]
            if listing:
                sentence = (f"{phrase}被確認造成車損的使用者有 {len(caused)} 位："
                            + "、".join(person_label(person) for person in caused) if caused
                            else f"{phrase}沒有已確認責任的車損紀錄")
            else:
                sentence = (f"{phrase}已確認使用者責任車損 {stats['caused_damage_count']} 件"
                            if stats["caused_damage_count"] else f"{phrase}沒有已確認責任的車損紀錄")
            if stats["unconfirmed_damage_count"]:
                sentence += f"；另有 {stats['unconfirmed_damage_count']} 件待確認或爭議中，不計入"
            out.append(sentence + "。")
        if {"litter", "cleanliness"} & set(self.topics):
            littered = [person for person in people if person.litter_count]
            if listing and littered:
                sentence = f"{phrase}已確認留下垃圾的使用者：" + "、".join(person_label(person) for person in littered)
            else:
                sentence = f"{phrase}已確認留下垃圾 {stats['litter_count']} 次"
            extra = [f"待確認 {stats['pending_litter_count']} 次"] if stats["pending_litter_count"] else []
            extra += [f"爭議中 {stats['disputed_litter_count']} 次"] if stats["disputed_litter_count"] else []
            out.append(sentence + ("；" + "、".join(extra) if extra else "") + "。")
            if "cleanliness" in self.topics and stats["confirmed_cleanliness_count"]:
                out.append(f"已確認髒污 {stats['confirmed_cleanliness_count']} 次。")
        return out

    # ------------------------------------------------------------ intents

    def _events(self) -> HistoryAnswerCard:
        ranked = rank_for_intent(self.events, self.question, self.topics, self.directory)
        people = people_breakdown(self.events, self.directory) if self.scope not in {"user"} else []
        sentences = self._topic_sentences(people)
        if not sentences:
            sentences = [f"{self._phrase()}找到 {len(ranked)} 筆相關紀錄。" if ranked
                         else f"{self._phrase()}沒有找到符合條件的紀錄。"]
        for person in people:
            self.names.add(person.name)
            if person.caused_damage_count or person.litter_count:
                self.confirmed.add(person.name)
            self.plates.update(person.license_plates)
        return HistoryAnswerCard(intent=self.intent, headline="".join(sentences), metrics=self._metrics(),
                                 rows=self._rows(ranked), total_rows=len(ranked), people=people[:12],
                                 people_columns=self._people_columns())

    def _people_columns(self) -> list[str]:
        topics = set(self.topics)
        columns = (["damage"] if {"damage", "compensation", "dispute", "blacklist"} & topics else []) + (
            ["litter"] if {"litter", "cleanliness"} & topics else [])
        return columns or ["damage", "litter"]

    def _timeline(self) -> HistoryAnswerCard:
        rows_data = sorted(self.response._rental_rows, key=lambda row: (str(row.get("rental_date") or ""),
                                                                         str(row.get("return_date") or "")),
                           reverse=True)
        by_rental: dict[str, RentalEvent] = {}
        for event in self.events:
            if event.event_type == "rental_history_uploaded" or event.rental_id not in by_rental:
                by_rental[event.rental_id] = event
        rows: list[EvidenceRow] = []
        if rows_data:
            for index, row in enumerate(rows_data[:self.row_limit], start=1):
                event = by_rental.get(row["rental_id"])
                name = row.get("display_name") or row.get("renter_name") or self.directory.name(row.get("customer_id"))
                rows.append(EvidenceRow(
                    ref=index, event_id=event.id if event else None, rental_id=row["rental_id"],
                    occurred_on=_as_date(row.get("rental_date")), return_date=_as_date(row.get("return_date")),
                    renter_name=name, license_plate=normalize_plate(row.get("license_plate") or ""), kind="rental",
                    kind_label="租借", title="租借紀錄",
                    summary=(f"{row.get('rental_date') or '日期未填'} 借車 → "
                             f"{row.get('return_date') or '日期未填'} 還車"),
                    photo_source=photo_source_for(event) if event else None))
            total = len(rows_data)
        else:
            events = sorted((event for event in self.events if event.event_type not in {"automation_decision"}),
                            key=lambda event: (event.occurred_on or date.min, event.created_at), reverse=True)
            rows = self._rows(events)
            total = len(events)
        first = rows[0] if rows else None
        if first and first.kind == "rental":
            headline = (f"{self._phrase()}有 {total} 次租借紀錄；最近一次是 {first.renter_name} "
                        f"{first.occurred_on or '日期未填'} 借車、{first.return_date or '日期未填'} 還車"
                        + (f"（車牌 {first.license_plate}）" if first.license_plate else "") + "。")
        elif first:
            headline = f"{self._phrase()}找到 {total} 筆紀錄；最近一筆是 {first.occurred_on} 的「{first.title}」。"
        else:
            headline = f"{self._phrase()}沒有找到租借紀錄。"
        return HistoryAnswerCard(intent="timeline", headline=headline, rows=rows, total_rows=total,
                                 metrics=[CardMetric(label="租借次數" if rows_data else "紀錄", value=f"{total} 筆")])

    def _detail_targets(self) -> list[RentalEvent]:
        context = self.response.context
        focus = focus_events(self.store, context, self.directory)
        if context and context.user_ids:
            focus = [event for event in focus if self.directory.renter_id(event) in context.user_ids] or focus
        damage = [event for event in focus if row_kind(event) == "damage"]
        if damage or focus:
            return (damage or focus)[:3]
        ranked = rank_for_intent(self.events, self.question, self.topics or ["damage"], self.directory)
        return [event for event in ranked if row_kind(event) == "damage"][:3] or ranked[:1]

    def _event_detail(self) -> HistoryAnswerCard:
        targets = self._detail_targets()
        if not targets:
            return HistoryAnswerCard(intent="event_detail", headline="找不到可以說明的紀錄，請先查詢一筆車損紀錄。")
        rows = self._rows(targets, limit=3)
        self.details = [build_event_detail(self.store, event, row.renter_name, row.license_plate)
                        for event, row in zip(targets, rows, strict=False)]
        first, row = self.details[0], rows[0]
        what = "".join(part for part in (first.location, TYPE_LABELS.get(first.damage_type or "")) if part) or "此損傷"
        headline = f"{row.renter_name} {row.occurred_on or ''} 的紀錄是{what}：{first.conclusion}。"
        if row.renter_name != "使用者":
            self.names.add(row.renter_name)
            if row.counted:
                self.confirmed.add(row.renter_name)
        return HistoryAnswerCard(intent="event_detail", headline=headline.replace("  ", " "), rows=rows,
                                 total_rows=len(rows), detail_event_ids=[event.id for event in targets])

    def _prior_damage(self) -> HistoryAnswerCard:
        targets = self._detail_targets()
        if not targets:
            return HistoryAnswerCard(intent="prior_damage", headline="找不到要追溯的車損紀錄，請先查詢一筆車損紀錄。")
        focus = targets[0]
        check = check_prior_damage(self.store, focus, self.directory.details, self.directory.rental_names())
        rows = self._rows([focus], limit=1)
        check.renter_name = rows[0].renter_name
        for rental in check.earlier_rentals:
            self.names.add(rental.renter_name)
        self.names.add(rows[0].renter_name)
        if check.license_plate:
            self.plates.add(check.license_plate)
        return HistoryAnswerCard(intent="prior_damage", headline=check.verdict_text, rows=rows, total_rows=1,
                                 prior_damage=check, detail_event_ids=[focus.id])

    def _risk(self) -> HistoryAnswerCard:
        rows = risk_rows(self.store, self.response.context, self.directory)
        flagged = [row for row in rows if row.level in {"high", "watch"}]
        noted = [row for row in rows if row.level not in {"high", "watch"}
                 and (row.caused_damage_count or row.litter_count)]
        if flagged:
            ranked = [f"{row.name}（{row.level_label}風險）" for row in flagged[:2]]
            headline = "最需要注意的是" + "，其次是".join(ranked) + "。"
        elif noted:
            headline = ("目前沒有高風險使用者；" + "、".join(
                f"{row.name}有已確認責任車損 {row.caused_damage_count} 件" if row.caused_damage_count
                else f"{row.name}有已確認垃圾 {row.litter_count} 次" for row in noted[:3])
                + "，但租借次數不足以分級，建議查看紀錄。")
        elif rows:
            headline = "目前沒有使用者有已確認的車損或垃圾事件，風險皆為低或樣本不足。"
        else:
            headline = "找不到可以評估風險的使用者。"
        if any(row.disputed_count for row in flagged[:3]):
            headline += "有爭議中的紀錄者，爭議結案前不得作不利處分。"
        for row in rows:
            self.names.add(row.name)
            if row.caused_damage_count or row.litter_count:
                self.confirmed.add(row.name)
        metrics = [CardMetric(label="高風險", value=f"{sum(row.level == 'high' for row in rows)} 位", tone="bad"),
                   CardMetric(label="觀察", value=f"{sum(row.level == 'watch' for row in rows)} 位", tone="warn")]
        self.labels["period"] = "全部期間"
        return HistoryAnswerCard(intent="risk", headline=headline, risk=rows, metrics=metrics)

    def _blacklist(self) -> HistoryAnswerCard:
        context = self.response.context
        allowed = {self.directory.rentals[rid][0] for rid in scope_rentals(context, self.directory)}
        listed = [user for user in self.store.list_history_users(limit=None)
                  if user.blacklisted and user.user_id in allowed]
        rows = [EvidenceRow(ref=index, renter_name=user.display_name, kind="blacklist", kind_label="黑名單",
                            title="列入黑名單", summary=user.blacklist_reason or "未填原因")
                for index, user in enumerate(listed[:self.row_limit], start=1)]
        if self.scope == "user" and context and context.user_ids:
            name = self.directory.name(context.user_ids[0])
            user = next((user for user in listed if user.user_id == context.user_ids[0]), None)
            headline = (f"{name}目前已列入黑名單，原因：{user.blacklist_reason}。" if user
                        else f"{name}目前未列入黑名單。")
        else:
            headline = (f"目前黑名單共 {len(listed)} 位：" + "、".join(user.display_name for user in listed[:10]) + "。"
                        if listed else "目前沒有使用者列入黑名單。")
        headline += "黑名單為目前狀態。"
        for row in rows:
            self.names.add(row.renter_name)
        return HistoryAnswerCard(intent="blacklist", headline=headline, rows=rows, total_rows=len(listed),
                                 metrics=[CardMetric(label="黑名單", value=f"{len(listed)} 位",
                                                     tone="bad" if listed else "good")])

    # ------------------------------------------------------------ chips and follow-ups

    def _chips(self, card: HistoryAnswerCard) -> list[ScopeChip]:
        if card.intent in {"clarify"}:
            return []
        period = ScopeChip(kind="period", label=self.labels["period"], options=[
            ChipOption(label="本月", question="改查這個月"), ChipOption(label="上個月", question="改查上個月"),
            ChipOption(label="今年", question="改查今年"), ChipOption(label="全部期間", question="改查全部期間")])
        topic = ScopeChip(kind="topic", label=self.labels["topic"] or "租借歷程", options=[
            ChipOption(label="車損", question="改查車損"), ChipOption(label="賠償金額", question="改查賠償金額"),
            ChipOption(label="垃圾", question="改查垃圾"), ChipOption(label="爭議", question="改查爭議"),
            ChipOption(label="租借歷程", question="改查租借紀錄")])
        options = [ChipOption(label="全部使用者", question="改查全部使用者")] if self.scope != "all" else []
        names = [person.name for person in card.people[:4]] or [row.name for row in card.risk[:4]]
        duplicates = {name for name in names if sum(name in aliases for aliases in self.directory.users.values()) > 1}
        for person in card.people[:4]:
            if person.name in duplicates and person.license_plates:
                options.append(ChipOption(label=f"{person.name}（{person.license_plates[0]}）",
                                          question=f"改查車牌 {person.license_plates[0]}"))
            elif person.name not in duplicates:
                options.append(ChipOption(label=person.name, question=f"改查{person.name}"))
        scope = ScopeChip(kind="scope", label=self.labels["scope"], options=options)
        return [scope, period, topic]

    def _followups(self, card: HistoryAnswerCard) -> list[str]:
        out: list[str] = []
        topics = set(self.topics)
        photo_row = next((row for row in card.rows if row.kind == "damage" and row.photo_source), None)
        if card.intent == "list_users":
            if {"damage", "dispute"} & topics:
                out.append("他們總共要賠多少錢？")
            if photo_row:
                out.append("照片的損傷是什麼？")
            if any(person.disputed_count for person in card.people):
                out.append("有爭議的有哪些？")
            out.append("誰最需要注意？")
        elif card.intent == "aggregate":
            if "compensation" in topics:
                out += ["哪些人還沒付清？", "改查上個月"]
            elif "damage" in topics and card.rows:
                out += ["照片的損傷是什麼？", "總共要賠多少錢？"]
            elif {"litter", "cleanliness"} & topics:
                out.append("有哪些人留下垃圾？")
        elif card.intent == "event_detail":
            out += ["這個損傷之前就有嗎？", "總共要賠多少錢？"]
        elif card.intent == "prior_damage":
            out += ["照片的損傷是什麼？", "這台車的租借紀錄"]
        elif card.intent == "risk" and card.risk:
            top = card.risk[0].name
            out += [f"{top}的車損紀錄", f"{top}還欠多少錢？"]
        elif card.intent == "timeline" and card.rows:
            out += ["有沒有車損？", "有沒有留下垃圾？"]
        return list(dict.fromkeys(out))[:4]

    # ------------------------------------------------------------ the model's pack

    def _pack(self, card: HistoryAnswerCard) -> EvidencePack:
        topic = self.labels["topic"] or "租借歷程"
        lines = [f"範圍：{self.labels['scope']}｜期間：{self.labels['period']}｜主題：{topic}",
                 f"重點：{card.headline}"]
        stats = self.stats
        if card.intent in {"aggregate", "list_users"}:
            parts = []
            if {"damage", "compensation", "dispute"} & set(self.topics):
                parts.append(f"已確認使用者責任車損 {stats['caused_damage_count']} 件")
                parts.append(f"待確認或爭議中 {stats['unconfirmed_damage_count']} 件（不計入）")
            if "compensation" in self.topics or Decimal(stats["compensation_amount"]) > 0:
                parts += [f"應賠 {money(stats['compensation_amount'])}", f"已賠 {money(stats['paid_amount'])}",
                          f"未賠 {money(stats['outstanding_amount'])}"]
                if Decimal(stats["disputed_amount"]) > 0:
                    parts.append(f"爭議中金額 {money(stats['disputed_amount'])}（不計入）")
            if {"litter", "cleanliness"} & set(self.topics):
                parts += [f"已確認留下垃圾 {stats['litter_count']} 次",
                          f"垃圾待確認 {stats['pending_litter_count']} 次",
                          f"垃圾爭議中 {stats['disputed_litter_count']} 次"]
            if card.people:
                parts.append(f"涉及使用者 {len(card.people)} 位")
            if parts:
                lines.append("統計：" + "；".join(parts))
            if card.people and self.scope != "user":
                lines.append("每人：")
                for person in card.people[:8]:
                    who = ([f"編號 {person.user_id}"] if person.user_id else []) + (
                        [f"車牌 {'、'.join(person.license_plates)}"] if person.license_plates else [])
                    plate = f"（{'，'.join(who)}）" if who else ""
                    facts = []
                    if person.caused_damage_count:
                        facts.append(f"責任車損 {person.caused_damage_count} 件")
                    if person.unconfirmed_damage_count:
                        facts.append(f"待確認 {person.unconfirmed_damage_count} 件（不計入）")
                    if person.disputed_count:
                        facts.append(f"爭議中 {person.disputed_count} 件（不計入）")
                    if person.litter_count:
                        facts.append(f"留下垃圾 {person.litter_count} 次")
                    if Decimal(person.outstanding_amount) > 0:
                        facts.append(f"未賠 {money(person.outstanding_amount)}")
                    lines.append(f"- {person.name}{plate}：" + "；".join(facts or ["無已確認紀錄"]))
                # Who has the most, counting only what this answer covers: a damage-only answer has no litter.
                damage = bool({"damage", "compensation", "dispute"} & set(self.topics))
                litter = bool({"litter", "cleanliness"} & set(self.topics))
                label, unit = (("不良紀錄最多", "已確認車損與垃圾合計 {} 次") if damage and litter
                               else ("留下垃圾最多", "已確認留下垃圾 {} 次") if litter
                               else ("責任車損最多", "已確認責任車損 {} 件"))
                totals = {person.name: person.caused_damage_count * (damage or not litter)
                          + person.litter_count * litter for person in card.people}
                top = max(totals.values(), default=0)
                if top and len(totals) > 1:
                    leaders = "、".join(name for name, total in totals.items() if total == top)
                    lines.append(f"{label}：{leaders}（{unit.format(top)}）")
        refs: list[str | None] = []
        if card.risk:
            lines.append("風險排名（[編號] 姓名｜風險等級｜責任車損｜垃圾｜未賠金額｜爭議｜原因）：")
            for row in card.risk[:PACK_ROWS]:
                lines.append(f"[{row.rank}] {row.name}｜{row.level_label}｜{row.caused_damage_count} 件｜"
                             f"{row.litter_count} 次｜{money(row.outstanding_amount)}｜{row.disputed_count} 件｜"
                             + "；".join(row.reasons))
                refs.append(None)
        if self.details:
            for row, detail in zip(card.rows, self.details, strict=False):
                lines.append(f"照片判讀 [{row.ref}]：{describe(detail)}")
                lines.append(f"結論 [{row.ref}]：{detail.conclusion}")
        if card.prior_damage:
            check = card.prior_damage
            lines.append(f"既有損傷追溯：部位 {check.location or '未記錄'}｜結論 {check.verdict_text}")
            for rental in check.earlier_rentals:
                score = f"｜影像相似分數 {rental.visual_score:.2f}" if rental.visual_score is not None else ""
                period = f"{rental.rental_date or ''}～{rental.return_date or ''}"
                lines.append(f"- 較早租借：{rental.renter_name} {period}｜"
                             f"相同部位紀錄：{'有' if rental.same_location else '無'}{score}｜"
                             + ("；".join(rental.records) or "無車損紀錄"))
        if card.rows and not card.risk:
            if card.intent == "timeline" and card.rows[0].kind == "rental":
                lines.append("紀錄（[編號] 借車日｜還車日｜承租人｜車牌）：")
                for row in card.rows[:PACK_ROWS]:
                    lines.append(f"[{row.ref}] {row.occurred_on or '未填'}｜{row.return_date or '未填'}｜"
                                 f"{row.renter_name}｜{row.license_plate or '未填'}")
            else:
                lines.append("紀錄（[編號] 日期｜承租人｜車牌｜租借期間｜類型｜責任｜"
                             "借車照片有此損傷｜還車照片有此損傷｜金額｜描述）：")
                for row in card.rows[:PACK_ROWS]:
                    lines.append(_pack_row(row))
            refs = [row.event_id for row in card.rows[:PACK_ROWS]]
        if card.rows and card.intent in {"event_detail", "prior_damage", "timeline", "blacklist"}:
            lines.append(f"紀錄數：{card.total_rows or len(card.rows)} 筆")
        if card.total_rows > PACK_ROWS and not card.risk:
            lines.append(f"（另有 {card.total_rows - PACK_ROWS} 筆未列出）")
        names = {name for name in self.names if name and name != "使用者"}
        plates = {plate for plate in self.plates if plate}
        forbidden = self.directory.ids() | {value for event in self.events for value in (
            event.id, event.source_id, event.rental_id, damage_case_id_for(event)) if value}
        # The numbers of the people this answer lists are shown to the reader, so the summary may repeat them.
        shown_ids = {person.user_id for person in card.people if person.user_id}
        allowed = {value.casefold() for value in names | plates | shown_ids}
        facts = set()
        if any(row.kind == "damage" and (row.counted or row.present_at_return) for row in card.rows) or any(
                detail.present_at_return for detail in self.details):
            facts.add("damage_present")
        if Decimal(stats["outstanding_amount"]) > 0:
            facts.add("money_owed")
        if any(row.kind == "damage" and row.counted for row in card.rows):
            facts.add("someone_liable")
        if any(row.kind == "litter" for row in card.rows):
            facts.add("litter_present")
        if card.prior_damage and card.prior_damage.verdict == "already_present":
            facts.add("prior_present")
        elif card.prior_damage and card.prior_damage.verdict == "new_damage":
            facts.add("prior_absent")
        context = self.response.context
        return EvidencePack(
            facts=facts, periods=relative_period_keys(context.date_from, context.date_to, self.today)
            if context else set(),
            text="\n".join(lines), refs=refs, names=names, confirmed_names=self.confirmed | {
                row.renter_name for row in card.rows if row.counted}, plates=plates,
            known_names=self.directory.known_names(),
            forbidden={value for value in forbidden if value.casefold() not in allowed and len(value) >= 3},
            intent=card.intent)


def relative_period_keys(start: date | None, end: date | None, today: date) -> set[str]:
    """Which of 「今天／本月／今年…」 describe this period (end is exclusive)."""
    if start is None:
        return set()
    month = today.replace(day=1)
    next_month = (month + timedelta(days=32)).replace(day=1)
    last_month = (month - timedelta(days=1)).replace(day=1)
    year, next_year = date(today.year, 1, 1), date(today.year + 1, 1, 1)
    spans = {"today": [(today, today + timedelta(days=1))],
             "yesterday": [(today - timedelta(days=1), today)],
             "this_month": [(month, next_month), (month, None)],
             "last_month": [(last_month, month)],
             "this_year": [(year, next_year), (year, None), (year, today + timedelta(days=1))],
             "last_year": [(date(today.year - 1, 1, 1), year)]}
    return {key for key, options in spans.items() if (start, end) in options}


def _presence(value: bool | None) -> str:
    return "是" if value else "否" if value is False else "未記錄"


def _pack_row(row: EvidenceRow) -> str:
    amount = ""
    if row.compensation_amount is not None:
        amount = f"應賠 {money(row.compensation_amount)}、已賠 {money(row.paid_amount)}"
        if not row.counted:
            amount += "（不計入）"
    title = row.title.removeprefix("[假資料] ").removeprefix("[模擬資料] ")
    summary = row.summary.replace("\n", " ")[:80]
    presence = (f"借車照片有此損傷：{_presence(row.present_at_pickup)}｜"
                f"還車照片有此損傷：{_presence(row.present_at_return)}"
                if row.kind == "damage" else "借車照片有此損傷：不適用｜還車照片有此損傷：不適用")
    rental = (f"借車 {row.pickup_on or '未填'}、還車 {row.returned_on or '未填'}"
              if row.pickup_on or row.returned_on else "租借期間未填")
    return (f"[{row.ref}] {row.occurred_on or '日期未填'}｜{row.renter_name}｜{row.license_plate or '未填'}｜"
            f"{rental}｜{row.kind_label}｜責任：{RESPONSIBILITY_LABELS[row.responsibility]}｜{presence}｜"
            f"{amount or '無金額'}｜{title}：{summary}")


def _as_date(value) -> date | None:
    if value in (None, ""):
        return None
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def title_for(card: HistoryAnswerCard) -> str:
    period = "" if card.period_label in {"全部期間", "目前保存期間"} else card.period_label
    scope = "" if card.scope_label == "全部使用者" else card.scope_label.split("｜")[0]
    topic = card.topic_label.split("、")[0] if card.topic_label else "租借"
    title = {
        "list_users": f"{period}{topic}使用者",
        "aggregate": f"{scope}{period}{topic}統計",
        "event_detail": f"{scope or (card.rows[0].renter_name if card.rows else '')}{topic}照片判讀",
        "timeline": f"{scope}租借歷程",
        "risk": "風險使用者排名",
        "prior_damage": f"{(card.prior_damage.license_plate if card.prior_damage else '') or ''}既有損傷追溯",
        "blacklist": f"{scope}黑名單查詢",
    }.get(card.intent, "")
    return title[:20] or "租借紀錄查詢"
