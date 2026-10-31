"""「你可以做什麼？」: the assistant introduces what it can answer now, from live data and settings.

Every answer intent has an entry in CAPABILITIES (a test enforces this), so a new
intent cannot ship without being introduced. Example questions use real names and
plates from the data and are checked to resolve to their intent.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

import httpx

from app.schemas.history import CapabilityItem, CardMetric, HistoryAnswerCard, HistoryQueryRequest, HistoryQueryResponse
from app.services import history_llm
from app.services.history_event_detail import photo_source_for
from app.services.history_intent import catalogue, rental_details

_PREFIX = re.compile(r"^(?:請問|嗨|哈囉|你好|您好|hi|hello|hey|那|所以)+")
_SUFFIX = re.compile(r"(?:呢|啊|呀|嗎|吧|阿|一下)+$")
_SUBJECT = r"(?:你|妳|您|助理|這個助理|ai助理|借還車ai助理)?"
_WHAT = r"(?:些)?(?:什麼|甚麼|啥|哪些)(?:事|事情|功能|資料|問題|東西)?"
CAPABILITY_QUESTION = re.compile("|".join((
    rf"{_SUBJECT}(?:可以|能|會|能夠)(?:幫我|幫忙|替我)?(?:做|查|查詢|回答|處理|提供)?{_WHAT}",
    rf"我?(?:可以|能|要|該)?(?:問|查){_WHAT}",
    rf"{_SUBJECT}(?:有|提供)?(?:什麼|甚麼|哪些)功能|功能(?:介紹|說明|列表|有哪些)",
    rf"(?:要)?(?:怎麼|如何)(?:使用|用){_SUBJECT}|使用說明|教學",
    r"(?:請)?(?:你|妳)?(?:自我介紹|介紹(?:一下)?(?:你|妳)?自己?)",
    rf"{_SUBJECT}是(?:誰|什麼|甚麼)",
    r"help|幫助|說明",
)))


def is_capability_question(question: str) -> bool:
    """Only the whole question counts: 「你可以查哪些使用者造成車損」 is a data question."""
    text = re.sub(r"[\s？?！!。，,.~～…:：]", "", question).lower()
    text = _SUFFIX.sub("", _PREFIX.sub("", text))
    return bool(text) and CAPABILITY_QUESTION.fullmatch(text) is not None


@dataclass
class _Live:
    users: int
    rentals: int
    first: str | None
    last: str | None
    damage: int
    litter: int
    photos: int
    blacklisted: int
    top_user: str | None  # the name with the most damage records; a trailing space marks a plate fallback
    top_is_plate: bool
    photo_subject: str | None
    damage_plate: str | None
    busiest_plate: str | None
    llm_model: str
    llm_state: str  # "ready" | "offline" | "off"
    visual_ready: bool


def _subject(uid: str, users: dict, plate_of: dict) -> str | None:
    """A name that resolves to exactly this person; a shared name falls back to their plate."""
    names = sorted(name for name in users.get(uid, ()) if name != uid)
    if not names:
        return None
    shared = sum(names[0] in aliases for aliases in users.values()) > 1
    if not shared:
        return names[0]
    return f"{plate_of[uid]} " if plate_of.get(uid) else None


def _llm_state(settings) -> tuple[str, str]:
    model = settings.history_rag_model or settings.ollama_vision_model
    if not settings.history_rag_generation_enabled:
        return model, "off"
    try:
        with httpx.Client(transport=history_llm.TRANSPORT, timeout=httpx.Timeout(0.8)) as client:
            response = client.get(f"{settings.ollama_base_url.rstrip('/')}/api/tags")
        response.raise_for_status()
        names = {item.get("name", "") for item in response.json().get("models", [])}
    except (httpx.HTTPError, ValueError, TypeError):
        return model, "offline"
    return model, "ready" if model in names or f"{model}:latest" in names else "offline"


def _live(store, settings) -> _Live:
    from app.services.damage_pair_specialist import MODEL_PATH
    from app.services.history_evidence import Directory, row_kind

    users, rentals = catalogue(store)
    directory = Directory(store)
    events = store.list_rental_events(user_id=None, limit=None)
    damage = [event for event in events if row_kind(event) == "damage" and event.event_type != "damage_reviewed"]
    plate_of: dict[str, str] = {}
    for uid, plate in rentals.values():
        if plate:
            plate_of.setdefault(uid, plate)
    by_user = Counter(directory.renter_id(event) for event in damage)
    top = next((subject for uid, _ in by_user.most_common() if (subject := _subject(uid, users, plate_of))), None)
    with_photo = [event for event in damage if photo_source_for(event)]
    photo_subject = _subject(directory.renter_id(with_photo[0]), users, plate_of) if with_photo else None
    damage_plate = next((plate for event in (with_photo or damage) if (plate := directory.plate(event))), None)
    busiest = Counter(plate for _, plate in rentals.values() if plate).most_common(1)
    dates = sorted(str(value)[:10] for row in rental_details(store).values()
                   for value in (row.get("rental_date"), row.get("return_date")) if value)
    with store._connect() as connection:
        blacklisted = connection.execute("SELECT COUNT(*) FROM history_blacklist WHERE blacklisted = 1").fetchone()[0]
    model, state = _llm_state(settings) if settings else ("", "off")
    return _Live(users=len(users), rentals=len(rentals), first=dates[0] if dates else None,
                 last=dates[-1] if dates else None, damage=len({event.rental_id or event.id for event in damage}),
                 litter=len({event.rental_id or event.id for event in events if row_kind(event) == "litter"}),
                 # Every event of a photographed rental carries the photos; count the rentals.
                 photos=len({event.rental_id or event.id for event in events if photo_source_for(event)}),
                 blacklisted=blacklisted,
                 top_user=top, top_is_plate=bool(top and top.endswith(" ")), photo_subject=photo_subject,
                 damage_plate=damage_plate,
                 busiest_plate=busiest[0][0] if busiest else None, llm_model=model, llm_state=state,
                 visual_ready=MODEL_PATH.is_file())


def _item(key, title, description, examples, status="ready", status_label="可用") -> CapabilityItem:
    return CapabilityItem(key=key, title=title, description=description, status=status, status_label=status_label,
                          examples=[example for example in examples if example][:3])


# Answer intents each capability covers; "clarify" and "capabilities" are not data answers.
CAPABILITIES = {
    "statistics": ("aggregate", "list_users"),
    "timeline": ("timeline",),
    "photos": ("event_detail",),
    "prior_damage": ("prior_damage",),
    "risk": ("risk",),
    "blacklist": ("blacklist",),
    "conversation": (),
    "summary": (),
    "export": (),
}


def _items(live: _Live) -> list[CapabilityItem]:
    empty = live.rentals == 0
    no_data = ("limited", "尚無資料") if empty else ("ready", "可用")
    top = live.top_user
    items = [
        _item("statistics", "統計與名單",
              "依對象、期間與主題計算車損、垃圾、賠償的件數、金額和名單。數字由系統直接計算，不由 AI 推算；"
              "待確認或爭議中的紀錄會標示且不計入。",
              ["這個月有哪些使用者造成車損？", f"{top}總共要賠多少錢？" if top else "全部使用者總共要賠多少錢？",
               "哪些使用者曾留下垃圾？"], *no_data),
        _item("timeline", "租借歷程",
              "列出某位使用者或某台車的租借紀錄，可看出同一台車前後由誰租借。",
              [f"{live.busiest_plate} 的租借歷程" if live.busiest_plate else "查詢同車牌的租借歷程",
               None if not top or live.top_is_plate else f"{top}的租借紀錄"], *no_data),
    ]
    if live.photos:
        photo = ("ready", f"{live.photos} 張租單有照片")
        photo_text = (f"車損紀錄附有借還車照片時，答案中直接並排顯示（最多 10 組），並說明部位、類型、"
                      f"借車時是否已有。目前有 {live.photos} 張租單附照片。")
    else:
        photo = ("limited", "目前沒有照片")
        photo_text = "目前資料沒有附照片，只能從紀錄文字判讀部位與類型；上傳借還車照片後會直接顯示在答案中。"
    items.append(_item("photos", "照片判讀", photo_text + "看完車損答案後，可追問「照片的損傷是什麼？」。",
                       [f"{live.photo_subject}的車損照片" if live.photo_subject else None], *photo))
    visual = "並用車損比對模型比較前後兩次的還車照片" if live.visual_ready else "（車損比對模型未安裝，只比對紀錄文字）"
    items.append(_item("prior_damage", "既有損傷追溯",
                       f"找出同一台車較早的租單，檢查相同部位是否已有損傷紀錄，{visual}。結論僅供人工判斷，不會自動改責任；"
                       "看完照片判讀後，可追問「這個損傷之前就有嗎？」。",
                       [f"{live.damage_plate} 的損傷之前就有嗎？" if live.damage_plate else None],
                       *(("ready", "可用") if live.damage and live.visual_ready else
                         ("limited", "僅文字比對") if live.damage else ("limited", "尚無車損"))))
    items.append(_item("risk", "風險雷達",
                       "依風險等級、未賠金額、責任車損與垃圾次數排出最需要注意的使用者，並列出原因。",
                       ["誰最需要注意？", "風險最高的使用者有哪些？"], *no_data))
    items.append(_item("blacklist", "黑名單",
                       f"查詢誰被列入黑名單與原因；目前有 {live.blacklisted} 位。",
                       ["目前有哪些人列入黑名單？",
                        None if not top or live.top_is_plate else f"{top}有被列入黑名單嗎？"],
                       *no_data))
    items.append(_item("conversation", "連續追問",
                       "記得上一題的對象、期間與主題，可以直接追問，例如「總共要賠多少錢？」「那上個月呢？」"
                       "「有爭議的有哪些？」；答案上方的「對象／期間／主題」可點選切換。", [], "ready", "可用"))
    summary = {"ready": ("ready", "可用", f"由本機模型 {live.llm_model} 用一段話說明答案卡的重點，每句都與資料核對，"
                                        "不符的句子不會顯示。"),
               "offline": ("limited", "模型離線", f"本機模型 {live.llm_model} 目前離線，只顯示系統計算的答案卡；"
                                              "數字與名單不受影響。"),
               "off": ("off", "已停用", "AI 摘要已在設定中停用，只顯示系統計算的答案卡。")}[live.llm_state]
    items.append(_item("summary", "AI 摘要", summary[2], [], summary[0], summary[1]))
    items.append(_item("export", "下載與回饋",
                       "對話中最多顯示 20 筆依據與 10 組照片，其餘可按「下載完整 Excel」取得；每則回答可按讚或倒讚，"
                       "倒讚的題目會成為之後的測試題。", [], "ready", "可用"))
    return items


def capability_card(store, settings) -> HistoryAnswerCard:
    live = _live(store, settings)
    if live.rentals:
        span = f"（{live.first}～{live.last}）" if live.first and live.last else ""
        headline = (f"我是借還車AI助理，目前可查詢 {live.users} 位使用者、{live.rentals} 張租單{span}的"
                    "車損、垃圾、賠償與黑名單紀錄。點下面的範例問題就能直接問。")
    else:
        headline = "我是借還車AI助理，目前還沒有租借資料；匯入租借歷程或建立 RAG 種子資料後就能開始查詢。"
    metrics = [CardMetric(label="使用者", value=f"{live.users} 位"),
               CardMetric(label="租單", value=f"{live.rentals} 張"),
               CardMetric(label="車損紀錄", value=f"{live.damage} 張租單", tone="warn" if live.damage else "neutral"),
               CardMetric(label="有照片的租單", value=f"{live.photos} 張"),
               CardMetric(label="黑名單", value=f"{live.blacklisted} 位", tone="bad" if live.blacklisted else "good")]
    return HistoryAnswerCard(intent="capabilities", headline=headline, metrics=metrics, capabilities=_items(live),
                             topic_label="功能介紹")


def example_questions(store) -> list[str]:
    """The composer's + menu: questions that answer on the current data, with real names and plates."""
    live = _live(store, None)  # examples never depend on the model, so skip probing it
    top, plate = live.top_user, live.busiest_plate
    person = None if not top or live.top_is_plate else top
    questions = [
        "你可以做什麼？",
        "這個月有哪些使用者造成車損？", "上個月有哪些使用者造成車損？", "去年有哪些使用者造成車損？",
        "有爭議的車損有哪些？", "有哪些車損還沒賠？", "全部使用者總共還欠多少錢？",
        f"{top}總共要賠多少錢？" if top else None,
        "哪些使用者曾留下垃圾？", "這個月的垃圾紀錄",
        "誰最需要注意？", "不良紀錄最多的是誰？",
        "目前有哪些人列入黑名單？", f"{person}有被列入黑名單嗎？" if person else None,
        "全部使用者的租借紀錄", f"{plate} 的租借歷程" if plate else None,
        f"{person}的租借紀錄" if person else None,
        f"{live.photo_subject}的車損照片" if live.photo_subject else None,
        f"{live.damage_plate} 的損傷之前就有嗎？" if live.damage_plate else None,
    ]
    return [question for question in questions if question]


def capability_response(request: HistoryQueryRequest, store, settings) -> HistoryQueryResponse:
    card = capability_card(store, settings)
    # The previous scope stays in the context, so 「總共要賠多少錢？」 still follows the earlier answer.
    return HistoryQueryResponse(question=request.question, answer=card.headline, query_mode="capabilities",
                                generated_by="template", conversation_title="助理功能介紹", context=request.context,
                                answer_card=card)
