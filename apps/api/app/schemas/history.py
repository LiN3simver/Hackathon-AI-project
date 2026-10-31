from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator


class AutomationRoute(str, Enum):
    AUTO_CLEAR = "auto_clear"
    AUTO_CASE = "auto_case"
    MANUAL_REVIEW = "manual_review"


class RentalEvent(BaseModel):
    id: str
    user_id: str
    rental_id: str
    vehicle_id: str
    session_id: str | None = None
    event_type: str
    stage: str | None = None
    source_type: str
    source_id: str
    title: str
    summary: str
    outcome: str | None = None
    severity: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    verified: bool = False
    disputed: bool = False
    metadata: dict = Field(default_factory=dict)
    created_at: datetime
    occurred_on: date | None = None


class AutomationDecision(BaseModel):
    id: str
    job_id: str
    session_id: str
    user_id: str
    rental_id: str
    vehicle_id: str
    task: str
    recommended_route: AutomationRoute
    effective_route: AutomationRoute
    shadow_mode: bool
    confidence: float | None = Field(default=None, ge=0, le=1)
    reason_codes: list[str] = Field(default_factory=list)
    evidence: dict = Field(default_factory=dict)
    policy_version: str
    created_at: datetime


class AutomationSummary(BaseModel):
    enabled: bool
    shadow_mode: bool
    policy_version: str
    total: int
    recommended: dict[str, int]
    effective: dict[str, int]
    simulated_review_reduction_rate: float
    effective_manual_review_rate: float
    decisions: list[AutomationDecision]


class UserHistoryListItem(BaseModel):
    user_id: str
    display_name: str
    rental_ids: list[str]
    vehicle_ids: list[str]
    rental_count: int
    event_count: int
    confirmed_damage_count: int
    confirmed_cleanliness_count: int
    pending_review_count: int
    last_event_at: datetime | None = None
    caused_damage_count: int = 0
    compensation_count: int = 0
    compensation_amount: Decimal = Decimal("0.00")
    paid_amount: Decimal = Decimal("0.00")
    outstanding_amount: Decimal = Decimal("0.00")
    blacklisted: bool = False
    blacklist_reason: str = ""
    litter_count: int = 0
    pending_litter_count: int = 0
    disputed_litter_count: int = 0
    has_litter_history: bool = False
    dirty_count: int = 0
    damage_compensation_count: int = 0
    damage_compensation_amount: Decimal = Decimal("0.00")
    damage_paid_amount: Decimal = Decimal("0.00")
    damage_outstanding_amount: Decimal = Decimal("0.00")


class UserHistoryProfile(UserHistoryListItem):
    disputed_count: int
    risk_level: str
    risk_reasons: list[str]
    events: list[RentalEvent]


HistoryTopic = Literal["history", "damage", "cleanliness", "litter", "compensation", "blacklist", "dispute"]
HistoryIntent = Literal[
    "aggregate", "list_users", "event_detail", "timeline", "blacklist", "risk", "prior_damage", "clarify",
    "capabilities",
]


class HistoryChatContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scope: Literal["user", "users", "vehicle", "vehicles", "all"]
    user_ids: list[str] = Field(default_factory=list, max_length=20)
    license_plates: list[str] = Field(default_factory=list, max_length=20)
    rental_ids: list[str] = Field(default_factory=list, max_length=20)
    topics: list[HistoryTopic] = Field(default_factory=list, max_length=7)
    date_from: date | None = None
    date_to: date | None = None
    # Dialogue state: what the last answer was about. Client-held, so re-validated on every turn.
    intent: HistoryIntent | None = None
    focus_event_ids: list[str] = Field(default_factory=list, max_length=20)

    @model_validator(mode="after")
    def validate_period(self) -> "HistoryChatContext":
        if self.date_from and self.date_to and self.date_to <= self.date_from:
            raise ValueError("結束日期必須晚於開始日期")
        return self


class HistoryUserSuggestion(BaseModel):
    user_id: str
    display_name: str


class HistoryModelOption(BaseModel):
    """A text model the assistant can phrase its summary with; endpoints and keys stay on the server."""

    id: str
    label: str
    provider: Literal["ollama", "openai_compatible"]
    tier: Literal["local", "paid", "gpu"]
    description: str = ""
    available: bool


class HistoryModelList(BaseModel):
    default_id: str
    options: list[HistoryModelOption]


class HistoryScopeOption(BaseModel):
    scope: Literal["user", "vehicle"]
    value: str = Field(min_length=1, max_length=80)
    label: str = Field(min_length=1, max_length=200)


class HistoryChatTurn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class HistoryQueryRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    question: str = Field(min_length=2, max_length=500)
    user_id: str | None = Field(default=None, max_length=80)
    all_users: bool = False
    license_plate: str | None = Field(default=None, min_length=5, max_length=10)
    context: HistoryChatContext | None = None
    messages: list[HistoryChatTurn] = Field(default_factory=list, max_length=12)
    selection: HistoryScopeOption | None = None
    date_from: date | None = None
    date_to: date | None = None
    user_ids: list[str] = Field(default_factory=list, max_length=20)
    renter_ids: list[str] = Field(default_factory=list, max_length=20)
    rental_ids: list[str] = Field(default_factory=list, max_length=20)
    # An option id from GET /history/models; unset uses the default model.
    model_id: str | None = Field(default=None, max_length=60)

    @model_validator(mode="after")
    def validate_scope(self) -> "HistoryQueryRequest":
        if self.all_users and self.user_id:
            raise ValueError("全部使用者查詢不能同時指定使用者")
        if self.date_from and self.date_to and self.date_to <= self.date_from:
            raise ValueError("結束日期必須晚於開始日期")
        if self.license_plate:
            from app.services.license_plate import normalize_plate
            plate = normalize_plate(self.license_plate)
            if plate is None:
                raise ValueError("請輸入有效車牌")
            self.license_plate = plate
            if self.user_id or self.all_users:
                raise ValueError("車牌查詢不能同時指定使用者或全部使用者")
        return self


class HistoryCitation(BaseModel):
    event_id: str
    rental_id: str
    source_type: str
    source_id: str
    created_at: datetime


class EvidenceRow(BaseModel):
    """One numbered source in the answer card; [n] in the summary points here."""

    ref: int
    event_id: str | None = None
    rental_id: str | None = None
    occurred_on: date | None = None
    return_date: date | None = None
    # The rental this record belongs to: lets 「2026-09-21 還車時…」 be checked against the record itself.
    pickup_on: date | None = None
    returned_on: date | None = None
    renter_name: str
    license_plate: str | None = None
    kind: str
    kind_label: str
    title: str
    summary: str
    responsibility: Literal["confirmed", "pending", "disputed", "ai_only", "not_applicable"] = "not_applicable"
    counted: bool = False
    present_at_pickup: bool | None = None
    present_at_return: bool | None = None
    compensation_amount: str | None = None
    paid_amount: str | None = None
    photo_slots: list[Literal["pickup", "return"]] = Field(default_factory=lambda: ["pickup", "return"])
    photo_source: Literal["rag_seed", "damage_case"] | None = None
    damage_case_id: str | None = None
    cited: bool = False


class CardMetric(BaseModel):
    label: str
    value: str
    tone: Literal["neutral", "good", "warn", "bad"] = "neutral"


class ChipOption(BaseModel):
    label: str
    question: str


class ScopeChip(BaseModel):
    kind: Literal["scope", "period", "topic"]
    label: str
    options: list[ChipOption] = Field(default_factory=list)


class PersonRow(BaseModel):
    name: str
    user_id: str = ""
    license_plates: list[str] = Field(default_factory=list)
    caused_damage_count: int = 0
    unconfirmed_damage_count: int = 0
    disputed_count: int = 0
    litter_count: int = 0
    compensation_amount: str = "0.00"
    outstanding_amount: str = "0.00"


class RiskRow(BaseModel):
    rank: int
    name: str
    license_plates: list[str] = Field(default_factory=list)
    level: str
    level_label: str
    reasons: list[str] = Field(default_factory=list)
    rental_count: int = 0
    caused_damage_count: int = 0
    litter_count: int = 0
    disputed_count: int = 0
    outstanding_amount: str = "0.00"
    blacklisted: bool = False
    score: float = Field(default=0, ge=0, le=1)


class PriorRental(BaseModel):
    renter_name: str
    rental_date: date | None = None
    return_date: date | None = None
    records: list[str] = Field(default_factory=list)
    same_location: bool = False
    visual_score: float | None = None


class PriorDamageCheck(BaseModel):
    """Did the mark already exist before this rental? Advisory only; never changes liability."""

    focus_event_id: str | None = None
    renter_name: str | None = None
    license_plate: str | None = None
    location: str | None = None
    damage_type: str | None = None
    verdict: Literal["already_present", "new_damage", "unknown"] = "unknown"
    verdict_text: str
    visual_checked: bool = False
    earlier_rentals: list[PriorRental] = Field(default_factory=list)


class CapabilityItem(BaseModel):
    """One thing the assistant can do now, introduced in answer to 「你可以做什麼？」."""

    key: str
    title: str
    description: str
    status: Literal["ready", "limited", "off"] = "ready"
    status_label: str = "可用"
    examples: list[str] = Field(default_factory=list)


class HistoryAnswerCard(BaseModel):
    """Deterministic answer computed before (and without) the language model."""

    intent: HistoryIntent
    headline: str
    scope_label: str = ""
    period_label: str = ""
    topic_label: str = ""
    metrics: list[CardMetric] = Field(default_factory=list)
    rows: list[EvidenceRow] = Field(default_factory=list)
    total_rows: int = 0
    people: list[PersonRow] = Field(default_factory=list)
    # Which people-table column groups the question asked about; a damage-only answer shows no litter column.
    people_columns: list[Literal["damage", "litter"]] = Field(default_factory=lambda: ["damage", "litter"])
    risk: list[RiskRow] = Field(default_factory=list)
    chips: list[ScopeChip] = Field(default_factory=list)
    followups: list[str] = Field(default_factory=list)
    detail_event_ids: list[str] = Field(default_factory=list)
    prior_damage: PriorDamageCheck | None = None
    # Photo pairs shown inline (photo_limit) out of the photo rows in this card; the export has all of them.
    photo_limit: int = 10
    photo_total: int = 0
    capabilities: list[CapabilityItem] = Field(default_factory=list)


class HistoryQueryResponse(BaseModel):
    question: str
    answer_id: str | None = None  # set when an AI summary is shown; 👎 feedback refers to it
    user_id: str | None = None
    license_plate: str | None = None
    query_mode: str
    answer: str
    generated_by: str
    conversation_title: str | None = None
    agent_steps: list[dict] = Field(default_factory=list)
    interpreted_filters: dict = Field(default_factory=dict)
    citations: list[HistoryCitation] = Field(default_factory=list)
    matched_events: list[RentalEvent] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    statistics: dict = Field(default_factory=dict)
    context: HistoryChatContext | None = None
    clarification_options: list[HistoryScopeOption] = Field(default_factory=list)
    answer_card: HistoryAnswerCard | None = None
    # Every event that matched the filters (matched_events is only the ranked excerpt).
    _all_events: list[RentalEvent] = PrivateAttr(default_factory=list)
    _rental_rows: list[dict] = PrivateAttr(default_factory=list)


class DamageRegion(BaseModel):
    description: str
    score: float | None = None
    bounding_box: list[float] | None = None


class HistoryEventDetail(BaseModel):
    """What the photos show for one damage record, from the AI comparison or the record text."""

    event_id: str
    renter_name: str
    license_plate: str | None = None
    occurred_on: date | None = None
    title: str
    summary: str
    responsibility: str
    disputed: bool = False
    source: Literal["damage_case", "record_text", "none"]
    view_label: str | None = None
    location: str | None = None
    damage_type: str | None = None
    present_at_pickup: bool | None = None
    present_at_return: bool | None = None
    conclusion: str
    ai_summary: str | None = None
    ai_score: float | None = None
    pickup_match_score: float | None = None
    review_decision: str | None = None
    reviewer_notes: str | None = None
    regions: list[DamageRegion] = Field(default_factory=list)
    photo_slots: list[Literal["pickup", "return"]] = Field(default_factory=lambda: ["pickup", "return"])
    photo_source: Literal["rag_seed", "damage_case"] | None = None
    damage_case_id: str | None = None


class HistoryFeedbackInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    question: str = Field(min_length=1, max_length=500)
    answer: str = Field(default="", max_length=8000)
    rating: Literal[1, -1]
    comment: str = Field(default="", max_length=1000)
    expected: str = Field(default="", max_length=2000)
    context: HistoryChatContext | None = None
    intent: HistoryIntent | None = None
    generated_by: str | None = Field(default=None, max_length=40)
    warnings: list[str] = Field(default_factory=list, max_length=20)
    cited_event_ids: list[str] = Field(default_factory=list, max_length=50)
    answer_id: str | None = Field(default=None, max_length=40)


class HistoryFeedback(HistoryFeedbackInput):
    id: str
    created_at: datetime
    created_by: str


class DamageHistoryRecordInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    rental_id: str = Field(min_length=1, max_length=80)
    record_type: Literal["damage"] = "damage"
    description: str = Field(min_length=1, max_length=1000)
    occurred_on: date | None = None
    damage_case_id: str | None = Field(default=None, min_length=1, max_length=80)
    compensation_amount: Decimal = Field(default=Decimal("0"), ge=0, le=100000000,
                                         decimal_places=2)
    paid_amount: Decimal = Field(default=Decimal("0"), ge=0, le=100000000,
                                decimal_places=2)
    responsibility_confirmed: bool = False
    disputed: bool = False
    # Optional photo facts; when absent the assistant reads them from the description.
    damage_location: str | None = Field(default=None, max_length=40)
    damage_type: Literal[
        "scratch", "dent", "crack", "broken_part", "paint_transfer", "deformation", "other",
    ] | None = None
    present_at_pickup: bool | None = None

    @model_validator(mode="after")
    def validate_payment(self) -> "DamageHistoryRecordInput":
        if self.paid_amount > self.compensation_amount:
            raise ValueError("已賠金額不能超過應賠金額")
        return self


class OptionalClaimInput(BaseModel):
    claim_enabled: bool = False
    compensation_amount: Decimal = Field(default=Decimal("0"), ge=0, le=100000000, decimal_places=2)
    paid_amount: Decimal = Field(default=Decimal("0"), ge=0, le=100000000, decimal_places=2)

    @model_validator(mode="after")
    def validate_claim(self) -> "OptionalClaimInput":
        if self.paid_amount > self.compensation_amount:
            raise ValueError("已賠金額不能超過索賠金額")
        if not self.claim_enabled and (self.compensation_amount or self.paid_amount):
            raise ValueError("請先選擇新增索賠金額")
        return self


class LitterHistoryRecordInput(OptionalClaimInput):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    rental_id: str = Field(min_length=1, max_length=80)
    description: str = Field(min_length=1, max_length=1000)
    occurred_on: date | None = None
    cleanliness_alert_id: str | None = Field(default=None, min_length=1, max_length=80)
    responsibility_confirmed: bool = False
    disputed: bool = False


class CleanlinessHistoryRecordInput(OptionalClaimInput):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    rental_id: str = Field(min_length=1, max_length=80)
    description: str = Field(min_length=1, max_length=1000)
    occurred_on: date | None = None
    responsibility_confirmed: bool = False
    disputed: bool = False


class BlacklistUpdate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    blacklisted: bool
    reason: str = Field(min_length=1, max_length=1000)
