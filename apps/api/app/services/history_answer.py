"""Answer pipeline: validated scope → deterministic card → one verified, streamed summary.

The card is complete on its own and is sent first. The language model, when
available, only adds a short verified summary; any failure leaves the card intact.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime

import httpx

from app.core.config import get_settings
from app.schemas.history import HistoryCitation, HistoryQueryRequest, HistoryQueryResponse
from app.services import history_llm, history_llm_models, history_regressions
from app.services.history_capabilities import capability_response, is_capability_question
from app.services.history_evidence import PACK_ROWS, Evidence, build_evidence, title_for

logger = logging.getLogger("api")
LLM_ERRORS = (httpx.HTTPError, ValueError, TypeError, KeyError)


@dataclass
class Prepared:
    request: HistoryQueryRequest
    response: HistoryQueryResponse
    evidence: Evidence
    messages: list[dict] | None
    store: object = None

    @property
    def needs_llm(self) -> bool:
        return self.evidence.pack is not None and self.response.answer_card.intent != "clarify"


def _today():
    from app.services.history_query import _TAIPEI

    return datetime.now(_TAIPEI).date()


def _previous_question(request: HistoryQueryRequest) -> str | None:
    return next((turn.content for turn in reversed(request.messages) if turn.role == "user"), None)


def prepare_answer(request: HistoryQueryRequest, store) -> Prepared:
    """Everything except the model: scope, retrieval, ranking, answer card and evidence pack."""
    from app.services.history_query import query_history_data

    if is_capability_question(request.question):
        response = capability_response(request, store, get_settings())
        return Prepared(request=request, response=response, evidence=Evidence(card=response.answer_card, pack=None),
                        messages=None, store=store)
    response = query_history_data(request, store=store)
    today = _today()
    evidence = build_evidence(request, response, store, today=today)
    card = evidence.card
    response.answer_card = card
    if card.intent != "clarify":
        response.conversation_title = title_for(card)
    if response.context is not None and card.intent != "clarify":
        focus = card.detail_event_ids or [row.event_id for row in card.rows[:PACK_ROWS] if row.event_id]
        response.context = response.context.model_copy(update={"intent": card.intent, "focus_event_ids": focus[:20]})
    messages = None
    if evidence.pack is not None:
        messages = history_llm.build_messages(evidence.pack, request.question, _previous_question(request),
                                              today.isoformat())
    return Prepared(request=request, response=response, evidence=evidence, messages=messages, store=store)


def assemble(prepared: Prepared, result: history_llm.SummaryResult,
             answer_id: str | None = None) -> HistoryQueryResponse:
    response = prepared.response.model_copy(deep=True)
    response.answer_id = answer_id
    response._all_events = prepared.response._all_events
    response._rental_rows = prepared.response._rental_rows
    pack = prepared.evidence.pack
    response.agent_steps = [{"tool": "summary", "status": result.failure or "completed",
                             "rejected": result.reason,
                             "rejected_sentence": result.rejected_sentence, **result.metrics}]
    if result.failure == "unverified":
        response.warnings.append("ollama_unverified")
    elif result.failure == "partial":
        response.warnings.append("ollama_partial" if result.sentences else "ollama_timeout")
    if not result.sentences:
        return response
    response.answer = result.text
    response.generated_by = "ollama_summary"
    response.query_mode = "rag_card_plus_summary"
    cited = [pack.refs[ref - 1] for ref in result.cited if pack and 0 < ref <= len(pack.refs) and pack.refs[ref - 1]]
    card = response.answer_card
    if card is not None:
        for row in card.rows:
            row.cited = row.event_id in cited
    events = prepared.evidence.events
    response.citations = [HistoryCitation(event_id=event_id, rental_id=events[event_id].rental_id,
                                          source_type=events[event_id].source_type,
                                          source_id=events[event_id].source_id,
                                          created_at=events[event_id].created_at)
                          for event_id in cited if event_id in events]
    if cited and response.context is not None and card is not None and not card.detail_event_ids:
        response.context = response.context.model_copy(update={"focus_event_ids": cited[:20]})
    return response


def chosen_backend(request: HistoryQueryRequest, response: HistoryQueryResponse, settings):
    """The picker's model; an unknown id (e.g. removed since the page loaded) answers with the default."""
    option, found = history_llm_models.resolve(settings, request.model_id)
    if not found:
        response.warnings.append("llm_model_unknown")
    return history_llm_models.backend_for(option, settings)


def complete_sync(prepared: Prepared, settings, backend=None) -> HistoryQueryResponse:
    result = history_llm.summarize_sync(prepared.evidence.pack, prepared.messages, settings, backend)
    if backend is not None:
        result.metrics["model"] = backend.option.id
    _log(prepared, result)
    return assemble(prepared, result, _remember(prepared, result))


def _remember(prepared: Prepared, result: history_llm.SummaryResult) -> str | None:
    """Refused sentences become regression cases; a shown summary keeps its evidence for a later 👎."""
    if prepared.store is None or prepared.evidence.pack is None or prepared.messages is None:
        return None
    return history_regressions.remember_summary(prepared.store, prepared.request.question, prepared.evidence.pack,
                                                prepared.messages, result.sentences, result.rejected_sentence,
                                                result.reason)


def _log(prepared: Prepared, result: history_llm.SummaryResult) -> None:
    logger.info("history_summary intent=%s verified=%s sentences=%s first_token_ms=%s total_ms=%s reason=%s",
                prepared.response.answer_card.intent if prepared.response.answer_card else None,
                result.failure is None, len(result.sentences), result.metrics.get("first_token_ms"),
                result.metrics.get("total_ms"), result.reason)


def sse(event: str, payload) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n\n"


async def answer_events(request: HistoryQueryRequest, store, settings):
    """Server-sent events: card → (status) → delta* → done. Errors still end with a usable done."""
    from app.services.history_query import _generation_slot, query_history_data

    started = time.perf_counter()
    try:
        prepared = await asyncio.to_thread(prepare_answer, request, store)
    except Exception:
        logger.exception("history_stream_prepare_failed")
        yield sse("error", {"message": "查詢暫時無法完成，請再試一次。"})
        return
    yield sse("card", prepared.response.model_dump(mode="json"))
    logger.info("history_card intent=%s elapsed_ms=%.0f", prepared.response.answer_card.intent,
                (time.perf_counter() - started) * 1000)
    response = prepared.response
    if not settings.history_rag_generation_enabled:
        response.warnings.append("llm_generation_disabled")
    if not settings.history_rag_generation_enabled or not prepared.needs_llm:
        yield sse("done", response.model_dump(mode="json"))
        return
    deadline = time.monotonic() + settings.history_llm_queue_seconds
    acquired, announced = _generation_slot.acquire(blocking=False), False
    while not acquired and time.monotonic() < deadline:
        if not announced:
            yield sse("status", {"state": "queued"})
            announced = True
        await asyncio.sleep(0.2)
        acquired = _generation_slot.acquire(blocking=False)
    if not acquired:
        response.warnings.append("ollama_busy_fallback")
        yield sse("done", response.model_dump(mode="json"))
        return
    final = response
    try:
        result = None
        backend = chosen_backend(request, response, settings)
        async for item in history_llm.summarize_stream(prepared.evidence.pack, prepared.messages, settings, backend):
            if isinstance(item, history_llm.SummaryResult):
                result = item
            else:
                yield sse("delta", {"text": item})
        if result is not None:
            result.metrics["model"] = backend.option.id
            _log(prepared, result)
            answer_id = await asyncio.to_thread(_remember, prepared, result)
            final = assemble(prepared, result, answer_id)
    except LLM_ERRORS as exc:
        logger.warning("history_stream_llm_failed error=%s", type(exc).__name__)
        final = response.model_copy(update={"warnings": [*response.warnings, f"ollama_fallback:{type(exc).__name__}"]})
    except Exception:
        logger.exception("history_stream_failed")
        final = query_history_data(request, store=store)
        final.warnings.append("ollama_fallback:Error")
    finally:
        _generation_slot.release()
    yield sse("done", final.model_dump(mode="json"))
