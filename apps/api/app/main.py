import asyncio
import time
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.requests import Request

from app.api.routes import (
    admin,
    auth,
    capture,
    damage,
    health,
    history,
    inspections,
    interior,
    learning,
    mobile,
    operations,
    plate,
    quality,
    rag_seed,
    vehicle_history,
)
from app.core.config import get_settings
from app.core.logging_config import configure_logging
from app.services.account_store import get_account_store
from app.services.activity_audit import import_legacy_http_logs
from app.services.mobile_access import generate_mobile_access
from app.services.operations_store import OPERATIONS_ACTIVITY_RETENTION_DAYS

settings = get_settings()
logger = configure_logging("api")
OPERATIONS_ACTIVITY_CLEANUP_INTERVAL_SECONDS = 24 * 60 * 60


async def _warm_history_llm() -> None:
    """Load the assistant's model with the fixed system prompt so the first question skips the cold start."""
    if not (settings.history_rag_generation_enabled and settings.history_llm_warmup):
        return
    from app.services import history_llm

    await asyncio.sleep(3)
    try:
        await asyncio.to_thread(history_llm.warm_up, settings)
    except Exception as exc:  # optional: the assistant still answers with its card
        logger.warning("history_llm_warmup_failed error=%s", type(exc).__name__)


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Keep the fixed active-timeline retention and generate the phone entry at startup."""

    await asyncio.to_thread(get_account_store)
    closed = await asyncio.to_thread(operations.get_operations_store().reconcile_inspection_sessions)
    if closed:
        logger.info("obsolete_inspection_sessions_closed count=%s", closed)

    def recover_logs():
        with operations.get_operations_store()._connect() as connection:
            return import_legacy_http_logs(connection, settings.log_directory)

    recovered = await asyncio.to_thread(recover_logs)
    if recovered:
        logger.info("operations_activity_legacy_logs recovered=%s", recovered)
    application.state.mobile_access = await asyncio.to_thread(generate_mobile_access)
    await _clear_expired_operations_activity()
    stopped = asyncio.Event()

    async def run_retention_schedule() -> None:
        while True:
            try:
                await asyncio.wait_for(stopped.wait(), timeout=OPERATIONS_ACTIVITY_CLEANUP_INTERVAL_SECONDS)
                return
            except TimeoutError:
                await _clear_expired_operations_activity()

    retention_task = asyncio.create_task(run_retention_schedule())
    warmup_task = asyncio.create_task(_warm_history_llm())
    try:
        yield
    finally:
        stopped.set()
        warmup_task.cancel()
        await asyncio.gather(warmup_task, return_exceptions=True)
        await retention_task


async def _clear_expired_operations_activity() -> None:
    try:
        deleted = await asyncio.to_thread(
            operations.get_operations_store().clear_operations_activity_older_than,
            days=OPERATIONS_ACTIVITY_RETENTION_DAYS,
        )
        if deleted:
            logger.info(
                "operations_activity_retention archived_and_deleted=%s older_than_days=%s",
                deleted,
                OPERATIONS_ACTIVITY_RETENTION_DAYS,
            )
    except Exception:
        logger.exception("operations_activity_retention_failed")


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="Guardian：智能車況拍照品質檢查與 AI 分析 API",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        settings.web_origin,
        "https://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_origin_regex=(
        r"https?://(?:localhost|127\.0\.0\.1):51\d{2}" if settings.environment == "development" else None
    ),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    # Lets the browser read the Excel export's file name.
    expose_headers=["Content-Disposition"],
)
app.include_router(health.router, prefix="/api/v1")
app.include_router(history.router, prefix="/api/v1")
app.include_router(auth.router, prefix="/api/v1")
app.include_router(admin.router, prefix="/api/v1")
app.include_router(quality.router, prefix="/api/v1")
app.include_router(inspections.router, prefix="/api/v1")
app.include_router(interior.router, prefix="/api/v1")
app.include_router(operations.router, prefix="/api/v1")
app.include_router(damage.router, prefix="/api/v1")
app.include_router(learning.router, prefix="/api/v1")
app.include_router(plate.router, prefix="/api/v1")
app.include_router(capture.router, prefix="/api/v1")
app.include_router(rag_seed.router, prefix="/api/v1")
app.include_router(vehicle_history.router, prefix="/api/v1")
app.include_router(mobile.router, prefix="/api/v1")


class RequestAuditMiddleware:
    """Audit after the full response and background tasks, including stream failures."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        request = Request(scope)
        started = time.perf_counter()
        request_id = uuid4().hex
        status_code = 500
        error_type = None
        completed = False

        async def audit_send(message):
            nonlocal status_code, completed
            if message["type"] == "http.response.start":
                status_code = message["status"]
                message["headers"] = [*message.get("headers", []), (b"x-request-id", request_id.encode())]
            if message["type"] == "http.response.body" and not message.get("more_body", False):
                completed = True
            await send(message)

        try:
            await self.app(scope, receive, audit_send)
        except Exception as exc:
            error_type = type(exc).__name__
            logger.exception(
                "request_failed method=%s path=%s request_id=%s", request.method, request.url.path, request_id
            )
            raise
        finally:
            elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
            logger.info(
                "request method=%s path=%s status=%s elapsed_ms=%.1f request_id=%s",
                request.method,
                request.url.path,
                status_code,
                elapsed_ms,
                request_id,
            )
            if request.url.path.startswith("/api/v1/") and request.method != "OPTIONS":
                route = scope.get("route")
                path = request.url.path
                failed = status_code >= 400 or error_type is not None or not completed
                metadata = {
                    "request_id": request_id,
                    "method": request.method,
                    "path": path,
                    "route": getattr(route, "path", path),
                    "status_code": status_code,
                    "elapsed_ms": elapsed_ms,
                    "error_type": error_type,
                    "response_completed": completed,
                    "outcome": "failed" if failed else "success",
                }
                # Bodies, Authorization headers and token query strings are never persisted.
                try:
                    await asyncio.to_thread(
                        operations.get_operations_store().record_request_activity,
                        request_id=request_id,
                        actor=getattr(request.state, "audit_actor", "anonymous"),
                        summary=f"{request.method} {path} → {status_code}" + ("（失敗）" if failed else ""),
                        metadata=metadata,
                        restricted=path.startswith(("/api/v1/admin/", "/api/v1/learning/")),
                    )
                except Exception:
                    logger.exception("request_audit_failed request_id=%s", request_id)


app.add_middleware(RequestAuditMiddleware)


@app.get("/", include_in_schema=False)
def root() -> dict[str, str]:
    return {"name": settings.app_name, "docs": "/docs"}
