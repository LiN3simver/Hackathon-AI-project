import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from app.core.config import POWER_AUTOMATE_PROVIDER, Settings, get_settings
from app.schemas.operations import NotificationDeliveryStatus, NotificationDispatchResponse
from app.services.operations_store import OperationsStore, get_operations_store

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EmailMessage:
    notification_id: str
    event_key: str
    recipient: str
    subject: str
    body_text: str
    created_at: str


class PowerAutomateOneDriveEmailAdapter:
    """Hands complete, immutable notification files to a synced OneDrive folder."""

    provider = POWER_AUTOMATE_PROVIDER

    def __init__(self, settings: Settings) -> None:
        configured_path = settings.power_automate_outbox_path
        if configured_path is None:
            raise ValueError(
                "Power Automate 模式需要設定 GUARDIAN_POWER_AUTOMATE_OUTBOX_PATH"
            )
        self.outbox_path = Path(configured_path).expanduser()
        if not self.outbox_path.is_dir():
            raise ValueError(
                "Power Automate 待寄資料夾不存在，請先建立並確認 OneDrive 正在同步："
                f" {self.outbox_path}"
            )

    def send(self, message: EmailMessage) -> None:
        if not message.notification_id:
            raise ValueError("Power Automate 通知缺少識別碼")
        target = self.outbox_path / f"{message.notification_id}.json"
        if target.exists():
            return
        payload = {
            "schema_version": 1,
            "notification_id": message.notification_id,
            "event_key": message.event_key,
            "to": message.recipient,
            "subject": message.subject,
            "body_text": message.body_text,
            "created_at": message.created_at,
        }
        temporary = self.outbox_path / f".{message.notification_id}.{uuid4().hex}.tmp"
        try:
            with temporary.open("x", encoding="utf-8") as output:
                json.dump(payload, output, ensure_ascii=False, indent=2)
                output.flush()
                os.fsync(output.fileno())
            temporary.replace(target)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise


def reconcile_power_automate_outbox(
    *, store: OperationsStore | None = None, settings: Settings | None = None
) -> None:
    """Use the Flow's file moves as delivery acknowledgements."""
    settings = settings or get_settings()
    outbox = settings.power_automate_outbox_path
    if outbox is None:
        return
    store = store or get_operations_store()
    root = Path(outbox).expanduser().parent
    sent_dirs = (root / "已寄出", root / "已寄")
    failed_dirs = (root / "失敗", root / "寄送失敗")
    for status in (
        NotificationDeliveryStatus.PROCESSING,
        NotificationDeliveryStatus.POWER_AUTOMATE_QUEUED,
        NotificationDeliveryStatus.POWER_AUTOMATE_FAILED,
    ):
        for item in store.list_notification_outbox(limit=None, status=status):
            if item.provider != POWER_AUTOMATE_PROVIDER:
                continue
            try:
                if any((folder / f"{item.id}.json").is_file() for folder in sent_dirs):
                    store.mark_power_automate_result(item.id, sent=True)
                elif status in (
                    NotificationDeliveryStatus.PROCESSING,
                    NotificationDeliveryStatus.POWER_AUTOMATE_QUEUED,
                ) and any(
                    (folder / f"{item.id}.json").is_file() for folder in failed_dirs
                ):
                    store.mark_power_automate_result(item.id, sent=False)
                elif status == NotificationDeliveryStatus.PROCESSING and (
                    Path(outbox).expanduser() / f"{item.id}.json"
                ).is_file():
                    store.mark_notification_handed_off(item.id)
            except OSError as exc:
                logger.warning(
                    "power_automate_result_check_failed notification_id=%s error=%s",
                    item.id,
                    exc,
                )


def dispatch_notification_outbox(
    *,
    store: OperationsStore | None = None,
    adapter: PowerAutomateOneDriveEmailAdapter | None = None,
    limit: int = 20,
) -> NotificationDispatchResponse:
    store = store or get_operations_store()
    settings = get_settings()
    reconcile_power_automate_outbox(store=store, settings=settings)
    if not settings.notification_email_enabled:
        return NotificationDispatchResponse(
            processed=0,
            handed_off=0,
            failed=0,
            message="Email 通知目前停用，通知保留待送。",
        )
    try:
        adapter = adapter or PowerAutomateOneDriveEmailAdapter(settings)
    except ValueError as exc:
        logger.error("email_adapter_configuration_error error=%s", exc)
        return NotificationDispatchResponse(
            processed=0,
            handed_off=0,
            failed=0,
            message=str(exc),
        )

    claimed = store.claim_notification_outbox(
        limit=limit,
        provider=adapter.provider,
    )
    handed_off = 0
    failed = 0
    for item in claimed:
        try:
            adapter.send(
                EmailMessage(
                    notification_id=item.id,
                    event_key=item.event_key,
                    recipient=item.recipient,
                    subject=item.subject,
                    body_text=item.body_text,
                    created_at=item.created_at.isoformat(),
                )
            )
        except Exception as exc:
            failed += 1
            store.mark_notification_failed(
                item.id,
                error_message=f"{type(exc).__name__}: {exc}",
            )
            logger.exception("email_delivery_failed notification_id=%s", item.id)
        else:
            handed_off += 1
            store.mark_notification_handed_off(item.id)

    if not claimed:
        message = "目前沒有待派送的 Email 通知。"
    else:
        message = (
            f"已處理 {len(claimed)} 封 Email：交給 Power Automate {handed_off}、"
            f"失敗 {failed}。"
        )

    return NotificationDispatchResponse(
        processed=len(claimed),
        handed_off=handed_off,
        failed=failed,
        message=message,
    )
