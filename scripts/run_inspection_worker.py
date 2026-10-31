import argparse
import logging
import time
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))

from app.services.inspection_analysis import process_next_inspection_job  # noqa: E402
from app.core.logging_config import configure_logging  # noqa: E402
from app.services.operations_store import get_operations_store  # noqa: E402

logger = logging.getLogger("guardian.worker")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process persistent inspection AI jobs.")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Process at most one available job and exit.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=2.0,
        help="Idle polling interval for daemon mode.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging("worker")
    get_operations_store().initialize()
    logger.info("worker_started poll_seconds=%.1f once=%s", args.poll_seconds, args.once)
    while True:
        job = process_next_inspection_job()
        if job is not None:
            logger.info(
                "job_processed id=%s task=%s status=%s attempt=%s/%s error=%r",
                job.id,
                job.task.value,
                job.status.value,
                job.attempts,
                job.max_attempts,
                job.error_message,
            )
        if args.once:
            return 0
        if job is None:
            time.sleep(max(0.2, args.poll_seconds))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        logger.info("worker_stopped reason=keyboard_interrupt")
        raise SystemExit(0) from None
