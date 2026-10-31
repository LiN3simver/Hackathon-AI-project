"""Run one durable active-learning cycle from Task Scheduler or a terminal."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))

from app.schemas.operations import LearningTask, TrainingRunStatus  # noqa: E402
from app.services.operations_store import get_operations_store  # noqa: E402
from app.services.training_automation import (  # noqa: E402
    request_training_run,
    run_training_cycle,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Train below the normal sample limit")
    parser.add_argument("--trigger", default="scheduled")
    parser.add_argument(
        "--task",
        choices=[task.value for task in LearningTask],
        default=LearningTask.EXTERIOR_VIEW.value,
        help="Dataset to train; each dataset has its own runner and threshold",
    )
    args = parser.parse_args()

    store = get_operations_store()
    run = request_training_run(
        store, trigger=args.trigger, force=args.force, task=LearningTask(args.task)
    )
    if run is None:
        raise SystemExit("A training run is already queued or running")
    if run.status == TrainingRunStatus.SKIPPED:
        print(run.model_dump_json(indent=2))
        return
    run_training_cycle(
        run.id,
        str(store.database_path),
        str(store.learning_storage_path),
    )
    completed = store.get_training_run(run.id)
    print(json.dumps(completed.model_dump(mode="json"), ensure_ascii=False, indent=2))
    if completed.status == TrainingRunStatus.FAILED:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
