import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from app.core.config import get_settings


def configure_logging(component: str) -> logging.Logger:
    """Configure console and UTF-8 rotating file logs once per process."""
    settings = get_settings()
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    log_directory = Path(settings.log_directory)
    log_directory.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    file_handler_name = f"guardian-{component}-file"
    if not any(handler.name == file_handler_name for handler in root_logger.handlers):
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )
        file_handler = RotatingFileHandler(
            log_directory / f"{component}.log",
            maxBytes=5 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.name = file_handler_name
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

        if not any(handler.name == "guardian-console" for handler in root_logger.handlers):
            console_handler = logging.StreamHandler()
            console_handler.name = "guardian-console"
            console_handler.setLevel(level)
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)
    return logging.getLogger(f"guardian.{component}")
