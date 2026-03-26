import json
import logging
import sys
from datetime import datetime, timezone

from app.core.config import LOG_FILE, LOG_LEVEL


class StructuredFormatter(logging.Formatter):
    """
    Formats logs as JSON and injects Tenant Context automatically.
    """

    def format(self, record):
        # Base Data
        log_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
        }

        # Extra Fields (e.g., logger.info(..., extra={"user_id": 123}))
        standard_attributes = {
            "args",
            "asctime",
            "created",
            "exc_info",
            "exc_text",
            "filename",
            "funcName",
            "levelname",
            "levelno",
            "lineno",
            "module",
            "msecs",
            "message",
            "msg",
            "name",
            "pathname",
            "process",
            "processName",
            "relativeCreated",
            "stack_info",
            "thread",
            "threadName",
        }

        for key, value in record.__dict__.items():
            if key not in standard_attributes and key not in log_record:
                log_record[key] = value

        if record.exc_info:
            log_record["stack_trace"] = self.formatException(record.exc_info)

        return json.dumps(log_record)


def get_logger(name: str):
    """
    Returns a logger configured with both File and Stream handlers,
    using JSON formatting.
    """
    logger = logging.getLogger(name)

    log_level = logging.DEBUG if LOG_LEVEL == "DEBUG" else logging.INFO
    # Always set the level (in case env changed)
    logger.setLevel(log_level)

    # prevent adding handlers multiple times if get_logger is called repeatedly
    if logger.hasHandlers():
        # Update existing handler levels too
        for handler in logger.handlers:
            handler.setLevel(log_level)
        return logger

    logger.propagate = False  # Prevent double logging to root

    logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
    logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)

    formatter = StructuredFormatter()

    # 1. Stream Handler (Console)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(log_level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # 2. File Handler (Optional - matches your old config)
    # This creates a "JSON Lines" (NDJSON) file, which is much better for parsing.
    if LOG_FILE:
        file_handler = logging.FileHandler(LOG_FILE)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
