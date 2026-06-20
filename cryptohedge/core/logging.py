"""Structured logging for agents.

Provides a :class:`StructuredLogger` that records actions, decisions, errors and
operation timings both to the console (human readable) and to a JSONL file
(machine readable, one event per line). A :meth:`StructuredLogger.timer` context
manager measures wall-clock duration of any operation.
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


class _JsonlHandler(logging.Handler):
    """Logging handler that appends a JSON object per record to a file."""

    def __init__(self, path: Path) -> None:
        super().__init__()
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8")

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - io
        payload: Dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "agent": getattr(record, "agent", record.name),
            "event": record.getMessage(),
        }
        extra = getattr(record, "structured", None)
        if isinstance(extra, dict):
            payload.update(extra)
        try:
            self._fh.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
            self._fh.flush()
        except Exception:
            pass

    def close(self) -> None:  # pragma: no cover - io
        try:
            self._fh.close()
        finally:
            super().close()


@dataclass
class LoggerFactory:
    """Creates per-agent :class:`StructuredLogger` instances sharing one JSONL sink."""

    log_dir: Path
    level: str = "INFO"
    console: bool = True
    jsonl: bool = True
    file_name: str = "cryptohedge.jsonl"
    timing: bool = True
    _root: logging.Logger = field(init=False)

    def __post_init__(self) -> None:
        self.log_dir = Path(self.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._root = logging.getLogger("cryptohedge")
        self._root.setLevel(_LEVELS.get(self.level.upper(), logging.INFO))
        # close existing handlers before dropping them so file handles are released
        # (prevents leaked handles / Windows file locks when contexts are recreated)
        for handler in list(self._root.handlers):
            try:
                handler.close()
            finally:
                self._root.removeHandler(handler)
        self._root.propagate = False

        if self.console:
            stream = logging.StreamHandler()
            stream.setFormatter(
                logging.Formatter("%(asctime)s | %(levelname)-7s | %(agent)-22s | %(message)s")
            )
            stream.addFilter(_AgentDefaultFilter())
            self._root.addHandler(stream)

        if self.jsonl:
            self._root.addHandler(_JsonlHandler(self.log_dir / self.file_name))

    def get(self, agent_name: str) -> "StructuredLogger":
        return StructuredLogger(agent_name, self._root, timing=self.timing)


class _AgentDefaultFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "agent"):
            record.agent = record.name
        return True


class StructuredLogger:
    """Thin wrapper that attaches the agent name and structured fields to records."""

    def __init__(self, agent_name: str, logger: logging.Logger, timing: bool = True) -> None:
        self.agent_name = agent_name
        self._logger = logger
        self._timing = timing
        self.timings: Dict[str, float] = {}

    def _log(self, level: int, event: str, **fields: Any) -> None:
        self._logger.log(level, event, extra={"agent": self.agent_name, "structured": fields})

    def debug(self, event: str, **fields: Any) -> None:
        self._log(logging.DEBUG, event, **fields)

    def info(self, event: str, **fields: Any) -> None:
        self._log(logging.INFO, event, **fields)

    def warning(self, event: str, **fields: Any) -> None:
        self._log(logging.WARNING, event, **fields)

    def error(self, event: str, **fields: Any) -> None:
        self._log(logging.ERROR, event, **fields)

    def decision(self, what: str, **fields: Any) -> None:
        """Log a decision taken by the agent with its quantitative justification."""
        self._log(logging.INFO, f"DECISION: {what}", kind="decision", **fields)

    @contextmanager
    def timer(self, operation: str, **fields: Any) -> Iterator[None]:
        """Context manager measuring and logging the duration of ``operation``."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.timings[operation] = self.timings.get(operation, 0.0) + elapsed
            if self._timing:
                self._log(
                    logging.INFO,
                    f"timing: {operation}",
                    kind="timing",
                    operation=operation,
                    seconds=round(elapsed, 4),
                    **fields,
                )
