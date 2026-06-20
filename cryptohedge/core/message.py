"""Message protocol used for inter-agent communication.

Agents never call each other directly; they exchange immutable :class:`Message`
objects through the :class:`cryptohedge.core.bus.MessageBus`. Each message carries
a type, a sender, a recipient, a free-form payload and a correlation id linking a
request to its response, which makes routing and tracing explicit.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional


class MessageType(str, Enum):
    """Semantic type of a message used by the router to dispatch work."""

    # control
    START = "start"
    COMPLETED = "completed"
    FAILED = "failed"
    # data / analysis flow
    DATA_READY = "data_ready"
    ANALYSIS_READY = "analysis_ready"
    CALIBRATION_READY = "calibration_ready"
    GREEKS_READY = "greeks_ready"
    HEDGE_DECISION = "hedge_decision"
    PORTFOLIO_READY = "portfolio_ready"
    RISK_ASSESSMENT = "risk_assessment"
    BACKTEST_READY = "backtest_ready"
    DIAGNOSTIC_READY = "diagnostic_ready"
    EXPLANATION_READY = "explanation_ready"
    DASHBOARD_READY = "dashboard_ready"
    # generic request/response
    REQUEST = "request"
    RESPONSE = "response"


@dataclass(frozen=True)
class Message:
    """An immutable unit of communication between agents."""

    type: MessageType
    sender: str
    recipient: str
    payload: Dict[str, Any] = field(default_factory=dict)
    correlation_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    message_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def reply(
        self,
        sender: str,
        type: MessageType,
        payload: Optional[Dict[str, Any]] = None,
        recipient: Optional[str] = None,
    ) -> "Message":
        """Create a response message preserving the correlation id."""
        return Message(
            type=type,
            sender=sender,
            recipient=recipient or self.sender,
            payload=payload or {},
            correlation_id=self.correlation_id,
        )

    def describe(self) -> str:
        return f"{self.sender} -> {self.recipient} [{self.type.value}] keys={list(self.payload)}"
