"""The unified agent contract.

Every agent is an independent module exposing the same interface: it ``consumes``
one or more :class:`MessageType` values, ``produces`` exactly one, and implements
:meth:`BaseAgent.execute`. The :meth:`BaseAgent.run` template method wraps every
execution with structured logging, timing, error capture and checkpointing, so
agents only contain domain logic (Single Responsibility / Open-Closed).
"""

from __future__ import annotations

import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List

from cryptohedge.core.context import AgentContext
from cryptohedge.core.message import Message, MessageType


@dataclass
class AgentResult:
    """Outcome of an agent execution."""

    agent: str
    message: Message
    success: bool = True
    error: str = ""
    duration_s: float = 0.0
    skipped: bool = False
    artifacts: Dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    """Abstract base class defining the unified agent interface."""

    #: Human-readable, unique agent name (also used as the checkpoint stage id).
    name: str = "base"
    #: Message types this agent is able to handle.
    consumes: List[MessageType] = []
    #: Message type emitted on success.
    produces: MessageType = MessageType.RESPONSE
    #: Blackboard keys this agent writes (persisted for checkpoint/resume).
    checkpoint_keys: List[str] = []

    def __init__(self) -> None:
        self._validate_contract()

    def _validate_contract(self) -> None:
        if not self.name or self.name == "base":
            raise ValueError(f"{type(self).__name__} must define a unique 'name'")
        if not self.consumes:
            raise ValueError(f"Agent '{self.name}' must declare the messages it consumes")

    # ------------------------------------------------------------------ contract
    @abstractmethod
    def execute(self, context: AgentContext, message: Message) -> Message:
        """Perform the agent's work and return the output message.

        Implementations read inputs from ``context``/``message``, write their
        artifacts onto ``context`` (blackboard) and return a message describing
        the result. They must not catch their own fatal errors: the template
        method :meth:`run` handles logging and recovery uniformly.
        """

    def can_handle(self, message: Message) -> bool:
        return message.type in self.consumes

    # ------------------------------------------------------------- template method
    def run(self, context: AgentContext, message: Message) -> AgentResult:
        log = context.logger(self.name)
        resume = context.config.runtime.resume

        if resume and context.checkpoints.is_completed(self.name) and self._can_restore(context):
            self._restore(context)
            log.info("restored from checkpoint", stage=self.name)
            out = Message(self.produces, self.name, "orchestrator", {"restored": True},
                          correlation_id=message.correlation_id)
            return AgentResult(self.name, out, success=True, skipped=True)

        log.info("started", consumes=message.type.value)
        start = time.perf_counter()
        try:
            with log.timer(f"{self.name}.execute"):
                out_message = self.execute(context, message)
            duration = time.perf_counter() - start
            self._persist(context)
            context.checkpoints.mark_completed(self.name, {"duration_s": round(duration, 3)})
            log.info("completed", duration_s=round(duration, 3), produces=out_message.type.value)
            return AgentResult(self.name, out_message, success=True, duration_s=duration)
        except Exception as exc:  # noqa: BLE001 - top-level agent guard
            duration = time.perf_counter() - start
            tb = traceback.format_exc()
            log.error("failed", error=str(exc), traceback=tb, duration_s=round(duration, 3))
            fail = Message(
                MessageType.FAILED,
                self.name,
                "orchestrator",
                {"error": str(exc), "agent": self.name},
                correlation_id=message.correlation_id,
            )
            return AgentResult(self.name, fail, success=False, error=str(exc), duration_s=duration)

    # ----------------------------------------------------------------- checkpoint
    def _can_restore(self, context: AgentContext) -> bool:
        return all(context.checkpoints.exists(self._ckpt_key(k)) for k in self.checkpoint_keys)

    def _restore(self, context: AgentContext) -> None:
        for key in self.checkpoint_keys:
            context.put(key, context.checkpoints.load(self._ckpt_key(key)))

    def _persist(self, context: AgentContext) -> None:
        for key in self.checkpoint_keys:
            if context.has(key):
                context.checkpoints.save(self._ckpt_key(key), context.get(key))

    def _ckpt_key(self, blackboard_key: str) -> str:
        return f"{self.name}__{blackboard_key}"
