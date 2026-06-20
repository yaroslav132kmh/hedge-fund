"""Pipeline orchestrator.

The orchestrator wires the registered agents into a directed flow and drives the
message routing between them. The canonical hedging pipeline is mostly linear
(data -> analysis -> calibration -> greeks -> hedging -> optimization -> risk ->
backtest -> diagnostics -> explanation -> dashboard), but the orchestrator routes
each produced message through the :class:`MessageBus`, supports per-stage
checkpoint/resume and fails fast (or continues) according to configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from cryptohedge.core.agent import AgentResult, BaseAgent
from cryptohedge.core.bus import MessageBus
from cryptohedge.core.context import AgentContext
from cryptohedge.core.message import Message, MessageType


@dataclass
class PipelineStage:
    """A node in the pipeline: an agent and the message type it expects."""

    agent: str
    expects: MessageType


@dataclass
class RunReport:
    success: bool
    results: List[AgentResult] = field(default_factory=list)
    failed_stage: Optional[str] = None

    def by_agent(self) -> Dict[str, AgentResult]:
        return {r.agent: r for r in self.results}

    def total_seconds(self) -> float:
        return sum(r.duration_s for r in self.results)


class Orchestrator:
    def __init__(self, context: AgentContext, fail_fast: bool = True) -> None:
        self.context = context
        self.bus = MessageBus()
        self.fail_fast = fail_fast
        self.pipeline: List[PipelineStage] = []

    def register(self, agent: BaseAgent, expects: Optional[MessageType] = None) -> "Orchestrator":
        self.bus.register(agent)
        expects = expects if expects is not None else (agent.consumes[0] if agent.consumes else MessageType.START)
        self.pipeline.append(PipelineStage(agent.name, expects))
        return self

    def run(self) -> RunReport:
        log = self.context.logger("orchestrator")
        report = RunReport(success=True)
        current = Message(MessageType.START, "orchestrator", self.pipeline[0].agent if self.pipeline else "")
        self.bus.publish(current)

        for stage in self.pipeline:
            agent = self.bus.agent(stage.agent)
            input_message = self._adapt(current, stage.expects, agent.name)
            self.bus.publish(input_message)
            result = agent.run(self.context, input_message)
            report.results.append(result)
            self.bus.publish(result.message)

            if not result.success:
                report.success = False
                report.failed_stage = stage.agent
                log.error("pipeline halted", stage=stage.agent, error=result.error)
                if self.fail_fast:
                    break
            current = result.message

        log.info(
            "pipeline finished",
            success=report.success,
            stages=len(report.results),
            total_seconds=round(report.total_seconds(), 2),
        )
        return report

    def _adapt(self, previous: Message, expected: MessageType, recipient: str) -> Message:
        """Bridge the previous output to the next stage, preserving the payload."""
        return Message(
            type=expected,
            sender=previous.sender,
            recipient=recipient,
            payload=previous.payload,
            correlation_id=previous.correlation_id,
        )
