"""In-process message bus and router.

The bus decouples senders from receivers. Agents register the message types they
consume; when a message is published the bus routes it to the matching agents and
records every hop in an auditable message log. This realises the "message routing
between agents" requirement without agents holding references to one another.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, DefaultDict, Dict, List

from cryptohedge.core.agent import BaseAgent
from cryptohedge.core.message import Message, MessageType


@dataclass
class RoutedMessage:
    message: Message
    handled_by: List[str] = field(default_factory=list)


class MessageBus:
    """Routes messages to subscribed agents and keeps a full message trace."""

    def __init__(self) -> None:
        self._agents: Dict[str, BaseAgent] = {}
        self._subscriptions: DefaultDict[MessageType, List[str]] = defaultdict(list)
        self._listeners: List[Callable[[Message], None]] = []
        self.trace: List[RoutedMessage] = []

    def register(self, agent: BaseAgent) -> None:
        if agent.name in self._agents:
            raise ValueError(f"Agent '{agent.name}' already registered")
        self._agents[agent.name] = agent
        for message_type in agent.consumes:
            self._subscriptions[message_type].append(agent.name)

    def add_listener(self, listener: Callable[[Message], None]) -> None:
        """Register an observer invoked for every published message (e.g. tracing)."""
        self._listeners.append(listener)

    def agent(self, name: str) -> BaseAgent:
        return self._agents[name]

    def recipients(self, message: Message) -> List[str]:
        """Resolve the target agents for a message.

        Explicit ``recipient`` wins; otherwise subscriptions to the message type
        are used (topic routing).
        """
        if message.recipient and message.recipient in self._agents:
            return [message.recipient]
        return list(self._subscriptions.get(message.type, []))

    def publish(self, message: Message) -> RoutedMessage:
        routed = RoutedMessage(message=message, handled_by=self.recipients(message))
        self.trace.append(routed)
        for listener in self._listeners:
            listener(message)
        return routed

    def registered_agents(self) -> List[str]:
        return list(self._agents)
