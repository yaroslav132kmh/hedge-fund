"""Shared execution context (blackboard) for the agent pipeline.

The :class:`AgentContext` is the single object threaded through every agent. It
exposes the validated configuration, a per-agent logger factory, the checkpoint
manager, deterministic RNG streams and a shared *blackboard* dictionary that
agents use to publish and consume intermediate artifacts. This keeps agents
decoupled: they depend on the context contract, not on each other.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from cryptohedge.core.checkpoint import CheckpointManager
from cryptohedge.core.config import SystemConfig
from cryptohedge.core.logging import LoggerFactory, StructuredLogger
from cryptohedge.core.seeding import set_global_seed, spawn_rng


class AgentContext:
    def __init__(self, config: SystemConfig, root: str | Path = ".") -> None:
        self.config = config
        self.root = Path(root).resolve()
        config.paths.ensure(self.root)

        self.master_rng = set_global_seed(config.seed)
        self._logger_factory = LoggerFactory(
            log_dir=self.root / config.paths.log_dir,
            level=config.logging.level,
            console=config.logging.console,
            jsonl=config.logging.jsonl,
            file_name=config.logging.file_name,
            timing=config.logging.timing,
        )
        self.checkpoints = CheckpointManager(
            checkpoint_dir=self.root / config.paths.checkpoint_dir,
            run_id=config.run_id,
            enabled=config.runtime.checkpointing,
        )
        self.blackboard: Dict[str, Any] = {}
        self._rng_streams: Dict[str, np.random.Generator] = {}

    # ------------------------------------------------------------------ services
    def logger(self, agent_name: str) -> StructuredLogger:
        return self._logger_factory.get(agent_name)

    def rng(self, name: str) -> np.random.Generator:
        """Return a deterministic, independent RNG stream for the named consumer."""
        if name not in self._rng_streams:
            stream_id = abs(hash(name)) % (2**31)
            self._rng_streams[name] = spawn_rng(self.config.seed, stream_id)
        return self._rng_streams[name]

    # --------------------------------------------------------------- blackboard
    def put(self, key: str, value: Any) -> None:
        self.blackboard[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self.blackboard.get(key, default)

    def require(self, key: str) -> Any:
        if key not in self.blackboard:
            raise KeyError(f"Required artifact '{key}' is missing from the blackboard")
        return self.blackboard[key]

    def has(self, key: str) -> bool:
        return key in self.blackboard

    # ---------------------------------------------------------------- file paths
    def path(self, *parts: str) -> Path:
        p = self.root.joinpath(*parts)
        return p

    def artifact_path(self, relative: str) -> Path:
        p = self.root / self.config.paths.artifacts_dir / relative
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def results_path(self, name: str) -> Path:
        p = self.root / self.config.paths.results_dir / name
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def calibration_path(self, name: str) -> Path:
        p = self.root / self.config.paths.calibration_dir / name
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
