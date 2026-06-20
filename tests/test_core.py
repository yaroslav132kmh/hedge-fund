"""Unit tests for the core framework: config, seeding, checkpoint, messaging."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cryptohedge.core.agent import BaseAgent
from cryptohedge.core.bus import MessageBus
from cryptohedge.core.checkpoint import CheckpointManager
from cryptohedge.core.config import load_config
from cryptohedge.core.context import AgentContext
from cryptohedge.core.message import Message, MessageType
from cryptohedge.core.orchestrator import Orchestrator
from cryptohedge.core.seeding import set_global_seed, spawn_rng

from conftest import CONFIG_DIR


# -------------------------------------------------------------------- config
def test_config_loads_and_validates():
    cfg = load_config(CONFIG_DIR)
    assert cfg.investment.capital_usd == 10_000_000.0
    assert cfg.data.universe_size == 100
    assert len(cfg.optimization.methods) == 5
    assert set(["pearson", "spearman", "kendall", "dcc_garch", "cointegration"]).issubset(
        set(cfg.market_analysis.correlation.methods))


def test_config_override_applies():
    cfg = load_config(CONFIG_DIR, overrides={"data": {"universe_size": 7}})
    assert cfg.data.universe_size == 7


def test_config_rejects_unknown_keys():
    with pytest.raises(Exception):
        load_config(CONFIG_DIR, overrides={"investment": {"nonexistent_param": 1}})


# -------------------------------------------------------------------- seeding
def test_seeding_is_reproducible():
    set_global_seed(42)
    a = np.random.rand(5)
    set_global_seed(42)
    b = np.random.rand(5)
    assert np.allclose(a, b)


def test_spawn_rng_streams_independent_but_deterministic():
    r1 = spawn_rng(7, 1).random(4)
    r1b = spawn_rng(7, 1).random(4)
    r2 = spawn_rng(7, 2).random(4)
    assert np.allclose(r1, r1b)        # deterministic
    assert not np.allclose(r1, r2)     # independent streams


# ------------------------------------------------------------------ checkpoint
def test_checkpoint_roundtrip(tmp_path):
    cm = CheckpointManager(tmp_path, run_id="t", enabled=True)
    df = pd.DataFrame({"a": [1, 2, 3]})
    cm.save("frame", df)
    cm.save("dict", {"x": 1, "y": [1, 2]})
    cm.save("obj", {"n": np.int64(3)})  # falls back to pickle-able json via default=str
    assert cm.exists("frame") and cm.exists("dict")
    pd.testing.assert_frame_equal(cm.load("frame"), df)
    assert cm.load("dict")["y"] == [1, 2]


def test_checkpoint_manifest_tracks_stages(tmp_path):
    cm = CheckpointManager(tmp_path, run_id="t", enabled=True)
    assert not cm.is_completed("stage1")
    cm.mark_completed("stage1", {"k": 1})
    assert cm.is_completed("stage1")
    cm2 = CheckpointManager(tmp_path, run_id="t", enabled=True)  # reload from disk
    assert cm2.is_completed("stage1")


# --------------------------------------------------------------------- message
def test_message_reply_preserves_correlation():
    m = Message(MessageType.DATA_READY, "a", "b", {"k": 1})
    r = m.reply("b", MessageType.ANALYSIS_READY, {"v": 2})
    assert r.correlation_id == m.correlation_id
    assert r.recipient == "a" and r.sender == "b"


def test_message_is_immutable():
    m = Message(MessageType.START, "a", "b")
    with pytest.raises(Exception):
        m.sender = "c"  # type: ignore[misc]


# -------------------------------------------------------------- bus + orchestrator
class _ProducerAgent(BaseAgent):
    name = "producer"
    consumes = [MessageType.START]
    produces = MessageType.DATA_READY
    checkpoint_keys = ["produced_value"]

    def execute(self, context, message):
        context.put("produced_value", 123)
        return Message(self.produces, self.name, "consumer", {"value": 123},
                       correlation_id=message.correlation_id)


class _ConsumerAgent(BaseAgent):
    name = "consumer"
    consumes = [MessageType.DATA_READY]
    produces = MessageType.COMPLETED
    checkpoint_keys = ["consumed_value"]

    def execute(self, context, message):
        context.put("consumed_value", message.payload["value"] * 2)
        return Message(self.produces, self.name, "orchestrator",
                       {"value": context.get("consumed_value")},
                       correlation_id=message.correlation_id)


def test_bus_routes_by_subscription():
    bus = MessageBus()
    bus.register(_ProducerAgent())
    bus.register(_ConsumerAgent())
    msg = Message(MessageType.DATA_READY, "x", "")
    assert bus.recipients(msg) == ["consumer"]


def test_orchestrator_runs_two_agents(tmp_path):
    cfg = load_config(CONFIG_DIR, overrides={"runtime": {"resume": False}})
    ctx = AgentContext(cfg, root=tmp_path)
    orch = Orchestrator(ctx, fail_fast=True)
    orch.register(_ProducerAgent()).register(_ConsumerAgent())
    report = orch.run()
    assert report.success
    assert ctx.get("consumed_value") == 246
    # message trace captured (routing is auditable)
    assert len(orch.bus.trace) > 0


def test_orchestrator_checkpoint_resume(tmp_path):
    """Second run with resume=True should skip the already-completed stages."""
    base = {"runtime": {"resume": False}}
    ctx = AgentContext(load_config(CONFIG_DIR, overrides=base), root=tmp_path)
    Orchestrator(ctx).register(_ProducerAgent()).register(_ConsumerAgent()).run()

    ctx2 = AgentContext(load_config(CONFIG_DIR, overrides={"runtime": {"resume": True}}), root=tmp_path)
    report2 = Orchestrator(ctx2).register(_ProducerAgent()).register(_ConsumerAgent()).run()
    assert report2.success
    assert all(r.skipped for r in report2.results)        # restored from checkpoint
    assert ctx2.get("consumed_value") == 246              # state recovered
