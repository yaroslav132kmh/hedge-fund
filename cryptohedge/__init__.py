"""cryptohedge: a multi-agent system for hedging crypto FX (volatility) risk.

The package follows a Clean Architecture layering:

* :mod:`cryptohedge.domain`   - pure, dependency-free domain entities / value objects.
* :mod:`cryptohedge.core`     - application framework: config, logging, messaging,
  checkpointing, the base agent contract and the orchestrator.
* :mod:`cryptohedge.services` - computational use-cases (volatility, correlation,
  calibration, optimization, risk metrics, stop-loss, data providers).
* :mod:`cryptohedge.agents`   - the eleven autonomous agents, each an independent
  module exposing the unified :class:`cryptohedge.core.agent.BaseAgent` interface.

Every random process is seeded from a single configuration value to guarantee
full reproducibility (see :func:`cryptohedge.core.seeding.set_global_seed`).
"""

from __future__ import annotations

__version__ = "1.0.0"

__all__ = ["__version__"]
