"""Generate (or regenerate) the reproducible bundled dataset under ``data/raw``.

Run once before the first pipeline execution (the bundled provider also does this
lazily). Everything is a deterministic function of the configured ``seed``.

Usage:
    python scripts/generate_data.py [--config config] [--root .]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from cryptohedge.core.config import load_config
from cryptohedge.services.providers.bundled import BundledProvider


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the bundled hedging dataset")
    parser.add_argument("--config", default="config")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()

    cfg = load_config(args.config)
    provider = BundledProvider(cfg.data, root=Path(args.root), seed=cfg.seed,
                               n_steps=cfg.horizons.analysis_days)
    print(f"Generating dataset: provider=bundled, universe={cfg.data.universe_size}, "
          f"seed={cfg.seed}, samples={cfg.horizons.analysis_days + 1} ...")
    bundle = provider.materialize()
    print(f"  symbols:            {len(bundle.symbols)}")
    print(f"  spot_close shape:   {bundle.spot_close.shape}")
    print(f"  option rows:        {bundle.option_market_data.shape[0]}")
    print(f"  written to:         {provider.raw_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
