"""Crash-recovery via checkpointing.

The :class:`CheckpointManager` persists arbitrary artifacts to disk and tracks
which pipeline stages have completed in a JSON manifest. On restart the
orchestrator can skip already-completed stages and reload their outputs, so a run
interrupted by a failure resumes where it left off.
"""

from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


class CheckpointManager:
    def __init__(self, checkpoint_dir: str | Path, run_id: str = "default", enabled: bool = True) -> None:
        self.enabled = enabled
        self.run_dir = Path(checkpoint_dir) / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.run_dir / "manifest.json"
        self._manifest: Dict[str, Any] = self._load_manifest()

    # ------------------------------------------------------------------ manifest
    def _load_manifest(self) -> Dict[str, Any]:
        if self.manifest_path.exists():
            with open(self.manifest_path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        return {"stages": {}, "created": datetime.now(timezone.utc).isoformat()}

    def _save_manifest(self) -> None:
        with open(self.manifest_path, "w", encoding="utf-8") as fh:
            json.dump(self._manifest, fh, indent=2, ensure_ascii=False, default=str)

    def is_completed(self, stage: str) -> bool:
        return self.enabled and stage in self._manifest["stages"]

    def completed_stages(self) -> List[str]:
        return list(self._manifest["stages"])

    def mark_completed(self, stage: str, meta: Optional[Dict[str, Any]] = None) -> None:
        self._manifest["stages"][stage] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "meta": meta or {},
        }
        self._save_manifest()

    def reset(self) -> None:
        self._manifest = {"stages": {}, "created": datetime.now(timezone.utc).isoformat()}
        self._save_manifest()

    # ----------------------------------------------------------------- artifacts
    def save(self, key: str, obj: Any) -> Path:
        """Persist ``obj`` choosing a format based on its type."""
        if isinstance(obj, pd.DataFrame):
            path = self.run_dir / f"{key}.parquet"
            obj.to_parquet(path)
        elif isinstance(obj, (dict, list)) and _is_jsonable(obj):
            path = self.run_dir / f"{key}.json"
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(obj, fh, indent=2, ensure_ascii=False, default=str)
        else:
            path = self.run_dir / f"{key}.pkl"
            with open(path, "wb") as fh:
                pickle.dump(obj, fh)
        return path

    def load(self, key: str) -> Any:
        for suffix, loader in ((".parquet", pd.read_parquet), (".json", _load_json), (".pkl", _load_pickle)):
            path = self.run_dir / f"{key}{suffix}"
            if path.exists():
                return loader(path)
        raise FileNotFoundError(f"No checkpoint artifact for key '{key}' in {self.run_dir}")

    def exists(self, key: str) -> bool:
        return any((self.run_dir / f"{key}{s}").exists() for s in (".parquet", ".json", ".pkl"))


def _is_jsonable(obj: Any) -> bool:
    try:
        json.dumps(obj, default=str)
        return True
    except (TypeError, ValueError):
        return False


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _load_pickle(path: Path) -> Any:
    with open(path, "rb") as fh:
        return pickle.load(fh)
