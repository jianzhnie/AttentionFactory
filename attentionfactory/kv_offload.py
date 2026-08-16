"""Teaching interface for on-disk KV cache offloading."""

from __future__ import annotations

from pathlib import Path

import torch


class OnDiskKVStore:
    """Simple per-sequence on-disk KV cache store.

    This is an interface simulation, not a production storage engine. It
    writes one ``.pt`` file per sequence using ``torch.save``.
    """

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, seq_id: int) -> Path:
        return self.directory / f"seq_{seq_id}.pt"

    def save(self, seq_id: int, key: torch.Tensor, value: torch.Tensor) -> None:
        """Persist key/value tensors for ``seq_id``."""
        torch.save({"key": key, "value": value}, self._path(seq_id))

    def load(self, seq_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Load key/value tensors for ``seq_id``."""
        path = self._path(seq_id)
        if not path.exists():
            raise FileNotFoundError(f"KV cache not found: {path}")
        payload = torch.load(path, weights_only=True)
        return payload["key"], payload["value"]

    def delete(self, seq_id: int) -> None:
        """Delete the on-disk KV cache for ``seq_id``."""
        self._path(seq_id).unlink(missing_ok=True)
