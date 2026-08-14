"""Educational PyTorch reimplementations of FlashAttention v1-v4.

Each ``faX`` module exposes the same trio of functions — ``forward``,
``backward`` and ``attention`` — built on the shared primitives in
`common`. The implementations mirror the *algorithmic* structure of each
paper (loop order, work partitioning, pipelining, scheduling) while staying
plain, readable PyTorch; no CUDA details are simulated.
"""

from __future__ import annotations

from types import ModuleType

from . import fa1, fa2, fa3, fa4

__all__ = ["VERSION_REGISTRY", "get_version_module", "list_versions"]

VERSION_REGISTRY: dict[str, ModuleType] = {
    "fa1": fa1,
    "fa2": fa2,
    "fa3": fa3,
    "fa4": fa4,
}


def get_version_module(version: str) -> ModuleType:
    """Return the module implementing ``version`` (one of ``fa1``..``fa4``).

    Raises:
        ValueError: If ``version`` is not a known FlashAttention version.
    """
    try:
        return VERSION_REGISTRY[version]
    except KeyError as exc:
        raise ValueError(f"Unknown FlashAttention version: {version}") from exc


def list_versions() -> list[str]:
    """Return the available FlashAttention version keys."""
    return list(VERSION_REGISTRY)
