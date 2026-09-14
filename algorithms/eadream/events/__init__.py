"""Stateful event extractors for native EADream environments."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .mog2 import MOG2Config, MOG2EventExtractor

__all__ = ["MOG2Config", "MOG2EventExtractor"]


def __getattr__(name: str):
    if name in __all__:
        from . import mog2

        return getattr(mog2, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
