"""Native EADream Atari-16 reproduction package."""

# Keep the package import side-effect free.  In particular, ``python -m``
# fixture modules must not preload their target module through package import.
__all__ = ["AgentState", "EADream", "EADreamAgent"]


def __getattr__(name: str):
    if name == "EADream":
        from .algorithm import EADream

        return EADream
    if name in {"AgentState", "EADreamAgent"}:
        from .engine.agent import AgentState, EADreamAgent

        return {"AgentState": AgentState, "EADreamAgent": EADreamAgent}[name]
    raise AttributeError(name)
