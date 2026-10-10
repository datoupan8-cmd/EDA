__all__ = ["CurrentTextStage"]


def __getattr__(name):
    if name == "CurrentTextStage":
        from .stage import CurrentTextStage
        return CurrentTextStage
    raise AttributeError(name)
