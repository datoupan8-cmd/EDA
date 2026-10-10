__all__ = ["TopologyStageV1", "TopologyStageV2", "TopologyStageV3"]


def __getattr__(name):
    from importlib import import_module
    if name in __all__:
        return getattr(import_module(f"{__name__}.versions"), name)
    raise AttributeError(name)
