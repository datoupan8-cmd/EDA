__all__ = ["ComponentStageV3", "ComponentStageV4"]


def __getattr__(name):
    # Compatibility exports stay available without eager circular imports.
    from importlib import import_module
    if name in __all__:
        module = "v3" if name == "ComponentStageV3" else "v4"
        return getattr(import_module(f"{__name__}.{module}"), name)
    raise AttributeError(name)
