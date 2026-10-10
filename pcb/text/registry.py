"""Text-stage registrations, maintained with the component frontend."""
def register_stages(registry):
    from .stage import CurrentTextStage
    registry.register("text", "current", CurrentTextStage)
