"""Official schema export registration; shared contract, no inference logic."""
def register_stages(registry):
    from .stage import OfficialSubmissionStage
    registry.register("submission", "official", OfficialSubmissionStage)
