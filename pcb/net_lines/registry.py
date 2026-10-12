"""Line geometry owner adds versions here only."""
def register_stages(registry):
    from .legacy_v2 import LegacyLines
    registry.register('net_lines','legacy_v2',LegacyLines)
