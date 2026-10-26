"""Line geometry owner adds versions here only."""
def register_stages(registry):
    from .legacy_v2 import LegacyLines
    from .path_v1 import PathLines
    from .path_collinear_v1 import CollinearPathLines
    registry.register('net_lines','legacy_v2',LegacyLines)
    registry.register('net_lines','path_v1',PathLines)
    registry.register('net_lines','path_collinear_v1',CollinearPathLines)
