"""Integration adapters for per-run NetLine selection and optional sidecars."""
from dataclasses import asdict

from ..core.registry import StageRegistry
from ..net_lines.registry import register_stages
from .selection import selected_versions


SPLIT_TOPOLOGIES = frozenset({"v2", "v3", "v3_power_terminals_v1", "v3_power_terminals_v2"})


def runtime_versions(topology, version=None, *, provenance=False):
    """Validate CLI capabilities without loading image models or writing config."""
    if topology not in SPLIT_TOPOLOGIES:
        if version is not None or provenance:
            raise ValueError("NetLine options require v2/v3 or a v3 power-terminal topology")
        return None
    versions = selected_versions()
    if version is not None:
        registry = StageRegistry()
        register_stages(registry)
        registry.create("net_lines", version)
        versions["net_lines"] = version
    return versions


class RecordedLines:
    """Retain only the most recent successful call, outside the public schema."""

    def __init__(self, backend, version):
        self.backend = backend
        self.version = version
        self.record = None

    def run(self, graph, plan):
        self.record = None
        geometry, provenance = self.backend.run_with_provenance(graph, plan)
        self.record = {
            "schema": "net_lines_provenance_v1",
            "version": self.version,
            "coordinate_space": "original_image_top_left_xy",
            "plan": asdict(plan),
            "geometry": asdict(geometry),
            "provenance": asdict(provenance),
        }
        return geometry


def configure_lines(topology, version=None, *, provenance=False):
    """Configure lines before inference; preserve the existing logical backend."""
    if version is None and not provenance:
        return
    if not hasattr(topology, "network_versions") or not hasattr(topology, "lines"):
        raise ValueError("Selected topology does not support independent NetLine options")
    versions = dict(topology.network_versions)
    backend = topology.lines
    if version is not None:
        registry = StageRegistry()
        register_stages(registry)
        backend = registry.create("net_lines", version)
        versions["net_lines"] = version
    if provenance:
        if not callable(getattr(backend, "run_with_provenance", None)):
            raise ValueError("Selected NetLine backend has no provenance capability")
        backend = RecordedLines(backend, versions["net_lines"])
    topology.lines = backend
    topology.network_versions = versions


def provenance_snapshot(topology):
    lines = getattr(topology, "lines", None)
    if not isinstance(lines, RecordedLines) or lines.record is None:
        raise ValueError("No NetLine provenance was recorded for this inference")
    return lines.record
