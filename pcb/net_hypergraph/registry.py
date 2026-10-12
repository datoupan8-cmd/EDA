"""Hypergraph owner adds versions here only."""
def register_stages(registry):
    from .legacy_v2 import LegacyHypergraph
    registry.register('net_hypergraph','legacy_v2',LegacyHypergraph)
