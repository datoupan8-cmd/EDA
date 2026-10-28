"""Reporting adapter only; peer scoring and assignment remain unchanged."""
from .peer_protocol2055 import evaluate, aggregate as aggregate_scores

VERSION = 'pin_protocol2055-66b9ba31'


def evaluate_case(pred, gt, diagnostics=None):
    return evaluate(pred, gt)


def aggregate(records):
    rows = [r['evaluation'] for r in records if r.get('status') == 'ok']
    result = aggregate_scores(rows)
    return {'overall': result} if result is not None else {}
