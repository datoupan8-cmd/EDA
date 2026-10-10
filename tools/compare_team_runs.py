"""Compare two completed local diagnostic runs using the same case list."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--before", type=Path, required=True)
    p.add_argument("--after", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    before = json.loads(a.before.read_text(encoding="utf-8"))
    after = json.loads(a.after.read_text(encoding="utf-8"))
    b_ids = [r['case_id'] for r in before['cases']]
    a_ids = [r['case_id'] for r in after['cases']]
    if b_ids != a_ids or len(b_ids) != len(set(b_ids)):
        raise ValueError("Case lists must match exactly and contain no duplicates")
    if before['failure_count'] or after['failure_count'] or before['device'] != after['device']:
        raise ValueError("Both runs must succeed under the same device condition")
    if before['sealed_holdout_used'] or after['sealed_holdout_used']:
        raise ValueError("Reserved data is prohibited")
    b, c = before['aggregate']['overall'], after['aggregate']['overall']
    lines = ['# 同条件整体对比（本地非官方诊断）', '', f'案例数：{len(b_ids)}', '',
             '| Metric | Before | After | Delta |', '|---|---:|---:|---:|']
    for metric in ('Component','Pin','NetHypergraph','NetLine','PinPair'):
        x, y = b['metrics'][metric]['macro_f1'], c['metrics'][metric]['macro_f1']
        lines.append(f'| {metric} macro F1 | {x:.6f} | {y:.6f} | {y-x:+.6f} |')
    lines.append(f"| FinalScore | {b['FinalScore']:.6f} | {c['FinalScore']:.6f} | {c['FinalScore']-b['FinalScore']:+.6f} |")
    lines += ['', 'Before config:', '```json', json.dumps(before['config'],ensure_ascii=False,indent=2), '```',
              '', 'After config:', '```json', json.dumps(after['config'],ensure_ascii=False,indent=2), '```',
              '', '固定30例属于历史开发样本；150例包含模型训练样本，均不代表独立泛化或官方成绩。']
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(str(a.output.resolve()))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
