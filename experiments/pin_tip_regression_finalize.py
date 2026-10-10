"""Read-only scoring of fixed saved experiments; no inference or tuning."""
from __future__ import annotations
import ast
from collections import Counter
import csv
import math
from pathlib import Path
import sys

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pin_tip_regression import OUT,PREVIOUS,load_cases,normalized_pairs,pin_semantic_payload
from pin_skeleton_followup import (
    read_json,write_json,verify_formal_state,AccessGuard,strict_ids,constrained_matches,
    _strict_box_pairs,_gt_pins,
)
from pin_skeleton_followup_finalize import overlay
from pcb.io import read_image
from pcb.coordinates import target_bbox_to_opencv
from pcb.submission import validate_strict
from evaluate_v2 import evaluate_case,aggregate,source_from_name,net_members,pair_set
from tools.run_box_skeleton_experiment import metrics

VARIANTS=('A','N','TN','G','E')


def geometry_guard(old,new):
    """Check every original candidate; allow only box tip normal changes."""
    if len(old)!=len(new):raise AssertionError('Component owner count changed')
    count=0
    for a,b in zip(old,new):
        if any(a[k]!=b[k] for k in ('component','type')):raise AssertionError('Owner order/type changed')
        if len(a['terminals'])!=len(b['terminals']):raise AssertionError('Candidate count changed')
        for x,y in zip(a['terminals'],b['terminals']):
            if {k:v for k,v in x.items() if k!='tip'}!={k:v for k,v in y.items() if k!='tip'}:
                raise AssertionError('Candidate attributes changed')
            if a['type']!='box' and x!=y:raise AssertionError('Non-box terminal changed')
            tangent=1 if x['side'] in ('left','right') else 0
            if x['tip'][tangent]!=y['tip'][tangent]:raise AssertionError('Candidate tangent changed')
            count+=1
    return count


def main():
    guard=AccessGuard();formal_before=verify_formal_state();cases=load_cases()
    reports={s:read_json(OUT/s/'latest.json') for s in ('design','check')}
    previous={s:read_json(PREVIOUS/('text_'+s)/'latest.json') for s in reports}
    eval_records={v:[] for v in VARIANTS};tp_sets={v:set() for v in VARIANTS}
    loc_counts={v:Counter() for v in VARIANTS}
    per_case=[];pair_diffs=[];pictures=[];remaining_pair_trace=[];validations=0;candidate_checks=0
    for subset,report in reports.items():
        if report['metadata']['target_in_inference_attempts'] or report['metadata']['sealed_attempts']:
            raise AssertionError('Invalid inference access')
        new=Path(report['metadata']['run_directory'])/'predictions'
        old=Path(previous[subset]['metadata']['run_directory'])/'predictions'
        for key in report['metadata']['case_ids']:
            case=cases[key];image=read_image(case.image_path);height,width=image.shape[:2]
            bases={v:(old if v in ('A','N','TN') else new)/v/key for v in VARIANTS}
            pred={v:read_json(path/'result.json') for v,path in bases.items()}
            diag={v:read_json(path/'diagnostics.json') for v,path in bases.items()}
            raw={v:read_json(path/'raw_terminals.json') for v,path in bases.items()}
            for v in VARIANTS:
                validate_strict(pred[v],(width,height));validations+=1
                if v!='A':
                    candidate_checks+=geometry_guard(raw['N'],raw[v])
                    if pin_semantic_payload(pred[v])!=pin_semantic_payload(pred['N']):
                        raise AssertionError('Exported pin semantics changed')
                    if pred[v]['components']!=pred['N']['components']:
                        raise AssertionError('Components changed')
                    for owner in pred['N']['components']:
                        if pred['N']['components'][owner]['type']!='box' and pred[v]['pins'][owner]!=pred['N']['pins'][owner]:
                            raise AssertionError('Non-box exported pin changed')
            with guard.evaluator():
                target=read_json(case.target_path)
                evaluations={v:evaluate_case(pred[v],target,diag[v]) for v in VARIANTS}
                for v in VARIANTS:
                    eval_records[v].append({'case_id':key,'source':source_from_name(case.image_path.name),
                                            'status':'ok','evaluation':evaluations[v]})
                    tp_sets[v].update((key,*r) for r in strict_ids(pred[v],target))
                    loc=next(r for r in (previous[subset] if v=='A' else report)['cases'] if r['case_id']==key)['localization'][v]
                    loc_counts[v].update(loc)
                row={'subset':subset,'case_id':key,'gt_pin_count':evaluations['N']['metrics']['Pin']['gt']}
                for v in VARIANTS:
                    m=evaluations[v]['metrics'];loc=next(r for r in (previous[subset] if v=='A' else report)['cases'] if r['case_id']==key)['localization'][v]
                    row.update({f'{v}_pin_tp':m['Pin']['tp'],f'{v}_pin_f1':m['Pin']['f1'],
                                f'{v}_pair_tp':m['PinPair']['tp'],f'{v}_pair_pred':m['PinPair']['pred'],
                                f'{v}_pair_f1':m['PinPair']['f1'],f'{v}_box_tp_5':loc['tp_5'],
                                f'{v}_box_tp_20':loc['tp_20']})
                per_case.append(row)
                known={f'{owner}.{pin[4:]}' for owner,pins in target['pins'].items() for pin in pins}
                truth=pair_set([net_members(net) for net in target['nets'].values()])
                pairs={v:normalized_pairs(pred[v],target) for v in VARIANTS}
                for v in ('TN','G','E'):
                    added=pairs[v]-pairs['N'];false=added-truth
                    unknown={pair for pair in false if any(ref not in known for ref in pair)}
                    lost_correct=(pairs['N']&truth)-pairs[v]
                    pair_diffs.append({'subset':subset,'case_id':key,'variant':v,
                                       'added_false_pair_count':len(false),
                                       'added_false_pairs_unknown_ref':len(unknown),
                                       'added_false_pairs_valid_refs_disconnected':len(false-unknown),
                                       'lost_correct_pair_count':len(lost_correct),
                                       'lost_correct_pairs':sorted(lost_correct),
                                       'new_valid_ref_false_pair_examples':sorted(false-unknown)[:5],
                                       'palette_N':diag['N']['wire_palette_selected'],
                                       'palette_new':diag[v]['wire_palette_selected']})
                    if v=='G' and lost_correct:
                        for pair in sorted(lost_correct):
                            trace={'case_id':key,'pair':pair,'N':{},'G':{}}
                            for group in ('N','G'):
                                trace[group]['snaps']=[s for s in diag[group]['snaps'] if s['pin'] in pair]
                                trace[group]['net_memberships']=[net['hyperGraph'] for net in pred[group]['nets'].values()
                                                                 if set(net_members(net))&set(pair)]
                            remaining_pair_trace.append(trace)
                # Evaluation-only crops. Case IDs select illustrations, never inference behavior.
                if key in ('0014','0017','0034','0046'):
                    old_by={r['component']:r['terminals'] for r in raw['N']}
                    new_by={r['component']:r['terminals'] for r in raw['G']}
                    ranked=[]
                    for owner,gowner in _strict_box_pairs(pred['N']['components'],target['components']):
                        points=_gt_pins(target,gowner,height)
                        gain=len(constrained_matches(points,new_by[owner],5))-len(constrained_matches(points,old_by[owner],5))
                        loss=len(constrained_matches(points,old_by[owner],20))-len(constrained_matches(points,new_by[owner],20))
                        moved=sum(math.dist(a['tip'],b['tip'])>1e-9 for a,b in zip(old_by[owner],new_by[owner]))
                        ranked.append(((loss if key=='0034' else gain,moved),owner,gowner))
                    if ranked:
                        _,owner,gowner=max(ranked)
                        comp=pred['N']['components'][owner]
                        comp={**comp,'bbox':target_bbox_to_opencv(comp['bbox'],height)}
                        path=OUT/'debug'/f'{key}_{owner}_N_vs_G.png'
                        overlay(image,comp,old_by[owner],new_by[owner],target['pins'].get(gowner,{}),height,path)
                        pictures.append(str(path))
    cross=read_json(OUT/'crossover/latest.json')
    cross_folder=Path(cross['metadata']['run_directory'])
    for key in cross['metadata']['case_ids']:
        h,w=read_image(cases[key].image_path).shape[:2]
        for v in cross['metadata']['definitions']:
            validate_strict(read_json(cross_folder/v/key/'result.json'),(w,h));validations+=1
    combined={v:aggregate(rows) for v,rows in eval_records.items()}
    changes={v:{'gained_vs_N':sorted(tp_sets[v]-tp_sets['N']),
                'lost_vs_N':sorted(tp_sets['N']-tp_sets[v]),
                'gained_vs_A':sorted(tp_sets[v]-tp_sets['A']),
                'lost_vs_A':sorted(tp_sets['A']-tp_sets[v]),
                'lost_vs_TN':sorted(tp_sets['TN']-tp_sets[v])} for v in VARIANTS}
    sources=[ROOT/'experiments'/f for f in ('pin_tip_regression.py','pin_tip_evidence.py','pin_tip_regression_finalize.py')]
    sources.append(ROOT/'tests/pin/test_pin_tip_regression_experiment.py')
    for path in sources:ast.parse(path.read_text(encoding='utf-8'),filename=str(path))
    tests=read_json(OUT/'tests.json')
    if not tests['passed'] or tests['failures'] or tests['errors']:
        raise AssertionError('Test suite did not pass')
    validation={'formal_state_before':formal_before,'formal_state_after':verify_formal_state(),
                'strict_submission_pass':validations,'strict_submission_fail':0,
                'candidate_record_checks':candidate_checks,'candidate_invariants_pass':True,
                'exported_semantics_equal_N_TN_G_E':True,'nonbox_pins_identical':True,
                'source_ast_pass':len(sources),'original_crossover_json_parity_cases':2,
                'original_crossover_json_parity_predictions':4,
                'target_in_inference_attempts':guard.target_attempts,'sealed_attempts':guard.sealed_attempts,
                'fixed_development_case_count':14,'full150_run':False,'sealed_holdout_used':False,
                'formal_config_changed':False,'git_commit':False,'git_push':False,'tests':tests}
    result={'metadata':{'OFFICIAL_SCORE':False,'scope':'fixed previously-used Design 6 + Check 8; not holdout or OOF',
                         'selected_for_further_study':'G','promotion_to_formal':False},
            'decision':'PARTIAL_GAIN_KEEP_EXPERIMENTAL_NO_FULL150_YET',
            'subsets':reports,'combined_14':combined,
            'combined_box_localization':{v:metrics(dict(c)) for v,c in loc_counts.items()},
            'strict_tp_changes':changes,'pair_change_details':pair_diffs,
            'remaining_pair_trace':remaining_pair_trace,
            'validation':validation,'debug_overlays':pictures}
    write_json(OUT/'decision.json',result);write_json(OUT/'validation.json',validation)
    with (OUT/'per_case.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(per_case[0]));writer.writeheader();writer.writerows(per_case)
    lines=['# Pin Tip回归定位与最小修正','',
        '本地非官方诊断；固定14个曾使用过的开发案例，不能作为150例成绩或泛化证明。正式baseline没有切换。','',
        '本轮结论：选择性Tip校正G保留了30个新增严格Pin TP，避免Design的网络指标大幅回退，但Check仍丢失1条正确Pair、0034仍丢失2个20px位置匹配。因此保留G作为下一轮实验候选，不立即晋升正式配置，不运行150例。','',
        '## 1. 回退是怎样产生的','',
        '上一轮N与TN的组件、脚号、脚名、导出key完全相同，只改变box Tip法线位置。变化传入原Wire算法后，会改变颜色选择和端点保护走廊；同时改变原Topology的吸附位置。算法源码虽没改，输入变化足以改变网络。',
        '0014：统一缩短使Pair预测169→247，正确Pair仍4；78个新增诊断FP中73个含GT不存在的引用，5个是GT合法端点之间错误连接。0017：11→270，正确Pair仍8；259个新增诊断FP中249个含未知引用，10个是合法端点错误连接。不能把全部FP都说成物理误接，许多是语义引用不匹配。',
        '2×2输入对照保持Wire/Topology代码不变：0017用旧Wire+新吸附仍有8/11正确/预测Pair，而新Wire+旧吸附变8/95，说明掩膜是主要因素；完整新输入8/270。0014只改变其中任一输入都会触发4/247，两个输入都敏感。此对照只用于归因，没有采用“导出位置与吸附位置分离”的生产方案。','',
        '## 2. 本轮只试两种最小策略','',
        '- G：已有脚号和脚名都被关联、两个OCR置信度≥0.8、非重复key，并且已有P3局部连续笔画支持通过，才用现有max(6,6×scale)校正。其他候选原样保留。阈值来自此前实验，没有用Check重新拟合。',
        '- E：已有candidate沿法线存在明确局部笔画末端、末端后有实测空白、未被OCR文字覆盖才改Tip。连续导线没有明确局部末端时回退原Tip。可见墨迹末端并不保证等于官方GT端点。',
        'A=正式V3；N=上一轮保序语义+脚号框距离+原Tip；TN=N+全部box短Tip；G=N+可靠证据选择性短Tip；E=N+图像局部末端。G不是骨架候选生成器；本轮没有再次增加候选。','',
        '## 3. Design与Check分开看','',
        '|范围|组|严格Pin TP|Pin宏F1|box TP@5|box TP@20|Pair TP/预测|Pair宏F1|',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for subset,report in reports.items():
        for v in ('N','TN','G','E'):
            m=report['end_to_end'][v]['overall']['metrics'];loc=report['localization'][v]
            lines.append(f"|{subset}|{v}|{m['Pin']['tp']}|{m['Pin']['macro_f1']:.6f}|{loc['tp_5']}|{loc['tp_20']}|{m['PinPair']['tp']}/{m['PinPair']['pred']}|{m['PinPair']['macro_f1']:.6f}|")
    lines+=['','Design：G取得81个严格TP，保留TN新增12个中的10个；Pair正确和预测数量恢复为N的22/216。Check：G保留TN的20个新增严格TP，且没有丢失N原有严格TP；但Pair45→44，宏F1仍0.048158→0.046422。',
            '0034的20px定位N/TN/G为8/0/6，最小门控减轻而未消除回归；0129和0143原先各丢1个5px位置，G恢复为N的1与5。G改动少于TN，因此其定位TP@5增长较小，这是对不可靠证据主动保守的代价。',
            'E的两组严格TP均与N相同：Design定位TP@5 9→6，Check10→12，未形成稳定收益，关闭作为下一轮主方案。','',
            '## 4. 合并14例仅用于核对','',
            '|组|严格Pin TP|导出Pin|Pin宏F1|Pin微F1|Pair TP/预测|Pair宏F1|',
            '|---|---:|---:|---:|---:|---:|---:|']
    for v in VARIANTS:
        m=combined[v]['overall']['metrics']
        lines.append(f"|{v}|{m['Pin']['tp']}|{m['Pin']['pred']}|{m['Pin']['macro_f1']:.6f}|{m['Pin']['f1']:.6f}|{m['PinPair']['tp']}/{m['PinPair']['pred']}|{m['PinPair']['macro_f1']:.6f}|")
    gain=len(changes['G']['gained_vs_A']);lost=len(changes['G']['lost_vs_A'])
    lines+=['',f'相对正式A：G新增{gain}个严格Pin TP、丢失{lost}个；相对N：新增{len(changes["G"]["gained_vs_N"])}个、丢失{len(changes["G"]["lost_vs_N"])}个；相对统一短Tip TN少{len(changes["G"]["lost_vs_TN"])}个TP。',
        '宏F1按case等权平均，微F1按全部TP/预测/GT计算；box定位为固定严格matched-box子集的一对一、side一致纯几何指标，不等同全部GT Pin召回。脚号/脚名不参与该几何匹配。','',
        '## 5. 仍需解决的最小问题','',
        '1. Check损失的正确Pair是0011的U2.4—U2.9；两端在N和G均成功吸附（距离均小于0.05px），N同网而G分网。因此不是简单“找不到导线”，还需追踪端点走廊与图连通性变化。原始吸附与网成员见decision.json的remaining_pair_trace；下一轮固定已有参数，不通过大范围调参补分。',
        '2. 0034仍有两个near-match被过度缩短。连续导线的末端不能可靠给出GT pin tip；需要可解释的局部引脚几何证据，而不能用GT extension直接回填。',
        '3. 更多候选依旧不能保证脚号、脚名、位置同时正确。骨架补候选和OCR召回仍是独立问题，当前不继续扩大候选池。',
        '4. G依赖已有语义可靠性，只适合作为独立实验，不应混进正式Localization V3。正式整合前需明确Stage边界，但本轮没有重构公共接口。','',
        '## 6. 验证与文件','',
        f'{tests["tests_run"]}项专项与既有实验测试全部通过；{validations}/{validations}份已保存预测严格格式验证通过；{candidate_checks}次候选记录检查通过；组件、非box Pin、candidate数量/顺序/base/side/method/切向位置保持。209个受保护正式文件SHA256未变。',
        'GT只在评测窗口读取；没有新的OCR/YOLO调用，没有读取0151～0200/Golden/10GT/QuickTest；没有完整150例、训练、commit、push、PR或打包。原手动脚本与配置未改、未调用。',
        '新增实验源码：experiments/pin_tip_regression.py、pin_tip_evidence.py、pin_tip_regression_finalize.py；新增专项测试：tests/pin/test_pin_tip_regression_experiment.py。',
        '结果：decision.json、per_case.csv、validation.json；Design/Check各自的summary.md；saved_audit.json是原回退证据；crossover/latest.json是2×2归因；debug中4张裁剪图展示N/G/GT。测试执行记录另见tests.json。']
    (OUT/'decision.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print({'report':str(OUT/'decision.md'),'pin_gained_vs_A':gain,'pin_lost_vs_A':lost,
           'combined_G':combined['G']['overall']['metrics']['Pin'],'validation':validation},flush=True)


if __name__=='__main__':main()
