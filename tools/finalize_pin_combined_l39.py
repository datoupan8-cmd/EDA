"""Write the fixed full-run conclusion and append-only handoff, without inference."""
from __future__ import annotations
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'experiments')]
from pcb.io import write_json
from tools.run_pin_combined_l38 import read
from pin_combined_l38 import digest

OUT = ROOT / 'reports/pin_full_validation_l39'


def main():
    if hasattr(sys.stdout, 'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')
    data = read(OUT / 'complete.json'); verified = read(OUT / 'verification.json')
    if not verified['unchanged'] or not verified['aggregate_reproduced']:
        raise AssertionError('Verify full run before reporting')
    a, b = (data['summary'][label]['overall'] for label in ('before', 'after'))
    case_changes = data['case_change_counts']; pairs = data['pair_changes']
    shifts = data['case_score_changes']; gross = sum(max(0, x['score_delta']) for x in shifts)
    ranked = sorted(shifts, key=lambda x: x['score_delta'], reverse=True)
    post = {'highest_gain_case': ranked[0], 'highest_gain_share_of_gross_positive_gain': ranked[0]['score_delta']/gross if gross else 0,
            'top5_share_of_gross_positive_gain': sum(max(0, x['score_delta']) for x in ranked[:5])/gross if gross else 0,
            'mean_score_delta_excluding_best_case': (150*data['score_delta_points']-ranked[0]['score_delta'])/149,
            'top_gain_cases': ranked[:10], 'regressed_cases': sorted((x for x in shifts if x['score_delta'] < -1e-9), key=lambda x:x['score_delta']),
            'network_wrong_pair_hotspots': sorted(shifts, key=lambda x:x['pair_changes']['incorrect_added'], reverse=True)[:10],
            'network_pair_cost': pairs, 'parameters_changed_after_scores': False,
            'OFFICIAL_SCORE': False, 'sealed_holdout_used': False}
    write_json(OUT / 'finite_postmortem.json', post)
    lines = ['# L39：L38组合150例验收结论', '',
        '本轮按运行前计划固定A正式current和B原样L38组合；全部150例从PNG重新执行，允许内容签名缓存。未改变算法、权重、参数、角色规则或评分器。', '',
        '**全部结果为本地非官方诊断，OFFICIAL_SCORE=FALSE。**', '',
        f'完成150/150；综合分 {a["FinalScore"]:.6f} → {b["FinalScore"]:.6f}（{data["score_delta_points"]:+.6f}分）。', '',
        '|指标|A 正式current|B L38组合|Delta|','|---|---:|---:|---:|']
    for family in ('Component', 'Pin', 'NetHypergraph', 'NetLine', 'PinPair'):
        x, y = a['metrics'][family]['macro_f1'], b['metrics'][family]['macro_f1']
        lines.append(f'|{family} macro F1|{x:.6f}|{y:.6f}|{y-x:+.6f}|')
    lines += [f'|综合分|{a["FinalScore"]:.6f}|{b["FinalScore"]:.6f}|{data["score_delta_points"]:+.6f}|', '',
        f'严格Pin TP {a["metrics"]["Pin"]["tp"]} → {b["metrics"]["Pin"]["tp"]}；新增{len(data["gained"])}、损失{len(data["lost"])}。GT Pin总数{b["metrics"]["Pin"]["gt"]}。', '',
        '|Pin micro指标|Before|After|','|---|---:|---:|']
    for key in ('precision','recall','f1'):
        lines.append(f'|{key}|{a["metrics"]["Pin"][key]:.6f}|{b["metrics"]["Pin"][key]:.6f}|')
    box_a,box_b=(data['pin_by_type'][k]['box'] for k in ('before','after'))
    contributions={family:100*weight*(b['metrics'][family]['macro_f1']-a['metrics'][family]['macro_f1'])
                   for family,weight in (('Component',.30),('Pin',.25),('NetHypergraph',.35),('NetLine',.10))}
    if abs(sum(contributions.values())-data['score_delta_points'])>1e-9:
        raise AssertionError('Score contribution accounting differs')
    post['score_metric_contributions']=contributions
    post['highest_pin_gain_case_share']=max(x['pin_gained'] for x in shifts)/len(data['gained']) if data['gained'] else 0
    write_json(OUT/'finite_postmortem.json',post)
    lines += ['',f'复杂box：GT {box_b["gt"]}个Pin，严格正确{box_a["tp"]}→{box_b["tp"]}，Recall {box_a["recall"]:.4%}→{box_b["recall"]:.4%}。403个新增TP均来自box；原R/C等非box输出未变。新增TP是实际严格收益，但6.31%的box召回仍低，不能称复杂IC已经解决。剩余缺口尚不能仅凭总数归因为OCR或定位。', '',
        'PinF1本身不需要网络正确；本地综合分还包括网络和导线指标。各指标对总分Delta的算术贡献如下，不代表已经分离出各算法的因果贡献。', '',
        '|评分项|贡献分数|','|---|---:|']
    for family,value in contributions.items():lines.append(f'|{family}|{value:+.6f}|')
    lines += ['', f'涨分{case_changes["improved"]}例、降分{case_changes["regressed"]}例、不变{case_changes["unchanged"]}例。去掉最高收益{ranked[0]["case_id"]}后，剩余149例平均仍{post["mean_score_delta_excluding_best_case"]:+.6f}分。', '',
        '## 收益来源与训练影响', '',
        '|分组|数量|A综合分|B综合分|Delta|A Pin TP|B Pin TP|','|---|---:|---:|---:|---:|---:|---:|']
    group_names = {'fixed_historical_check30':'历史固定Check30', 'L1_training119':'L1训练119例', 'excluded_duplicate1':'训练排除的重复图1例', 'exclude_design5_145':'去掉Design5的145例', 'outside_previous_L38_35':'此前L38之外115例'}
    for group, summary in data['stratified'].items():
        x,y = (summary[k]['overall'] for k in ('before','after'))
        lines.append(f'|{group_names[group]}|{x["case_count"]}|{x["FinalScore"]:.6f}|{y["FinalScore"]:.6f}|{y["FinalScore"]-x["FinalScore"]:+.6f}|{x["metrics"]["Pin"]["tp"]}|{y["metrics"]["Pin"]["tp"]}|')
    lines += ['', '119张独立图片用于L1模型训练，150例不能被称为泛化评测；Check30不在L1训练中，但已用于历史开发，也不是sealed独立测试。各分组只是诊断，不根据其成绩分case/source选择不同算法。', '',
        '|source|数量|A综合分|B综合分|Delta|A Pin TP|B Pin TP|','|---|---:|---:|---:|---:|---:|---:|']
    for source,x in data['summary']['before'].items():
        if source=='overall':continue
        y=data['summary']['after'][source]
        lines.append(f'|{source}|{x["case_count"]}|{x["FinalScore"]:.6f}|{y["FinalScore"]:.6f}|{y["FinalScore"]-x["FinalScore"]:+.6f}|{x["metrics"]["Pin"]["tp"]}|{y["metrics"]["Pin"]["tp"]}|')
    lines += ['', '## 网络代价', '',
        f'正确Pair新增{pairs["correct_gained"]}、损失{pairs["correct_lost"]}；错误Pair新增{pairs["incorrect_added"]}、消除{pairs["incorrect_removed"]}，净变化{pairs["incorrect_added"]-pairs["incorrect_removed"]:+d}。这是与target的评分集合差异，不代表每条都已人工证明为物理短接。', '',
        '|统计|A|B|','|---|---:|---:|']
    for key in ('pred_pin_count','pred_net_count','pred_edge_count','singleton_net_count','empty_edge_net_count','unattached_pin_count'):
        lines.append(f'|{key}|{a["counts"][key]}|{b["counts"][key]}|')
    for label, block in (('A',a),('B',b)):
        c=block['counts']
        lines.append(f'{label} singleton ratio={c["singleton_net_count"]/c["pred_net_count"]:.6f}；empty-edge ratio={c["empty_edge_net_count"]/c["pred_net_count"]:.6f}；unattached-pin ratio={c["unattached_pin_count"]/c["pred_pin_count"]:.6f}。')
    lines += ['', '|回退case|source|综合分Delta|Pin新增/损失|正确Pair损失|错误Pair新增|','|---|---|---:|---:|---:|---:|']
    for r in post['regressed_cases']:
        lines.append(f'|{r["case_id"]}|{r["source"]}|{r["score_delta"]:+.6f}|{r["pin_gained"]}/{r["pin_lost"]}|{r["pair_changes"]["correct_lost"]}|{r["pair_changes"]["incorrect_added"]}|')
    lines += ['', '## Pin按器件类型的严格结果', '',
        '沿用同一评分器的component key、pin key、pinname及5px点匹配。TP按GT owner type分组，仅报告相同GT分母的Recall。complete.json内pred数量按预测type统计；若预测和GT类型不同，不能把这种混合分组Precision/F1当作严格类型指标。全量macro/micro不受此限制。', '',
        '|GT type|GT Pin|A TP|B TP|A Recall|B Recall|','|---|---:|---:|---:|---:|---:|']
    kinds=sorted(set(data['pin_by_type']['before'])|set(data['pin_by_type']['after']))
    for kind in kinds:
        x,y = (data['pin_by_type'][k][kind] for k in ('before','after'))
        lines.append(f'|{kind}|{x["gt"]}|{x["tp"]}|{y["tp"]}|{x["recall"]:.6f}|{y["recall"]:.6f}|')
    version=read(OUT/'dataset_version_check.json')
    runtime=sum(read(OUT/f'cases/{case:04d}.json')['runtime_seconds_shared_frontend_A_plus_B'] for case in range(1,151))
    lines += ['', '## 可运行性与数据安全', '',
        f'18项测试通过；{verified["protected_files"]}个冻结文件不变，300份prediction hash及150例target已核验，已有35例预测和所有指标严格复现L38。A在0001/0002另外与正式ModularPipeline逐字段一致。', '',
        f'150张图片hash与冻结L1清单一致；{version["known_targets_unchanged"]}例target已有历史签名且不变；缺少历史签名的{version["first_hash_capture_cases"]}仅作本次首次冻结，不宣称其历史逐字节一致。只用当前最新target，推理读取GT为0。', '',
        f'150例共享前端A+B推理时间合计{runtime:.2f}秒。它包含既有缓存命中和首次局部文字读取，不能用作两方案独立冷启动速度对照。此前0103无缓存实测25.44秒，单例不代表全量冷启动。', '',
        '正式current、既有算法、模型和用户未提交修改保持原样；没有commit/push/PR/训练/升级/ZIP/DOCX/Docker。', '',
        'SEALED HOLDOUT USED FOR DEVELOPMENT = FALSE', '',
        '## 决策与下一步', '']
    if data['candidate_retained_for_competition']:
        lines += ['全量综合分净提高，保留B作为比赛候选。局部代价已经计入，不按零回退要求否决。现在具备本地比赛采用证据，下一步应做官方入口、依赖/模型资产、时间和设备条件的环境验收，再确定默认提交入口；本轮尚未运行官方容器，不宣称官方成绩。']
    else:
        lines += ['全量未产生净收益，不将B作为比赛候选。保留冻结记录，按分组/回退成本重新决定瓶颈；不在本轮调参数把分数拉回来。']
    lines += ['', '本轮不展开新的Pin算法实验，不自动按case/source挑选最好结果。后续若继续Pin研究，应以本轮box严格TP/Recall及剩余损失量级决定，不再用局部定位提升代替严格Pin收益。', '',
        '完整数据：complete.json；逐case：per_case.csv；有限复盘：finite_postmortem.json；保护核验：verification.json；最终资产签名：delivery_verification.json。', '']
    (OUT/'conclusion.md').write_text('\n'.join(lines),encoding='utf-8')
    handoff=ROOT/'reports/Pin_Current_Handoff_20261009_L39.md'
    prior=ROOT/'reports/Pin_Retained_Improvements_20261009_L38.json'
    handoff.write_text('\n'.join(['# Pin当前交接：L39冻结组合150例验收', '',
        f'工程：`{ROOT}`。先看reports/pin_full_validation_l39/conclusion.md，再看complete.json和finite_postmortem.json。', '',
        f'150/150，正式current分{a["FinalScore"]:.6f}→B组合{b["FinalScore"]:.6f}，Delta{data["score_delta_points"]:+.6f}。严格Pin TP {a["metrics"]["Pin"]["tp"]}→{b["metrics"]["Pin"]["tp"]}，新增{len(data["gained"])}/损失{len(data["lost"])}；Pin macro F1 {a["metrics"]["Pin"]["macro_f1"]:.6f}→{b["metrics"]["Pin"]["macro_f1"]:.6f}。', '',
        f'Pair成本：正确+{pairs["correct_gained"]}/−{pairs["correct_lost"]}、错误+{pairs["incorrect_added"]}/−{pairs["incorrect_removed"]}。涨/跌/不变={case_changes["improved"]}/{case_changes["regressed"]}/{case_changes["unchanged"]}。完整回退名单在conclusion及finite_postmortem。', '',
        'L38组合与参数完全原样，没有重训或加入其他实验。A/B前端与非box导出Pin一致；旧35例预测/指标一致；全部300输出hash/150target/冻结文件核验通过，18测试通过。单图入口仍为tools/run_pin_combined_l38.py；L39工具只评测，不改变Pipeline。', '',
        f'保留为比赛候选={data["candidate_retained_for_competition"]}；正式current未改。下一步为官方环境/入口资产与运行约束验收，不继续低收益调参。150含119训练图，只是开发诊断；固定30历史Check另列，不称独立sealed泛化。', '',
        '149例target历史hash不变，0142仅首次记录。所有当前reference均为最新target，GT仅在预测保存后由评测读取。没有读取0151～0200/Golden/10GT/QuickTest。', '',
        'OFFICIAL_SCORE=FALSE；SEALED HOLDOUT USED FOR DEVELOPMENT=FALSE；未commit/push/PR/ZIP/DOCX/Docker/安装升级。原用户tracked dirty3文件保留，Git状态见initial_state。', '']),encoding='utf-8')
    retained={'previous_snapshot':{'path':str(prior),'sha256':digest(prior)},'current_handoff':str(handoff),
        'L39':{'report':str(OUT/'conclusion.md'),'metrics':str(OUT/'complete.json'),
               'full150':True,'A_score':a['FinalScore'],'B_score':b['FinalScore'],
               'score_delta_points':data['score_delta_points'],'strict_Pin_TP_A':a['metrics']['Pin']['tp'],
               'strict_Pin_TP_B':b['metrics']['Pin']['tp'],'gained':len(data['gained']),'lost':len(data['lost']),
               'pair_cost':pairs,'candidate_retained':data['candidate_retained_for_competition'],
               'formal_current_changed':False,'official_environment_verified':False},
        'prior_results_preserved':True,'OFFICIAL_SCORE':False,'SEALED_HOLDOUT_USED_FOR_DEVELOPMENT':False}
    write_json(ROOT/'reports/Pin_Retained_Improvements_20261009_L39.json',retained)
    print(json.dumps({'conclusion':str(OUT/'conclusion.md'),'handoff':str(handoff),'score_A':a['FinalScore'],
                     'score_B':b['FinalScore'],'delta':data['score_delta_points']},ensure_ascii=False))


if __name__=='__main__':main()
