"""NON-OFFICIAL diagnostic evaluator, implementing the published guide approximately.

Exact official net matching costs/aggregation and fallback numbering are not supplied.
Never label these values official competition scores.
"""
import math
import numpy as np
from scipy.optimize import linear_sum_assignment

def f1(tp,pred,gt):return 2*tp/(pred+gt) if pred+gt else 1.0
def members(net):return set(filter(None,net['hyperGraph'][1:-1].split(',')))
def point(p):return p['x'],p['y']
def connected_pairs(sets):
    out=set()
    for memberset in sets:
        values=sorted(memberset)
        out.update((values[i],values[j]) for i in range(len(values)) for j in range(i+1,len(values)))
    return out

def evaluate(pred,gt):
    pc,gc=pred['components'],gt['components'];ct=0;pt=0
    for key,c in pc.items():
        g=gc.get(key)
        if not g or c['type']!=g['type']:continue
        if max(abs(a-b) for a,b in zip(c['bbox'],g['bbox']))>20:continue
        field='Name' if c['type']=='ic' else 'value' if c['type'] in ('resistor','capacitor') else None
        if field and c.get(field)!=g.get(field):continue
        ct+=1
    pcount=sum(map(len,pred['pins'].values()));gcount=sum(map(len,gt['pins'].values()))
    for key,pins in pred['pins'].items():
        for no,p in pins.items():
            g=gt['pins'].get(key,{}).get(no)
            if g and p['pinname']==g['pinname'] and math.dist(point(p['point']),point(g['point']))<=5:pt+=1
    pn=list(pred['nets'].values());gn=list(gt['nets'].values());ps=[members(n) for n in pn];gs=[members(n) for n in gn]
    costs=np.zeros((len(ps),len(gs)));nt=0;lt=0
    for i,a in enumerate(ps):
        for j,b in enumerate(gs):costs[i,j]=-f1(len(a&b),len(a),len(b))
    pairs=list(zip(*linear_sum_assignment(costs))) if costs.size else []
    for i,j in pairs:
        if not ps[i]&gs[j]:continue
        nt+=len(ps[i]&gs[j])
        pl=list(pn[i]['edges'].values());gl=list(gn[j]['edges'].values())
        matches=np.zeros((len(pl),len(gl)),dtype=bool)
        for a,(p1,p2) in enumerate(pl):
            for b,(g1,g2) in enumerate(gl):
                direct=max(math.dist(point(p1),point(g1)),math.dist(point(p2),point(g2)))
                reverse=max(math.dist(point(p1),point(g2)),math.dist(point(p2),point(g1)))
                matches[a,b]=min(direct,reverse)<=5
        if matches.size:
            x,y=linear_sum_assignment(-matches.astype(int));lt+=int(matches[x,y].sum())
    ppairs,gpairs=connected_pairs(ps),connected_pairs(gs);pairtp=len(ppairs&gpairs)
    exact=sum(1 for a in ps if a in gs);key_matches=sum(k in gc for k in pc);type_matches=sum(k in gc and pc[k]['type']==gc[k]['type'] for k in pc)
    bbox_matches=sum(k in gc and pc[k]['type']==gc[k]['type'] and max(abs(a-b) for a,b in zip(pc[k]['bbox'],gc[k]['bbox']))<=20 for k in pc)
    number_matches=name_matches=position_matches=0
    for key,pins in pred['pins'].items():
        for no,p in pins.items():
            g=gt['pins'].get(key,{}).get(no)
            if g:
                number_matches+=1;name_matches+=p['pinname']==g['pinname'];position_matches+=math.dist(point(p['point']),point(g['point']))<=5
    result={'ComponentF1':f1(ct,len(pc),len(gc)),'PinF1':f1(pt,pcount,gcount),
        'NetHypergraphF1':f1(nt,sum(map(len,ps)),sum(map(len,gs))),
        'NetLineF1':f1(lt,sum(len(n['edges']) for n in pn),sum(len(n['edges']) for n in gn))}
    result['diagnostic_layers']={'component_key_precision':key_matches/max(1,len(pc)),'component_key_recall':key_matches/max(1,len(gc)),
        'component_type_precision':type_matches/max(1,len(pc)),'component_type_recall':type_matches/max(1,len(gc)),'component_bbox_precision':bbox_matches/max(1,len(pc)),'component_bbox_recall':bbox_matches/max(1,len(gc)),
        'pin_number_precision':number_matches/max(1,pcount),'pin_number_accuracy_over_gt':number_matches/max(1,gcount),'pin_name_accuracy_over_gt':name_matches/max(1,gcount),'pin_position_accuracy_over_gt':position_matches/max(1,gcount),
        'NetPairF1_singleton_resistant':f1(pairtp,len(ppairs),len(gpairs)),'net_pair_false_merges':len(ppairs-gpairs),'net_pair_misses':len(gpairs-ppairs),
        'ExactNetF1':f1(exact,len(ps),len(gs)),'pred_singleton_nets':sum(len(x)==1 for x in ps),'pred_nets':len(ps),'gt_nets':len(gs)}
    result['weighted_diagnostic_score']=100*sum(result[k]*v for k,v in [('ComponentF1',.3),('PinF1',.25),('NetHypergraphF1',.35),('NetLineF1',.1)])
    result['official']=False
    result['note']='OFFICIAL=false. Published-weight approximation; singleton predictions can inflate NetHypergraphF1 because matched pin intersections receive credit. Use NetPairF1_singleton_resistant for error analysis. Exact official aggregation/fallback rules are unavailable.'
    result['counts']={'components_tp':ct,'pins_tp':pt,'net_pin_tp':nt,'lines_tp':lt,'components_pred':len(pc),'components_reference':len(gc),'pins_pred':pcount,'pins_reference':gcount}
    return result
