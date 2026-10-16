"""Offline oracle and fixed-detector experiments. This module alone reads GT for v3.
Oracle is segment-level diagnostic selection, not a deployable/test-time router.
"""
import argparse,json,time
import numpy as np
import pandas as pd
import core
import adaptive_signal as dsp
from parameter_guard import guard,TEMPLATES
from dsp_tools import execute
from agentic_dsp import rule_plan

def metrics(pred,truth):
    tp=len(core.match_intervals(pred,truth,.5));fp=len(pred)-tp;fn=len(truth)-tp
    p=tp/max(tp+fp,1);r=tp/max(tp+fn,1)
    return dict(tp=tp,fp=fp,fn=fn,precision=p,recall=r,f1=2*p*r/max(p+r,1e-12))

def summarize(rows):
    tp=sum(x['tp'] for x in rows);fp=sum(x['fp'] for x in rows);fn=sum(x['fn'] for x in rows)
    return dict(tp=tp,fp=fp,fn=fn,f1=2*tp/max(2*tp+fp+fn,1))

def benchmark(ids):
    rows=[];splits=[];start=time.time()
    for record in ids:
        raw=core.load(record)[0];z,_=dsp.calibrate(raw);conditions=dsp.inspect_conditions(record)
        # All alternative outputs are computed before any annotations are read.
        processed=[]
        for item in conditions['condition_segments']:
            a,b=item['sample_range'];lo=max(0,a-256);hi=min(len(raw),b+257);block=z[:,lo:hi];alternatives={}
            for template in TEMPLATES:
                plan=guard(dict(template=template,controls={}),item,hi-lo)
                env=execute(block,plan)[0];alternatives[template]=dsp.detect_envelope(env[:,a-lo:b-lo+1])[0]
            for name in dsp.ALGORITHMS:
                env=np.array([dsp.paper_filter(axis,name,item)[0] for axis in block]);alternatives['preset_'+name]=dsp.detect_envelope(env[:,a-lo:b-lo+1])[0]
            processed.append((item,alternatives))
        label=pd.read_csv(core.DATA/core.manifest(record)['annotation_path'])
        truth=[(int(x.start_row_idx),int(x.end_row_idx)) for x in label.itertuples() if str(x.label).lower()=='anomaly']
        for item,alternatives in processed:
            a,b=item['sample_range'];gt=[(max(a,s)-a,min(b,e)-a) for s,e in truth if s<=b and e>=a]
            scores={name:metrics(intervals,gt) for name,intervals in alternatives.items()}
            # On empty-negative GT, prefer fewer FP then stable template order.
            winner=max(scores,key=lambda name:(scores[name]['f1'],-scores[name]['fp']))
            rule=rule_plan(item,dict(status='disabled'))['template']
            rows.append(dict(record=record,segment_id=item['segment_id'],scores=scores,oracle=winner,rule=rule,
                route_agreement=rule==winner,oracle_gap_segment=scores[winner]['f1']-scores[rule]['f1']))
        splits.append(dict(record=record,physical_source=core.physical_source(record),specimen=record[:2],fold=record[:2]))
        print(record,flush=True)
    names=list(rows[0]['scores']) if rows else []
    totals={name:summarize([r['scores'][name] for r in rows]) for name in names}
    totals['rule']=summarize([r['scores'][r['rule']] for r in rows]);totals['oracle']=summarize([r['scores'][r['oracle']] for r in rows])
    fixed=max(names,key=lambda name:totals[name]['f1']) if names else None
    return dict(rows=rows,summary=totals,fixed_best=fixed,
        oracle_gap_vs_rule=totals['oracle']['f1']-totals['rule']['f1'],split_mapping=splits,elapsed_s=time.time()-start,
        detector='same v2 envelope detector for every variant',annotations_used='offline scoring only after all preprocessing',
        limitations=['开发集描述性结果，非独立测试','分段交界GT裁剪可能重复计数，不等于全记录F1',
            'Oracle逐段最大F1不保证汇总microF1为全局最优','尚缺Qwen密钥与人工工况标签，未运行模型/感知准确率消融'])

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--records',nargs='*');args=parser.parse_args()
    ids=args.records or [r['record_id'] for r in core.records() if r['source_type']=='physical_acquisition']
    doc=benchmark(ids);path=core.RUNS/'v3_oracle_benchmark.json';path.write_text(json.dumps(doc,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(doc['summary'],ensure_ascii=False),flush=True)
