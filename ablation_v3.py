"""Whole-record ablation runner. GT scoring occurs after each completed inference."""
import argparse,json
import pandas as pd
import core
from agentic_dsp import agentic_detect
from evaluation_v3 import metrics,summarize

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--records',nargs='+',required=True)
    p.add_argument('--planner',choices=['rules','cloud'],default='rules')
    p.add_argument('--perception',choices=['numeric','visual','fused'],default='numeric')
    p.add_argument('--verification',choices=['off','fallback','replan'],default='replan')
    p.add_argument('--parameters',choices=['fixed','bounded'],default='bounded');a=p.parse_args()
    if a.planner=='rules' and a.perception=='visual':p.error('visual-only requires cloud planner')
    if a.planner=='cloud':
        from cloud_agent import CloudAgent
        planner=CloudAgent().request
    else:planner=None
    rows=[]
    for record in a.records:
        result=agentic_detect(record,planner=planner,perception=a.perception,verification=a.verification,parameters=a.parameters)
        annotations=pd.read_csv(core.DATA/core.manifest(record)['annotation_path'])
        truth=[(int(r.start_row_idx),int(r.end_row_idx)) for r in annotations.itertuples() if str(r.label).lower()=='anomaly']
        measured=metrics([(c['start'],c['end']) for c in result['candidates']],truth)
        rows.append(dict(record=record,physical_source=core.physical_source(record),specimen=record[:2],metrics=measured,system_metrics=result['metrics'],elapsed_s=result['elapsed_s']))
    name='ablation_'+a.planner+'_'+a.perception+'_'+a.verification+'_'+a.parameters
    doc=dict(settings=vars(a),records=rows,summary=summarize([r['metrics'] for r in rows]),
        note='记录级全局检测；开发集描述性结果，模型/参数在运行前冻结。GT只用于结束后评分。')
    (core.RUNS/(name+'.json')).write_text(json.dumps(doc,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(doc['summary']))
