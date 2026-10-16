"""Human-label template and VLM attribute agreement; no fabricated expert labels."""
import argparse,json
import core
import adaptive_signal as dsp
from visual_condition import SCHEMA,render

def export_template(limit=20):
    samples=[]
    for m in core.records():
        if m['source_type']!='physical_acquisition':continue
        record=m['record_id'];raw=core.load(record)[0];z,_=dsp.calibrate(raw)
        for item in dsp.inspect_conditions(record)['condition_segments']:
            folder=core.RUNS/record;folder.mkdir(exist_ok=True);a,b=item['sample_range'];path=folder/(item['segment_id']+'_visual.png');render(z[:,a:b+1],path)
            samples.append(dict(record=record,physical_source=core.physical_source(record),specimen=record[:2],segment_id=item['segment_id'],image=str(path),
                expert_attributes={k:{f:None for f in fields} for k,fields in SCHEMA.items()},reviewer=None))
            if len(samples)>=limit:break
        if len(samples)>=limit:break
    doc=dict(schema=SCHEMA,samples=samples,status='awaiting human review',note='按物理记录/绳样分组；示例清单不能当独立测试集')
    (core.RUNS/'perception_human_template.json').write_text(json.dumps(doc,ensure_ascii=False,indent=2),encoding='utf-8')
    return doc

def score(labels,predictions):
    predictions={(x['record'],x['segment_id']):x['visual_condition'] for x in predictions if x.get('status')=='completed'}
    counts={};total=0
    for row in labels['samples']:
        pred=predictions.get((row['record'],row['segment_id']))
        if pred is None:continue
        for group,fields in row['expert_attributes'].items():
            for field,value in fields.items():
                if value is None:continue
                if value not in SCHEMA[group][field]:raise ValueError('人工标签不在schema内')
                key=group+'.'+field;correct,n=counts.get(key,(0,0));counts[key]=(correct+(pred[group][field]==value),n+1);total+=1
    return dict(attributes={k:dict(correct=c,total=n,accuracy=c/n) for k,(c,n) in counts.items()},labeled_attribute_count=total,
        status='scored' if total else 'no reviewed labels',confidence_calibration='not established')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--labels');p.add_argument('--predictions');a=p.parse_args()
    if a.labels and a.predictions:print(json.dumps(score(json.loads(open(a.labels,encoding='utf-8').read()),json.loads(open(a.predictions,encoding='utf-8').read())),ensure_ascii=False))
    else:print('Exported',len(export_template()['samples']),'unreviewed samples')
