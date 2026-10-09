"""Frozen-parameter descriptive benchmark. No parameter fitting or annotation-guided inference."""
import hashlib,json,time
from core import ROOT,DATA,records,physical_source,evaluate,audit,PARAMS

def aggregate(rows,stage):
    tp=sum(r[stage]['tp'] for r in rows); fp=sum(r[stage]['fp'] for r in rows); fn=sum(r[stage]['fn'] for r in rows)
    p=tp/max(tp+fp,1); r=tp/max(tp+fn,1)
    errors=[v for row in rows for v in row[stage]['center_errors_samples']]
    boundaries=[v for row in rows for v in row[stage]['boundary_errors_samples']]
    return dict(records=len(rows),tp=tp,fp=fp,fn=fn,precision=p,recall=r,f1=2*p*r/max(p+r,1e-12),
                mean_matched_center_error_samples=sum(errors)/max(len(errors),1),
                mean_matched_boundary_error_samples=sum(boundaries)/max(len(boundaries),1))

if __name__=='__main__':
    start=time.time(); allrows=records(); integrity=[]; results=[]; checks=[]
    for m in allrows:
        for field,sha in [('data_path','data_sha256'),('annotation_path','annotation_sha256')]:
            good=hashlib.sha256((DATA/m[field]).read_bytes()).hexdigest()==m[sha]
            if not good: integrity.append(m[field])
        checks.append(audit(m['record_id']))
        for mode in ['fast','balanced']:
            for threshold in [.5,.7]:
                results.append(evaluate(m['record_id'],mode,threshold))
        print(m['record_id'],flush=True)
    summary={}
    for source in ['physical_acquisition','derived_angle_rotation']:
        for mode in ['fast','balanced']:
            for threshold in [.5,.7]:
                selected=[r for r in results if r['source_type']==source and r['mode']==mode and r['iou_threshold']==threshold]
                summary[f'{source}/{mode}/IoU{threshold}']={s:aggregate(selected,s) for s in ['coarse','refined']}
    # For future training only. s1/s2 represent specimens; no random image-level split.
    mapping=[dict(record=m['record_id'],source=physical_source(m['record_id']),specimen=m['record_id'][:2],
                  proposed_fold='specimen_s1' if m['record_id'].startswith('s1') else 'specimen_s2') for m in allrows]
    doc=dict(params=PARAMS,integrity_failures=integrity,summary=summary,records=results,audit=checks,
             split_mapping=mapping,elapsed_s=time.time()-start,
             limitations=['未训练、未调参的原型描述性结果，非独立留出测试性能。','全部角度派生视图只用于单独的稳健性观察。',
                          '中心/边界误差仅计算已匹配区间，需同时查看召回率。','标注行索引按0起始、闭区间处理，需数据作者确认。'])
    (ROOT/'benchmark.json').write_text(json.dumps(doc,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
