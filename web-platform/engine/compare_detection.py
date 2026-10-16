"""Reconcile frozen reference/adaptive predictions offline, then score labels.

No cloud model is constructed. Saved prediction hashes identify exact inputs.
"""
import argparse,hashlib,json
from pathlib import Path
import core
from detection_signal import fuse_candidates,CONSENSUS_POLICY
from evaluate_agent import annotations_blocked,write_json,evaluate_record,save_reports
from view_evaluation import render

def reconcile(reference,adaptive):
    record=reference['record']
    if adaptive['record']!=record:raise ValueError('两份预测记录不一致')
    if reference.get('annotations_used') or adaptive.get('annotations_used'):raise ValueError('源预测不能使用标注')
    if reference.get('scope_samples') is not None or adaptive.get('scope_samples') is not None:raise ValueError('需要完整记录预测')
    with annotations_blocked():
        n=len(core.load(record)[0])
        candidates,supplementary,decision=fuse_candidates(reference['candidates'],adaptive['candidates'],adaptive.get('agentic_segments',[]),n_samples=n)
    for i,c in enumerate(candidates,1):
        c.update(id=f'D{i:02d}',start_s=c['refined_start']/500,end_s=c['refined_end']/500)
    result=dict(adaptive,candidates=candidates,supplementary_candidates=supplementary,
        detection_decision=decision,detector_mode='guarded',annotations_used=False,
        prediction_source='offline reconciliation of frozen predictions; no new model calls')
    return result

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--reference',type=Path,required=True);parser.add_argument('--adaptive',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True);parser.add_argument('--records',nargs='+')
    args=parser.parse_args();ids=args.records or json.loads((args.reference/'settings.json').read_text('utf-8'))['records']
    if len(set(ids))!=len(ids):parser.error('记录不能重复')
    folder=args.output_dir
    if folder.exists() and any(folder.iterdir()):parser.error('输出目录非空，请指定新目录')
    folder.mkdir(parents=True,exist_ok=True)
    settings=dict(records=ids,backend='offline_replay',mode='agentic',perception='numeric',
        planner_mode='saved_predictions',detector_mode='guarded',consensus_policy=dict(CONSENSUS_POLICY),
        input_sha256={},limitations=['开发集冻结预测重放，不是独立测试集；模型调用指标继承源预测，不是本次新调用'])
    rows=[];failures=[]
    # Produce and save every prediction before reading any ground truth.
    for record in ids:
        try:
            paths=[p/(record+'_prediction.json') for p in (args.reference,args.adaptive)]
            docs=[json.loads(path.read_text('utf-8')) for path in paths]
            settings['input_sha256'][record]=[hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
            result=reconcile(*docs);write_json(folder/(record+'_prediction.json'),result)
        except (ValueError,KeyError,FileNotFoundError) as exc:failures.append(dict(record=record,error=str(exc)))
    write_json(folder/'settings.json',settings)
    for record in ids:
        path=folder/(record+'_prediction.json')
        if not path.exists():continue
        try:
            row=evaluate_record(record,json.loads(path.read_text('utf-8')));rows.append(row)
            write_json(folder/(record+'_metrics.json'),row)
        except ValueError as exc:failures.append(dict(record=record,error=str(exc)))
    report=save_reports(folder,settings,rows,failures);render(folder)
    print(json.dumps(report['summary'],ensure_ascii=True))
    return 1 if report['incomplete'] else 0

if __name__=='__main__':raise SystemExit(main())
