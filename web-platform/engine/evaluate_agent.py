"""Frozen, whole-record detection evaluation; labels are inaccessible during inference.
Run: python evaluate_agent.py (19 physical acquisitions, real configured APIs).
Checkpoints allow --resume RUN_DIRECTORY without re-running completed predictions.
"""
import argparse,csv,hashlib,json,os,sys,time,uuid
from pathlib import Path
from contextlib import contextmanager
import numpy as np
import pandas as pd
import core
from detection_metrics import center_metrics

def write_json(path,data):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');temporary.replace(path)

def event_metrics(pred,truth,threshold=.5):
    pairs=core.match_intervals(pred,truth,threshold)
    used_p={p for p,g in pairs};used_g={g for p,g in pairs}
    matches=[dict(prediction_index=p,truth_index=g,predicted=list(pred[p]),ground_truth=list(truth[g]),
        iou=core.interval_iou(pred[p],truth[g]),center_error_samples=abs(sum(pred[p])-sum(truth[g]))/2,
        start_error_samples=abs(pred[p][0]-truth[g][0]),end_error_samples=abs(pred[p][1]-truth[g][1])) for p,g in pairs]
    tp=len(pairs);fp=len(pred)-tp;fn=len(truth)-tp
    return dict(tp=tp,fp=fp,fn=fn,**ratios(tp,fp,fn),matches=matches,
        false_positives=[dict(index=i,interval=list(x),best_iou=max((core.interval_iou(x,g) for g in truth),default=0)) for i,x in enumerate(pred) if i not in used_p],
        false_negatives=[dict(index=i,interval=list(x),best_iou=max((core.interval_iou(x,p) for p in pred),default=0)) for i,x in enumerate(truth) if i not in used_g])

def ratios(tp,fp,fn):
    # Empty denominators are undefined, never silently reported as perfect scores.
    return dict(precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None,
        f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None)

def masks(intervals,n):
    mask=np.zeros(n,dtype=bool)
    for a,b in intervals:
        if type(a)is not int or type(b)is not int or not 0<=a<=b<n:raise ValueError('区间超出记录或索引非整数')
        mask[a:b+1]=True
    return mask

def point_metrics(pred,truth,n):
    p=masks(pred,n);g=masks(truth,n)
    return point_counts(int(np.sum(p&g)),int(np.sum(p&~g)),int(np.sum(~p&g)),int(np.sum(~p&~g)))

def point_counts(tp,fp,fn,tn):
    values=ratios(tp,fp,fn);sensitivity=values['recall'];specificity=tn/(tn+fp) if tn+fp else None
    return dict(tp=tp,fp=fp,fn=fn,tn=tn,**values,accuracy=(tp+tn)/(tp+fp+fn+tn) if tp+fp+fn+tn else None,
        balanced_accuracy=(sensitivity+specificity)/2 if sensitivity is not None and specificity is not None else None,
        specificity=specificity)

@contextmanager
def annotations_blocked():
    original=pd.read_csv
    def checked(path,*args,**kwargs):
        if 'annotations' in str(path).replace('\\','/').split('/'):
            raise AssertionError('推理阶段试图读取真实标注，已中止')
        return original(path,*args,**kwargs)
    pd.read_csv=checked
    try:yield
    finally:pd.read_csv=original

def frozen_settings(args,ids):
    from api_config import public_config
    from visual_condition import public_config as visual_config
    code={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in core.ROOT.glob('*.py') if not p.name.startswith('test_')}
    rows={m['record_id']:m for m in core.records()}
    unified=args.planner_mode=='unified'
    return dict(records=ids,backend=args.backend,mode=args.mode,perception=args.perception,planner_mode=args.planner_mode,
        verification=args.verification,parameter_mode=args.parameters,detector_mode=args.detector_mode,
        thinking=os.getenv('MFL_ENABLE_THINKING','false').lower() in ('true','1','yes'),
        max_model_calls=args.max_model_calls,iou_thresholds=[.3,.5,.7],
        primary_iou=.5,primary_criterion=getattr(args,'criterion','iou'),cloud=public_config() if args.backend=='cloud' else None,
        vlm=visual_config() if args.mode=='agentic' and args.perception!='numeric' and not unified else None,
        code_sha256=code,dataset_sha256={r:dict(data=rows[r]['data_sha256'],annotations=rows[r]['annotation_sha256']) for r in ids},
        index_convention='0-based inclusive row indices; dataset author confirmation pending')

def evaluate_record(record,prediction):
    if prediction.get('record')!=record:raise ValueError('预测记录与评分记录不一致')
    if prediction.get('annotations_used'):raise ValueError('预测使用了真实标注，不能作为无标注推理结果评分')
    if prediction.get('scope_samples') is not None:raise ValueError('局部预测不能作为全记录结果评分')
    raw=core.load(record)[0];n=len(raw);meta=core.manifest(record)
    if not meta.get('annotation_path'):raise ValueError('这条现场记录没有真实标注，只能查看候选与执行证据，不能计算检出成绩。')
    label_path=core.DATA/meta['annotation_path']
    if hashlib.sha256(label_path.read_bytes()).hexdigest()!=meta['annotation_sha256']:raise ValueError('标注校验和与manifest不一致')
    label=pd.read_csv(label_path)
    required={'label','start_row_idx','end_row_idx'}
    if not required<=set(label):raise ValueError('缺少必要标注字段')
    truth=[]
    for row in label.itertuples():
        if str(row.label).lower()!='anomaly':continue
        a,b=float(row.start_row_idx),float(row.end_row_idx)
        if not np.isfinite([a,b]).all() or not a.is_integer() or not b.is_integer():raise ValueError('标注索引必须为有限整数')
        truth.append((int(a),int(b)))
    pred=[(c['refined_start'],c['refined_end']) for c in prediction['candidates']]
    masks(truth,n);masks(pred,n)
    events={str(t):event_metrics(pred,truth,t) for t in (.3,.5,.7)}
    events['center']=center_metrics(pred,truth)
    segments=prediction.get('agentic_segments',[])
    unified=prediction.get('planner_mode')=='unified'
    expects_visual=prediction.get('perception') in ('visual','fused') and not unified
    visual_complete=sum(s['visual']['status']=='completed' for s in segments) if expects_visual else 0
    return dict(record=record,source_type=meta['source_type'],physical_source=core.physical_source(record),specimen=meta['group'],
        samples=n,ground_truth_count=len(truth),prediction_count=len(pred),truth=[list(x) for x in truth],predictions=[list(x) for x in pred],
        events=events,points=point_metrics(pred,truth,n),system_metrics=prediction.get('metrics',{}),
        visual_segments=len(segments) if expects_visual else 0,visual_completed=visual_complete,
        degraded_visual=expects_visual and bool(segments) and visual_complete<len(segments),elapsed_s=prediction.get('elapsed_s'),
        direct_image_segments=prediction.get('metrics',{}).get('direct_image_segments',0),
        planner_mode=prediction.get('planner_mode','separated'),prediction_run_id=prediction.get('run_id'),
        note='区间级全记录匹配，不裁剪成分段；采样点指标假定所有anomaly区间外均为正常')

def aggregate(rows):
    events={}
    for threshold in ('0.3','0.5','0.7')+(('center',) if all('center' in row['events'] for row in rows) else ()):
        counts={k:sum(row['events'][threshold][k] for row in rows) for k in ('tp','fp','fn')}
        f1s=[row['events'][threshold]['f1'] for row in rows if row['events'][threshold]['f1'] is not None]
        events[threshold]=dict(**counts,**ratios(**counts),macro_f1=float(np.mean(f1s)) if f1s else None)
    counts={k:sum(row['points'][k] for row in rows) for k in ('tp','fp','fn','tn')}
    errors=[pair['center_error_samples'] for row in rows for pair in row['events']['0.5']['matches']]
    return dict(records=len(rows),events=events,points=point_counts(**counts),
        mean_matched_center_error_samples=float(np.mean(errors)) if errors else None,
        mean_matched_center_error_nominal_s=float(np.mean(errors))/500 if errors else None,
        visual_completed=sum(r['visual_completed'] for r in rows),visual_segments=sum(r['visual_segments'] for r in rows),
        degraded_visual_records=sum(r['degraded_visual'] for r in rows),
        direct_image_segments=sum(r.get('direct_image_segments',0) for r in rows),
        model_call_count=sum(r['system_metrics'].get('model_call_count',0) for r in rows),
        prompt_tokens=sum(r['system_metrics'].get('prompt_tokens',0) for r in rows),
        completion_tokens=sum(r['system_metrics'].get('completion_tokens',0) for r in rows),
        model_failures=sum(r['system_metrics'].get('model_failure_count',0) for r in rows))

def pct(value):return '未定义' if value is None else f'{value:.2%}'

def save_reports(folder,settings,rows,failures):
    summary=aggregate(rows)
    primary='center' if settings.get('primary_criterion')=='center' else '0.5'
    description='候选中心落在标注内，一对一匹配' if primary=='center' else 'IoU≥0.5，一对一匹配'
    report=dict(settings=settings,summary=summary,by_specimen={s:aggregate([r for r in rows if r['specimen']==s]) for s in sorted({r['specimen'] for r in rows})},
        results=rows,failures=failures,requested=len(settings['records']),completed=len(rows),
        incomplete=len(rows)!=len(settings['records']),limitations=[
            '当前数据用于过原型开发，属于开发集验证，不是独立测试性能',
            '同物理记录的角度派生视图不作为独立样本，默认仅19条物理采集',
            'IoU是响应范围与标注范围的一致性，不等于物理断口长度精度',
            '0起始闭区间解释仍需数据作者确认；正常点由anomaly区间补集推定',
            '匹配误差只统计已匹配缺陷，必须同时查看召回率',
            'VLM缓存可复用；云端规划非完全确定，结果只代表本次冻结配置',
            '失败记录单独列出，未完成时不得把子集分数称为全数据成绩'])
    write_json(folder/'report.json',report)
    with (folder/'records.csv').open('w',encoding='utf-8-sig',newline='') as f:
        fields=['record','specimen','gt','pred','tp','fp','fn','precision','recall','f1','point_accuracy','balanced_accuracy','visual_completed','visual_segments']
        writer=csv.DictWriter(f,fields);writer.writeheader()
        for r in rows:
            e=r['events'][primary];writer.writerow(dict(record=r['record'],specimen=r['specimen'],gt=r['ground_truth_count'],pred=r['prediction_count'],
                **{k:e[k] for k in ['tp','fp','fn','precision','recall','f1']},point_accuracy=r['points']['accuracy'],balanced_accuracy=r['points']['balanced_accuracy'],
                visual_completed=r['visual_completed'],visual_segments=r['visual_segments']))
    with (folder/'errors.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(['record','type','index','start_sample','end_sample','start_nominal_s','end_nominal_s','best_iou'])
        for r in rows:
            for kind,key in [('误检','false_positives'),('漏检','false_negatives')]:
                for error in r['events'][primary][key]:
                    a,b=error['interval'];writer.writerow([r['record'],kind,error['index'],a,b,a/500,b/500,error['best_iou']])
    e=summary['events'][primary]
    lines=['# 钢绳智能体真实标注评估','',f"已完成 {len(rows)}/{len(settings['records'])} 条，失败 {len(failures)} 条。调度={settings['backend']}，模式={settings['mode']}，感知={settings['perception']}。",'',
        '## 主要结果：'+description,'',f"TP={e['tp']}，FP={e['fp']}，FN={e['fn']}。",
        f"精确率={pct(e['precision'])}，召回率={pct(e['recall'])}，micro F1={pct(e['f1'])}，macro F1={pct(e['macro_f1'])}。",'',
        f"采样点准确率={pct(summary['points']['accuracy'])}；平衡准确率={pct(summary['points']['balanced_accuracy'])}。正常点多时，普通准确率容易掩盖漏检。",'',
        f"独立视觉成功 {summary['visual_completed']}/{summary['visual_segments']} 段；统一模型直接图像规划 {summary['direct_image_segments']} 段。",'',
        f"模型请求 {summary['model_call_count']} 次，失败 {summary['model_failures']} 次；已返回输入/输出 token 为 {summary['prompt_tokens']}/{summary['completion_tokens']}。未知用量不计入，不是账单费用。",'',
        '|记录|真实缺陷|候选|TP|FP|FN|精确率|召回率|F1|','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        e=r['events'][primary];lines.append(f"|{r['record']}|{r['ground_truth_count']}|{r['prediction_count']}|{e['tp']}|{e['fp']}|{e['fn']}|{pct(e['precision'])}|{pct(e['recall'])}|{pct(e['f1'])}|")
    lines+=['','## 计算方法','',
        '检出标准：候选中心落在标注闭区间内，最大基数一对一匹配。重复报警和标注外中心计FP，未匹配标注计FN；不加容差、不扩宽候选。',
        'IoU = 两个闭区间重叠点数 / 并集点数。达到阈值才允许匹配；最大基数二分图匹配保证一个候选/一个标注最多计一次。',
        'Precision=TP/(TP+FP)，Recall=TP/(TP+FN)，F1=2TP/(2TP+FP+FN)。事件级没有自然定义的TN，不能直接计算事件Accuracy。',
        'micro先汇总TP/FP/FN再算；macro按记录F1平均。另给IoU 0.3/0.7诊断，不据此挑最有利阈值。',
        '采样点Accuracy=(TP点+TN点)/全部点数；Balanced Accuracy=(缺陷点召回率+正常点特异度)/2。',
        '缺陷点是anomaly标注闭区间的并集。空分母记null/未定义，不伪造100%。','',
        '## 误检、漏检与限制','',
        'errors.csv逐项列出误检/漏检及最相近IoU；report.json含匹配对、三种IoU指标、绳样分组统计和原始区间。',
        '这次不根据标注重新调参，也不运行Oracle挑路线。真实标注只在检测完成后读取。','']
    lines += ['- '+x for x in report['limitations']]
    if failures:lines+=['','失败记录：']+['- '+r['record']+'：'+r['error'] for r in failures]
    (folder/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    return report

def main():
    p=argparse.ArgumentParser();p.add_argument('--records',nargs='+');p.add_argument('--backend',choices=['cloud','rules'],default='cloud')
    p.add_argument('--mode',choices=['agentic','fast','adaptive','tfo','pyramid_tfo'],default='agentic');p.add_argument('--perception',choices=['numeric','fused'],default='fused')
    p.add_argument('--planner-mode',choices=['separated','unified'],default='separated')
    p.add_argument('--criterion',choices=['center','iou'],default='center')
    p.add_argument('--verification',choices=['off','fallback','replan'],default='replan')
    p.add_argument('--parameters',choices=['fixed','bounded'],default='bounded')
    p.add_argument('--detector-mode',choices=['guarded','legacy'],default='guarded')
    p.add_argument('--max-model-calls',type=int)
    p.add_argument('--output-dir',type=Path);p.add_argument('--score-only',action='store_true')
    p.add_argument('--resume',type=Path);p.add_argument('--include-derived',action='store_true');args=p.parse_args()
    if args.score_only and not args.resume:p.error('--score-only 需要 --resume 指定已保存的预测目录')
    if args.resume and args.output_dir:p.error('--resume 和 --output-dir 不可同时指定')
    if args.max_model_calls is not None and args.max_model_calls<1:p.error('模型调用预算必须大于0')
    if args.score_only:
        folder=args.resume.resolve();settings=json.loads((folder/'settings.json').read_text('utf-8'));ids=settings['records']
    else:
        if args.planner_mode=='unified' and (args.backend!='cloud' or args.mode!='agentic' or args.perception!='fused'):
            p.error('统一多模态规划需要 cloud / agentic / fused')
    if not args.score_only:ids=args.records or [r['record_id'] for r in core.records() if args.include_derived or r['source_type']=='physical_acquisition']
    if len(set(ids))!=len(ids):p.error('记录不可重复')
    for record in ids:
        m=core.manifest(record)
        if m['source_type']!='physical_acquisition' and not args.include_derived and not args.score_only:p.error('派生视图需明确--include-derived；不得混作独立样本')
    if not args.score_only:
        if args.backend=='cloud' and args.mode!='agentic':p.error('fast/adaptive对照使用--backend rules，不冒称云端智能体')
        settings=frozen_settings(args,ids)
        if args.backend=='cloud' and not settings['cloud']['configured']:p.error('未配置调度API')
        if args.perception=='fused' and args.mode=='agentic' and args.planner_mode=='separated' and not settings['vlm']['configured']:p.error('未配置视觉API；分离模式不能称为多模态测试')
        folder=args.resume.resolve() if args.resume else args.output_dir or core.RUNS/'evaluations'/('eval_'+uuid.uuid4().hex[:10])
    folder.mkdir(parents=True,exist_ok=True)
    if args.resume and not args.score_only:
        frozen=json.loads((folder/'settings.json').read_text('utf-8'))
        if frozen!=settings:p.error('配置/代码/数据与断点不一致，请新开评估')
    elif not args.resume:
        if (folder/'settings.json').exists():p.error('评估目录已存在；请使用 --resume 或新目录')
        write_json(folder/'settings.json',settings)
    print('OUTPUT_DIR='+str(folder),flush=True)
    planner=None
    if args.backend=='cloud' and not args.score_only:
        from cloud_agent import CloudAgent
        agent=CloudAgent();agent.max_model_calls=args.max_model_calls;planner=agent.request
    rows=[];failures=[]
    for index,record in enumerate(ids,1):
        try:
            result_file=folder/(record+'_prediction.json')
            if result_file.exists():prediction=json.loads(result_file.read_text('utf-8'));print(f'[{index}/{len(ids)}] {record} checkpoint',flush=True)
            else:
                if args.score_only:raise ValueError('缺少预测快照；仅评分模式不会调用模型补齐')
                print(f'[{index}/{len(ids)}] {record} inference...',flush=True)
                with annotations_blocked():
                    raw_path=core.DATA/core.manifest(record)['data_path']
                    if hashlib.sha256(raw_path.read_bytes()).hexdigest()!=core.manifest(record)['data_sha256']:raise ValueError('信号文件校验和不符')
                    if args.mode=='agentic':
                        from agentic_dsp import agentic_detect
                        prediction=agentic_detect(record,planner=planner,perception=args.perception,planner_mode=args.planner_mode,
                            verification=args.verification,parameters=args.parameters,detector_mode=args.detector_mode)
                    else:prediction=core.analyze(record,args.mode)
                write_json(result_file,prediction)
            measured=evaluate_record(record,prediction);rows.append(measured);write_json(folder/(record+'_metrics.json'),measured)
            key='center' if settings.get('primary_criterion')=='center' else '0.5'
            e=measured['events'][key];print(f"  TP={e['tp']} FP={e['fp']} FN={e['fn']} P={pct(e['precision'])} R={pct(e['recall'])} F1={pct(e['f1'])}",flush=True)
        except Exception as error:
            # No provider response bodies or credentials are written into reports.
            message=str(error) if isinstance(error,(ValueError,AssertionError)) else type(error).__name__
            failures.append(dict(record=record,error=message));print(f'  FAILED: {message}',flush=True)
            from cloud_agent import ModelCallBudgetExceeded
            if isinstance(error,ModelCallBudgetExceeded):
                save_reports(folder,settings,rows,failures);break
        save_reports(folder,settings,rows,failures)
    report=save_reports(folder,settings,rows,failures)
    from view_evaluation import render
    render(folder)
    print(json.dumps(report['summary'],ensure_ascii=True),flush=True)
    print('REPORT='+str(folder/'report.md'),flush=True)
    return 1 if report['incomplete'] else 0

if __name__=='__main__':sys.exit(main())
