"""Evidence-based dual reports. Never reads labels or assigns defect probabilities."""
import html,json,secrets,re
from datetime import datetime
import numpy as np

METHOD_NAMES={
 'inspect_cross_section':'轴向环周响应与截面相对方位分析',
 'agentic_detect':'分段多模态规划、信号处理与验证',
 'adaptive_detect':'按分段噪声选择滤波与检测',
 'scan':'三轴漏磁候选扫描',
 'inspect_conditions':'分段噪声形态分析',
 'inspect_visual_condition':'漏磁灰度图视觉感知',
 'tfo_detect':'目标特征图与连通域检测',
 'pyramid_tfo_detect':'金字塔多尺度融合与连通域检测',
 'adaptive_paper2021':'斜向股纹滤波与环周中值背景抑制',
 'adaptive_paper2023':'Kalman去趋势与形态学背景抑制',
 'adaptive_baseline':'移动均值去趋势与Hilbert包络提取',
 'paper2021':'斜向股纹滤波与环周中值背景抑制',
 'paper2023':'Kalman去趋势与形态学背景抑制',
 'preset_baseline':'移动均值去趋势与Hilbert包络提取',
 'baseline':'移动均值去趋势与Hilbert包络提取',
 'reference_spatial_fusion':'环周空间滤波与三轴响应融合',
 'verified_adaptive_support':'通过验证的自适应响应支持',
 'temporal_channel':'环周空间滤波与三轴响应融合',
 'temporal_energy':'时间能量补充扫描',
 'agentic_dsp':'分段规划、信号处理与执行验证',
 'remove_slow_baseline':'慢变基线去除',
 'suppress_directional_strand_pattern':'斜向股纹滤波',
 'remove_circumferential_background':'环周中值背景抑制',
 'suppress_broad_morphological_background':'形态学宽背景抑制',
 'enhance_local_channel_response':'局部通道匹配增强',
 'extract_hilbert_envelope':'Hilbert包络提取',
 'fuse_multiaxis_response':'多轴响应融合',
 'aggregate_temporal_response':'时间响应聚合',
 'tfo':'目标特征图与连通域定位',
 'pyramid_tfo':'金字塔多尺度融合与连通域定位',
 'fast':'环周空间滤波与三轴响应融合',
 'balanced':'三轴响应融合与时间能量补充扫描',
 'adaptive':'按分段噪声选择滤波与包络检测',
 'agentic':'分段多模态规划与执行验证',
}

def readable(value):
    if isinstance(value,dict):return {k:readable(v) for k,v in value.items()}
    if isinstance(value,list):return [readable(v) for v in value]
    if not isinstance(value,str):return value
    for key in sorted(METHOD_NAMES,key=len,reverse=True):
        value=re.sub(r'(?<![A-Za-z0-9_])'+re.escape(key)+r'(?![A-Za-z0-9_])',METHOD_NAMES[key],value)
    return value

POLICY=dict(version='evidence-confidence-v1',calibrated=False,probability=False,
    description='0–100为工程证据指数，不是缺陷概率；阈值尚未在独立标注集标定。',
    weights=dict(local_response=30,multiaxis_support=20,verified_consensus=20,execution_verification=15,context_quality=15),
    high_minimum=75,medium_minimum=45,axis_ratio_minimum=3.,strong_ratio_minimum=5.,
    context_samples=512,excluded_margin_samples=16,minimum_background_samples=64,
    restrictions='较高等级须至少两轴支持、参考/自适应一致、对应分段验证通过；回退、宽背景、边界或感知冲突时最高中等，自适应补充响应最高较低。')

def assess(candidate,result,tensor,background_mask,*,supplementary=False):
    a,b=candidate['refined_start'],candidate['refined_end'];n=tensor.shape[1]
    if not 0<=a<=b<n:raise ValueError('报告候选范围越界')
    center=(a+b)//2
    segments=[s for s in result.get('agentic_segments',[]) if s['sample_range'][0]<=center<=s['sample_range'][1]]
    filters=[s for s in result.get('filter_segments',[]) if s['sample_range'][0]<=center<=s['sample_range'][1]]
    conditions=result.get('conditions') or {}
    numeric=next((s for s in conditions.get('condition_segments',[]) if s['sample_range'][0]<=center<=s['sample_range'][1]),{})
    lo=max(0,a-POLICY['context_samples']);hi=min(n,b+1+POLICY['context_samples'])
    ids=np.flatnonzero(background_mask[lo:hi])+lo;risks=[];reasons=[]
    ratios=None;parts=dict(local_response=0,multiaxis_support=0,verified_consensus=0,execution_verification=0,context_quality=0)
    if len(ids)>=POLICY['minimum_background_samples']:
        bg=tensor[:,ids,:];median=np.median(bg,axis=1,keepdims=True)
        scale=1.4826*np.median(np.abs(bg-median),axis=1)
        scale=np.maximum(scale,np.maximum(np.median(scale,axis=1,keepdims=True)*.1,1e-6))
        channel_ratio=np.sqrt(np.mean((tensor[:,a:b+1,:]-median)**2,axis=1))/scale
        # Median of the three strongest channels avoids one isolated channel deciding the grade.
        ratios=np.median(np.sort(channel_ratio,axis=1)[:,-3:],axis=1)
        support=int(np.sum(ratios>=POLICY['axis_ratio_minimum']))
        peak=float(np.max(ratios));parts['local_response']=30 if peak>=5 else 15 if peak>=3 else 0
        parts['multiaxis_support']=20 if support>=2 else 8 if support==1 else 0
        reasons.append(f'相邻背景归一化响应比（X/Y/Z）={np.round(ratios,2).tolist()}，{support}轴达到工程支持阈值')
        if support<2:risks.append('多轴支持不足，需检查单轴干扰或弱响应')
    else:support=0;risks.append('相邻非候选背景不足，置信度无法完整评估')
    consensus=bool((candidate.get('boundary_consensus') or {}).get('adaptive_range')) and bool(segments) and all(s['status']=='accepted' for s in segments)
    if consensus:parts['verified_consensus']=20;reasons.append('参考与通过验证的自适应流程在该处一致')
    else:risks.append('未取得通过验证的参考/自适应一致性支持')
    verified=bool(segments) and all(s['status']=='accepted' for s in segments)
    if verified:parts['execution_verification']=15;reasons.append('对应分段执行验证通过（不等于缺陷确诊）')
    fallback=any(s['status']!='accepted' for s in segments) or any(s.get('fallback_reason') for s in filters)
    if fallback:risks.append('对应分段回退或验证未通过')
    broad=numeric.get('shaking_energy_fraction',0)>=.5 and numeric.get('circumferential_coverage',0)>=.75
    edge=a<16 or b>=n-16 or bool(candidate.get('touches_crop_edge'))
    scope=result.get('scope_samples')
    if scope:edge=edge or a<=scope[0] or b>=scope[1]
    if broad:risks.append('周边跨通道背景较强，局部响应可能与背景混合')
    if edge:risks.append('响应接近记录、裁剪或分析范围边界，范围可能不完整')
    if any(s.get('perception_conflicts') for s in segments):risks.append('数值与视觉感知存在冲突')
    if len(ids)>=64:parts['context_quality']=15-(10 if broad else 0)-(5 if edge else 0)
    if supplementary:risks.append('自适应补充响应，可能与主候选重复；尚未纳入主检出')
    score=sum(parts.values()) if ratios is not None else None
    if score is not None:
        if fallback or broad or edge or any(s.get('perception_conflicts') for s in segments):score=min(score,69)
        if supplementary:score=min(score,44)
    high=score is not None and score>=75 and support>=2 and consensus and verified and not fallback and not supplementary
    level='较高' if high else '中等' if score is not None and score>=45 else '较低' if score is not None else '无法评估'
    return dict(level=level,evidence_index=score,calibrated=False,probability=None,
        classification='证据较充分的疑似缺陷' if high else '存疑异常，需人工复核',
        axis_response_background_ratio=None if ratios is None else np.round(ratios,3).tolist(),
        background_sample_count=int(len(ids)),supporting_axes=support,factors=parts,reasons=reasons,uncertainties=risks,
        review_action='优先核查对应实物位置，并通过重复采集确认' if high else '核对原始三轴波形、噪声背景与重复采集结果；必要时调整速度/提离后复测')

def attach_reports(result,record,axis=None):
    import core
    from progress_events import notify
    if 'fast' in result:
        attach_reports(result['fast'],record,axis);attach_reports(result['balanced'],record,axis);return
    if 'candidates' not in result or 'reports' in result:return
    from manual_review import attach_manual_review
    attach_manual_review(result,record)
    from cross_section import attach_cross_sections
    attach_cross_sections(result,record,axis)
    notify('reporting',message='正在评估候选证据并生成专业报告与快速报告')
    z=core.image_tensor(record);n=z.shape[1]
    candidates=result['candidates'];extra=result.get('supplementary_candidates',[])
    mask=np.ones(n,dtype=bool)
    for c in candidates+extra:
        a,b=c['refined_start'],c['refined_end'];mask[max(0,a-16):min(n,b+17)]=False
    rows=[]
    for supplemental,group in ((False,candidates),(True,extra)):
        for c in group:
            confidence=assess(c,result,z,mask,supplementary=supplemental);c['confidence']=confidence
            rows.append(dict(id=c['id'],supplementary=supplemental,sample_range=[c['refined_start'],c['refined_end']],
                time_range_s=[c['start_s'],c['end_s']],position_m=c.get('position_m'),
                sources=readable(c.get('sources',[])),confidence=confidence,cross_section=c.get('cross_section')))
    substantial=[r for r in rows if not r['supplementary'] and r['confidence']['level']=='较高']
    uncertain=sorted([r for r in rows if not r['supplementary'] and r not in substantial],key=lambda r:-(r['confidence']['evidence_index'] or 0))
    supplement=[r for r in rows if r['supplementary']]
    audit=result.get('audit') or core.audit(record)
    scope=result.get('scope_samples')
    coverage=f'采样点 {scope[0]}–{scope[1]}（局部检测）' if scope else f'整条记录，{n}个采样点，名义时长{n/500:.3f}秒'
    conclusion=f'检出{len(candidates)}处主候选：证据较充分{len(substantial)}处，存疑待复核{len(uncertain)}处；另有{len(extra)}处自适应补充响应。'
    if not candidates:conclusion+='本次未检出主候选，不能据此认定无缺陷。'
    limits=list(dict.fromkeys(audit.get('warnings',[])+result.get('limitations',[])+[
        '尚未由人工或实物检测确认；不判定断丝数量、真实损伤长度或安全使用等级。',
        '时间为采样点/500的名义值；位置以采样点为准。',POLICY['description'],
        '两条处理流程使用同一批传感器数据，结果一致不代表独立测量验证。',
        '响应区间表示漏磁信号范围；复核现场需同时考虑范围和周边位置。']))
    methods=[]
    for s in result.get('agentic_segments',[]):
        final=s['final_executed_pipeline']
        methods.append(dict(segment=s['segment_id'],sample_range=s['sample_range'],status=s['status'],
            executed=readable(final.get('selected_pipeline',[])),parameters=final.get('actual_parameters',{}),
            fallback_reason=readable(s.get('fallback_reason')),attempts=s['attempts'],
            verification=[dict(attempt=h.get('attempt'),verdict=(h.get('verification') or {}).get('verdict'),warnings=(h.get('verification') or {}).get('warnings',[])) for h in s.get('history',[])]))
    for s in result.get('filter_segments',[]):
        methods.append(dict(segment=s['segment_id'],sample_range=s['sample_range'],executed=readable(s['used']),
            attempted_operations=readable(s.get('operations',[])),parameters=s.get('parameters'),fallback_reason=s.get('fallback_reason')))
    if not methods:methods=[dict(executed=readable(result.get('tools',[result.get('mode','')])),parameters=result.get('params',{}))]
    condition_segments=(result.get('conditions') or {}).get('condition_segments',[])
    broad_count=sum(s.get('shaking_energy_fraction',0)>=.5 and s.get('circumferential_coverage',0)>=.75 for s in condition_segments)
    overview=f'已完成{len(condition_segments)}段噪声形态分析，其中{broad_count}段具有较强跨通道背景线索。' if condition_segments else '未进行独立分段噪声形态分析。'
    if result.get('processing_status')=='degraded':overview+='部分处理方案回退，需结合原始信号复核。'
    common=dict(record=record,created_at=datetime.now().isoformat(timespec='seconds'),coverage=coverage,conclusion=conclusion,signal_overview=overview,
        substantial=substantial,uncertain=uncertain,supplementary=supplement,limitations=readable(limits),
        confidence_policy=POLICY,primary_count=len(candidates),confirmed_defect_count=None,
        image=result.get('image'),manual_review=result.get('manual_review'),cross_sections=result.get('cross_sections'),run_id=result.get('run_id') or (result.get('workflow') or {}).get('run_id'))
    quick=dict(common,title='检测人员快速阅读报告',
        next_actions=['先核查证据较充分的疑似缺陷，再逐一复核存疑异常。',
            '补充响应单独比对主候选，避免重复统计；保留人工复核结论。',
            '缺少实测位置映射时，结合采样点和现场编码器/记录定位，不能直接换算成米。'])
    technical=dict(common,title='钢丝绳漏磁专业检测报告',audit=audit,methods=methods,
        thresholds=result.get('thresholds'),metrics=result.get('metrics'),signal_artifact=result.get('signal_artifact'),
        detector=readable(result.get('detector',result.get('mode',''))),scope_samples=scope,
        confidence_method='对去基线浮点信号，以候选前后各512点且排除候选±16点的背景估计通道MAD；计算各轴最强3通道响应比的中位数。指数由局部响应、多轴支持、经验证的一致性、执行验证和背景质量组成。')
    folder=core.RUNS/record;folder.mkdir(exist_ok=True);prefix='reports_'+secrets.token_hex(6)
    output={}
    for kind,report in (('quick',quick),('professional',technical)):
        base=prefix+'_'+kind
        report['html_url']='/runs/'+record+'/'+base+'.html';report['markdown_url']='/runs/'+record+'/'+base+'.md'
        md=markdown_report(report,kind=='professional');(folder/(base+'.md')).write_text(md,encoding='utf-8')
        (folder/(base+'.html')).write_text(html_report(report,kind=='professional'),encoding='utf-8')
        output[kind]=report
    result['reports']=output;result['confidence_policy']=POLICY
    (folder/(prefix+'.json')).write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')

def row_text(row,professional):
    c=row['confidence'];score='未评估' if c['evidence_index'] is None else str(c['evidence_index'])+'/100'
    a,b=row['time_range_s'];sa,sb=row['sample_range']
    line=f"{row['id']}：{a}–{b}秒，采样点{sa}–{sb}；证据可信程度{c['level']}（指数{score}）。"
    if row['position_m'] is not None:line+=f"实测映射位置{row['position_m']}米。"
    cs=row.get('cross_section')
    if cs:
        if cs['assumed_angle_deg'] is None:line+='环周方位无法判断。'
        else:line+=f"主响应S{cs['peak_channel_1based']}，假设编号下的方位{cs['assumed_angle_deg']}°；径向深度未确定。"
        dm=cs.get('depth_model',{})
        if dm.get('accepted'):line+=f"条件模型中心埋深{dm['model_depth_mm']:.2f}mm（未经标定），可相容范围{dm['compatible_depth_range_mm']}mm。"
        elif dm:line+='深度模型拒绝输出：'+'；'.join(dm.get('reasons',[]))+'。'
    if professional:
        line+=' '.join(c['uncertainties'])+'；'+c['review_action']
        line+='；依据：'+'；'.join(c['reasons'])+'；支持方法：'+'、'.join(row['sources'])+'；评分分项：'+json.dumps(c['factors'],ensure_ascii=False)
    else:
        salient=[x for x in c['uncertainties'] if any(t in x for t in ('宽','边界','背景较强','多轴','背景不足'))]
        if not salient:salient=c['uncertainties'][:1]
        if salient:line+='重点疑点：'+'；'.join(salient[:2])+'。'
        else:line+='优先现场确认。'
    return line

def markdown_report(report,professional):
    lines=['# '+report['title'],'',f"记录：{report['record']}；生成时间：{report['created_at']}",report['coverage'],'',report['conclusion'],report['signal_overview'],'',POLICY['description']]
    for key,title in (('substantial','证据较充分的疑似缺陷'),('uncertain','存疑异常与人工复核清单'),('supplementary','自适应补充响应（不计入主候选，可能重复）')):
        lines+=['','## '+title,'']+[('- '+row_text(r,professional)) for r in report[key]]
        if not report[key]:lines.append('无。')
    if professional:
        lines+=['','## 检测方法与执行记录','',report['confidence_method'],'','```json',json.dumps(dict(audit=report['audit'],methods=report['methods'],thresholds=report['thresholds'],metrics=report['metrics'],confidence_policy=POLICY),ensure_ascii=False,indent=2),'```']
    else:lines+=['','## 下一步','']+['- '+x for x in report['next_actions']]
    lines+=['','## 适用范围与局限','']+['- '+x for x in report['limitations']]
    if report.get('cross_sections'):
        lines+=['','## 截面方位展示','',report['cross_sections']['geometry']['note'],
            f"[逐候选截面图]({report['cross_sections']['html_url']})"]
    if report.get('manual_review'):lines+=['',f"[人工复检：500×500预测位置图与对齐波形]({report['manual_review']['html_url']})"]
    if report['image']:lines+=['',f"[检测证据图]({report['image']})"]
    return '\n'.join(lines)+'\n'

def html_report(report,professional):
    e=lambda x:html.escape(str(x),quote=True)
    parts=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+e(report['title'])+'</title><style>body{font:16px/1.7 system-ui;margin:32px auto;padding:0 20px;max-width:1000px;color:#243746}h1{font-size:26px}section{background:#f2f7f9;border-radius:12px;padding:18px;margin:20px 0}li{margin:14px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}img{max-width:100%}@media print{button{display:none}section{break-inside:avoid}}</style><h1>'+e(report['title'])+'</h1>',
        '<p>'+e(report['record'])+' · '+e(report['created_at'])+'</p><p>'+e(report['coverage'])+'</p><section><strong>'+e(report['conclusion'])+'</strong><p>'+e(report['signal_overview'])+'</p><p>'+e(POLICY['description'])+'</p></section>']
    for key,title in (('substantial','证据较充分的疑似缺陷'),('uncertain','存疑异常与人工复核清单'),('supplementary','自适应补充响应（不计入主候选，可能重复）')):
        parts.append('<h2>'+title+'</h2><ul>'+''.join('<li>'+e(row_text(r,professional))+'</li>' for r in report[key])+'</ul>' if report[key] else '<h2>'+title+'</h2><p>无。</p>')
    if professional:
        parts+=['<h2>检测方法与执行记录</h2><p>'+e(report['confidence_method'])+'</p><pre>'+e(json.dumps(readable(dict(audit=report['audit'],methods=report['methods'],thresholds=report['thresholds'],metrics=report['metrics'],confidence_policy=POLICY)),ensure_ascii=False,indent=2))+'</pre>']
    else:parts+=['<h2>下一步</h2><ul>'+''.join('<li>'+e(x)+'</li>' for x in report['next_actions'])+'</ul>']
    parts+=['<h2>适用范围与局限</h2><ul>'+''.join('<li>'+e(x)+'</li>' for x in report['limitations'])+'</ul>']
    if report.get('cross_sections'):
        cross=report['cross_sections'];parts+=['<h2>候选截面相对方位</h2><p>'+e(cross['geometry']['note'])+'</p><a href="'+e(cross['html_url'])+'">交互查看每个候选的截面方位与三轴分布</a>']
        if professional:
            parts+=['<h3>'+e(c['candidate_id'])+'</h3><img alt="截面方位示意" src="'+e(c['image'])+'">' for c in cross['candidates']]
    if report.get('manual_review'):parts+=['<h2>人工复检证据</h2><a href="'+e(report['manual_review']['html_url'])+'">查看每个候选的500×500灰度位置图与原始、检测输入和实际处理波形</a>']
    if report['image'] and report['image'].startswith('/runs/'):parts+=['<h2>证据图</h2><a href="'+e(report['image'])+'"><img alt="检测证据图" src="'+e(report['image'])+'"></a>']
    parts+=['<p><a href="'+e(report['markdown_url'])+'">保存文字报告</a> · 使用浏览器打印可保存PDF</p></html>']
    return ''.join(parts)
