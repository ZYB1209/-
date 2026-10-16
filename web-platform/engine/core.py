"""Local tri-axis MFL tool agent. Annotation access is isolated in evaluate()."""
from __future__ import annotations
import csv, hashlib, json, os, re, time
from pathlib import Path
from functools import lru_cache
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from expert_knowledge import lookup_knowledge, knowledge_question,reference_label
from progress_events import notify

ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get('MFL_DATA_DIR') or ROOT / 'dataset' / 'MFL_TriAxial_AllAngles_Dataset')
if not (DATA/'manifest.csv').exists() and (DATA/'MFL_TriAxial_AllAngles_Dataset'/'manifest.csv').exists():
    DATA = DATA/'MFL_TriAxial_AllAngles_Dataset'
RUNS = Path(os.environ.get('MFL_RUNS_DIR') or ROOT / 'runs')
RUNS.mkdir(parents=True,exist_ok=True)
PARAMS = dict(baseline_window=201, rms_window=51, quantile=.925,
              min_samples=40, merge_gap=25, channel_sigma=2.5,
              channel_tolerance=2, refine_ratio=.15, crop_margin=160)

def records():
    with DATA.joinpath('manifest.csv').open(encoding='utf-8-sig', newline='') as f:
        result=list(csv.DictReader(f))
    seen={r['record_id'] for r in result}
    for filename in filter(None,os.getenv('MFL_EXTRA_MANIFESTS','').split(os.pathsep)):
        with Path(filename).open(encoding='utf-8-sig',newline='') as f:
            for row in csv.DictReader(f):
                if row['record_id'] in seen:raise ValueError('附加记录编号重复')
                seen.add(row['record_id']);result.append(row)
    return result

def physical_source(record):
    return re.sub(r'a(?:180|270)from(0|90)', r'a\1', record)

def manifest(record):
    for row in records():
        if row['record_id'] == record:
            return row
    raise ValueError('记录不存在，请从列表选择。')

def smooth(x, w):
    w = min(w, len(x) if len(x) % 2 else len(x)-1)
    if w < 1: return np.array(x, copy=True)
    p = w//2
    a = np.pad(x, [(p,p)] + [(0,0)]*(x.ndim-1), mode='edge')
    s = np.concatenate([np.zeros_like(a[:1]), np.cumsum(a, axis=0)], axis=0)
    return (s[w:]-s[:-w])/w

@lru_cache(maxsize=3)
def load(record):
    m = manifest(record)
    if int(m.get('invalid_samples') or 0)>0:raise ValueError('这条现场记录含无效或时间异常样本；请先查看部署初测的隔离结果，暂不进入整条检测。')
    f = pd.read_csv(DATA / m['data_path'])
    cols = [f'mag_{c}{a}' for a in 'xyz' for c in range(1,33)]
    if any(c not in f for c in cols): raise ValueError('缺少三轴磁通道。')
    x = f[cols].to_numpy(dtype=float).reshape(-1,3,32)
    if not np.isfinite(x).all(): raise ValueError('磁通道存在缺失或非有限值，请先修复数据。')
    if len(x) != int(m['row_count']): raise ValueError('记录行数与 manifest 不一致。')
    # Nominal 500 Hz axis remains explicit; raw timestamp is separately audited.
    t = np.arange(len(x))/500.
    raw = f['time(0.1ms)'].to_numpy(dtype=float)/10000 if 'time(0.1ms)' in f else None
    posfile = DATA / 'positions' / (record+'.csv')
    position = None
    if posfile.exists():
        p = pd.read_csv(posfile)
        ids, vals = p['sample_index'].to_numpy(), p['position_m'].to_numpy(dtype=float)
        if len(ids)<2 or not np.isfinite(vals).all() or not np.all(np.diff(ids)>0) or ids[0]!=0 or ids[-1]!=len(x)-1:
            raise ValueError('位置表须按采样点升序、覆盖完整记录，且位置有限。')
        position = np.interp(np.arange(len(x)), ids, vals)
    return x, t, raw, position

def audit(record):
    x,t,raw,pos = load(record)
    warnings=[]
    stamp={}
    if raw is not None:
        d=np.diff(raw)
        stamp=dict(nonpositive_steps=int(np.sum(d<=0)), median_step_s=float(np.median(d)),
                   max_step_s=float(np.max(d)), raw_duration_s=float(raw[-1]-raw[0]))
        if np.any(d<=0) or abs(np.median(d)-.002)>.0005:
            warnings.append('原始时间戳不完全符合连续 500 Hz；本版本统一使用采样点/500 的名义时间。')
    if pos is None: warnings.append('缺少实测位置映射，不能输出米制缺陷位置或实际断丝长度。')
    return dict(record=record, samples=len(x), axes=3, channels_per_axis=32,
                nominal_sampling_hz=500, nominal_duration_s=len(x)/500,
                source_type=manifest(record)['source_type'], physical_source=physical_source(record),
                timestamp=stamp, has_position=pos is not None, warnings=warnings)

@lru_cache(maxsize=3)
def image_tensor(record):
    """Lossless float signal image, axes x rows x channels; no PNG quantization in detector."""
    x,*_=load(record)
    # Efficient moving-mean baseline, deliberately not claimed to reproduce CFO median baseline.
    r=x-smooth(x,PARAMS['baseline_window'])
    med=np.median(r,axis=0,keepdims=True)
    scale=1.4826*np.median(np.abs(r-med),axis=0,keepdims=True)
    z=(r-med)/np.maximum(scale,1.)
    return np.transpose(z,(1,0,2))

def masks_to_intervals(mask, minlen=40, gap=25):
    edges=np.diff(np.r_[False,mask,False].astype(int))
    spans=list(zip(np.where(edges==1)[0],np.where(edges==-1)[0]-1))
    merged=[]
    for a,b in spans:
        if merged and a-merged[-1][1]-1<=gap: merged[-1][1]=int(b)
        else: merged.append([int(a),int(b)])
    return [s for s in merged if s[1]-s[0]+1>=minlen]

def score_intervals(score, tool):
    threshold=float(np.quantile(score,PARAMS['quantile']))
    spans=masks_to_intervals(score>threshold, PARAMS['min_samples'], PARAMS['merge_gap'])
    return dict(tool=tool, threshold=threshold,
                intervals=[dict(start=a,end=b,tool=tool,peak=float(np.max(score[a:b+1]))) for a,b in spans],
                score=score)

@lru_cache(maxsize=3)
def temporal_channel(record):
    z=image_tensor(record)
    freq=np.fft.rfft(z,axis=2)
    freq[:,:,:2]=0
    high=np.fft.irfft(freq,n=32,axis=2)
    d=np.minimum(np.arange(32),32-np.arange(32))
    kernel=np.exp(-d*d/(2*PARAMS['channel_sigma']**2)); kernel-=kernel.mean()
    kernel/=np.sqrt(np.sum(kernel*kernel))
    response=np.abs(np.fft.irfft(np.fft.rfft(high,axis=2)*np.fft.rfft(kernel),n=32,axis=2))
    near=np.maximum.reduce([np.roll(response,k,axis=2) for k in range(-2,3)])
    fused=np.prod(np.maximum(near,1e-8),axis=0)**(1/3)
    score=np.sqrt(smooth(np.max(fused,axis=1)**2,PARAMS['rms_window']))
    return score_intervals(score,'temporal_channel')

@lru_cache(maxsize=3)
def temporal_energy(record):
    z=image_tensor(record)
    # Complementary axis evidence: no circular matched filter or geometric tri-axis fusion.
    score=np.sqrt(smooth(np.quantile(np.abs(z),.90,axis=(0,2))**2,51))
    return score_intervals(score,'temporal_energy')

def union_candidates(results):
    spans=sorted([s for r in results for s in r['intervals']],key=lambda s:s['start'])
    out=[]
    for s in spans:
        if out and s['start']<=out[-1]['end']+26:
            out[-1]['end']=max(out[-1]['end'],s['end'])
            out[-1]['sources']=sorted(set(out[-1]['sources']+[s['tool']]))
        else: out.append(dict(start=s['start'],end=s['end'],sources=[s['tool']]))
    return out

def refine(record,candidate):
    z=image_tensor(record); a,b=candidate['start'],candidate['end']; n=z.shape[1]
    lo=max(0,a-PARAMS['crop_margin']); hi=min(n-1,b+PARAMS['crop_margin'])
    response=np.quantile(np.abs(z[:,lo:hi+1,:]),.90,axis=(0,2))
    score=np.sqrt(smooth(response**2,31))
    # Local image response boundary around strongest point within coarse interval.
    peak=int(np.argmax(score[a-lo:b-lo+1]))+a-lo
    bg=float(np.quantile(score,.20)); level=bg+PARAMS['refine_ratio']*(float(score[peak])-bg)
    left=right=peak
    while left>0 and score[left-1]>level: left-=1
    while right<len(score)-1 and score[right+1]>level: right+=1
    aa=max(lo,lo+left-20); bb=min(hi,lo+right+20)
    energy=np.mean(z[:,aa:bb+1,:]**2,axis=(1,2))
    # Broad candidates can contain multiple lobes: never collapse them to one peak by default.
    use_refinement=(b-a+1<=150 and bb-aa+1<=.8*(b-a+1) and left>0 and right<len(score)-1)
    return dict(**candidate, coarse_start=a,coarse_end=b,
                refined_start=aa if use_refinement else a,refined_end=bb if use_refinement else b,
                proposed_refinement=[aa,bb],refinement_used=use_refinement,
                refinement_reason='短区间局部边界细化' if use_refinement else '保留原范围；单峰细化可能丢失宽响应或多峰证据',
                axis_response_rms=np.sqrt(energy).round(3).tolist(),
                touches_crop_edge=left==0 or right==len(score)-1,
                status='待复核' if len(candidate['sources'])==1 or left==0 or right==len(score)-1 else '多工具支持，待确认',
                extent_kind='漏磁响应范围，未经实物缺陷长度校准')

def render_image(record,start=0,end=None,candidates=None,name='overview'):
    z=image_tensor(record); end=z.shape[1]-1 if end is None else min(int(end),z.shape[1]-1)
    start=max(0,int(start))
    if end<start: raise ValueError('区间结束点必须大于起始点。')
    width=1100; height=550; plot=(70,45,1030,460)
    img=Image.new('RGB',(width,height),'#101b2b'); dr=ImageDraw.Draw(img)
    try: font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17)
    except OSError: font=ImageFont.load_default()
    dr.text((70,10),f'{record} | nominal 500 Hz | rows {start}..{end}',font=font,fill='white')
    for ai,axis in enumerate('XYZ'):
        # Full-resolution float tensor is source; RGB image only a human view.
        vals=np.clip(z[ai,start:end+1,:].T,-6,6)/6
        rgb=np.empty((*vals.shape,3),dtype=np.uint8)
        rgb[:,:,0]=np.clip(30+220*np.maximum(vals,0),0,255)
        rgb[:,:,1]=np.clip(35+130*(1-np.abs(vals)),0,255)
        rgb[:,:,2]=np.clip(45+205*np.maximum(-vals,0),0,255)
        panel=Image.fromarray(rgb).resize((960,125))
        y=45+ai*145; img.paste(panel,(70,y)); dr.text((12,y+40),axis,font=font,fill='white')
        if candidates:
            for c in candidates:
                a=max(start,c['refined_start']); b=min(end,c['refined_end'])
                if b<a: continue
                xa=70+int((a-start)/max(end-start,1)*959); xb=70+int((b-start)/max(end-start,1)*959)
                dr.rectangle((xa,y,xb,y+124),outline='#ffe36d',width=2)
                if ai==0: dr.text((xa,y+2),c.get('id',''),font=font,fill='#ffe36d')
    dr.text((70,490),'X: sample index / 500 (s)     Y: channels 1..32 per axis',font=font,fill='white')
    dr.text((70,515),f'{start/500:.3f} s  ------------------------------------------------  {end/500:.3f} s',font=font,fill='white')
    folder=RUNS/record; folder.mkdir(exist_ok=True)
    path=folder/(name+'.png'); img.save(path)
    metadata=dict(record=record,start_sample=start,end_sample=end,image_size=[width,height],
                  panel_x=[70,1029],pixel_to_sample='start + (pixel_x-70)/959*(end-start)',
                  axes=['x','y','z'],channels=32,display_clip=[-6,6],normalization='moving mean201 + per-channel median/MAD',
                  detection_source='lossless float tensor, not resized RGB PNG')
    path.with_suffix('.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    return '/runs/'+record+'/'+path.name

def analyze(record,mode='fast',region=None,trace=None):
    if mode in ('tfo','pyramid_tfo'):
        from paper_methods import literature_detect
        return literature_detect(record,mode,region)
    if mode=='agentic':
        from agentic_dsp import agentic_detect
        return agentic_detect(record,region=region)
    if mode=='adaptive':
        from adaptive_signal import adaptive_detect
        return adaptive_detect(record,region=region,trace=trace)
    from workflow import DetectionWorkflow
    return DetectionWorkflow().run(record,mode,region,trace)


def interval_iou(a,b):
    inter=max(0,min(a[1],b[1])-max(a[0],b[0])+1)
    return inter/(a[1]-a[0]+1+b[1]-b[0]+1-inter)

def match_intervals(pred,truth,threshold):
    # Maximum cardinality bipartite matching; each prediction/GT used at most once.
    edges=[[j for j,g in enumerate(truth) if interval_iou(p,g)>=threshold] for p in pred]
    owners={}
    def visit(i,seen):
        for j in sorted(edges[i],key=lambda j:interval_iou(pred[i],truth[j]),reverse=True):
            if j in seen: continue
            seen.add(j)
            if j not in owners or visit(owners[j],seen): owners[j]=i; return True
        return False
    for i in range(len(pred)): visit(i,set())
    return [(i,j) for j,i in owners.items()]

def evaluate(record,mode='fast',threshold=.5):
    # This is the ONLY function reading annotations. Inference never reads GT.
    result=analyze(record,mode)
    f=pd.read_csv(DATA/manifest(record)['annotation_path'])
    f=f[f['label'].astype(str).str.lower()=='anomaly']
    truth=[(int(r.start_row_idx),int(r.end_row_idx)) for r in f.itertuples()]
    n=len(load(record)[0])
    if any(a<0 or b<a or b>=n for a,b in truth): raise ValueError('标注边界超出记录。')
    out={}
    for stage,keys in [('coarse',('coarse_start','coarse_end')),('refined',('refined_start','refined_end'))]:
        pred=[(c[keys[0]],c[keys[1]]) for c in result['candidates']]
        pairs=match_intervals(pred,truth,threshold); tp=len(pairs); fp=len(pred)-tp; fn=len(truth)-tp
        p=tp/max(tp+fp,1); r=tp/max(tp+fn,1)
        out[stage]=dict(tp=tp,fp=fp,fn=fn,precision=p,recall=r,f1=2*p*r/max(p+r,1e-12),
            center_errors_samples=[abs((sum(pred[i])-sum(truth[j]))/2) for i,j in pairs],
            boundary_errors_samples=[(abs(pred[i][0]-truth[j][0])+abs(pred[i][1]-truth[j][1]))/2 for i,j in pairs])
    return dict(record=record,source_type=manifest(record)['source_type'],mode=mode,iou_threshold=threshold,
                annotation_convention='0-based, inclusive row indices (assumed; verify with author)',**out)

TOOL_DESCRIPTIONS={
    'inspect_cross_section':'复用当前检测候选，提取轴向X分量环周稳健峰峰值，取最强通道及相邻通道插值，展示32通道截面相对方位。candidate_id可省略显示全部；假设直径32mm、提离2mm、逆时针编号，可通过有符号中心截面偶极子检查输出条件模型深度；实物深度、真实坐标或断丝数不输出。不重跑检测。',
    'tfo_detect':'P2024TFO工程工具：9×9双极性扩展Prewitt→gamma2→峰值三次样条包络→全记录提议门限→8邻域平台定位。输出是目标特征响应，不承诺论文性能。',
    'pyramid_tfo_detect':'P2025PYRAMID工程工具：三层Gaussian金字塔→双极性模板/gamma/峰值包络→融合→连通域定位。有同步实测位置/时间才计算SSR，否则明确等权参考，禁止猜测速度。',
    'inspect_visual_condition':'调用独立Qwen视觉接口描述指定segment_id的三轴图形态；未配置时明确disabled。只感知，不检测。',
    'agentic_detect':'v3多模态感知→受限DSP模板组合→参数guard→Verifier→最多一次重规划→安全回退→统一检测。',
    'inspect_conditions':'读取三轴图矩阵，按分段量化环周低阶背景、股纹斜向频谱脊、局部响应与背景共现；返回每段建议算法。识别的是信号噪声形态，实际速度/提离未知。自适应检测前必须调用。',
    'adaptive_detect':'先inspect_conditions，再按工况分段运行自适应滤波与检测。strategy=auto采用工具建议，也可选baseline/paper2021/paper2023，或routes逐段指定已知segment_id的算法。会记录响应保留检查与回退；返回处理前后图、候选和实际算法。论文算法为工程复现，未证明独立测试性能。',
    'lookup_knowledge':'检索五篇钢丝绳漏磁论文与一份模型方法说明的专家知识卡，返回出处、适用条件与实现状态。用于原理、抖动/股纹解释和工具选择。不是数据检测工具。query为中文问题，limit为1–6。',
    'audit':'检查记录、时间戳、三轴通道和物理位置映射。',
    'scan':'执行已注册的检测工作流：检查→三轴图矩阵→时间工具组合→候选融合→局部范围复核→坐标映射→证据报告。fast=通道响应主流程；balanced=加时间能量补充扫描。region为采样点闭区间；返回workflow执行日志。',
    'explain':'解释指定候选，提供原始采样点、三轴证据与局部图。candidate_id 如 D01。',
    'compare':'比较单工具和双工具组合的候选差异。',
    'evaluate':'用户明确要求评估时，读取标注计算 IoU、P/R/F1。',
}

class Agent:
    def __init__(self): self.memory={}; self.trace=[]; self.condition_memory={}
    def call(self,tool,record,**args):
        if tool not in TOOL_DESCRIPTIONS: raise ValueError('未知工具')
        started=time.time()
        notify('tool_start',tool=tool,record=record)
        if tool=='inspect_cross_section':
            from cross_section import attach_cross_sections
            previous=self.memory.get(record)
            if previous is None:raise ValueError('请先检测当前记录，再查看候选截面方位。')
            attach_cross_sections(previous,record,getattr(self,'localization_axis',None))
            result=dict(record=record,cross_sections=previous['cross_sections'])
            if args.get('candidate_id'):
                cid=args['candidate_id'].upper()
                selected=[c for c in result['cross_sections']['candidates'] if c['candidate_id']==cid]
                if not selected:raise ValueError('当前检测结果没有该候选')
                result['cross_sections']=dict(result['cross_sections'],candidates=selected)
        elif tool=='inspect_visual_condition':
            from agentic_dsp import inspect_visual_condition
            result=inspect_visual_condition(record,args.get('segment_id','S01'))
        elif tool in ('tfo_detect','pyramid_tfo_detect'):
            from paper_methods import literature_detect
            result=literature_detect(record,'tfo' if tool=='tfo_detect' else 'pyramid_tfo',region=args.get('region'))
            self.memory[record]=result
        elif tool=='agentic_detect':
            from agentic_dsp import agentic_detect
            result=agentic_detect(record,region=args.get('region'),planner=getattr(self,'transport',None),
                planner_mode=getattr(self,'dsp_planner_mode','separated'),detector_mode=getattr(self,'agentic_detector_mode','guarded'))
            self.memory[record]=result
        elif tool=='inspect_conditions':
            from adaptive_signal import inspect_conditions
            result=inspect_conditions(record);self.condition_memory[record]=result
        elif tool=='adaptive_detect':
            if record not in self.condition_memory:raise ValueError('请先调用inspect_conditions，再根据测量工况选择滤波路线。')
            from adaptive_signal import adaptive_detect
            result=adaptive_detect(record,args.get('strategy','auto'),args.get('routes'),args.get('region'),self.trace)
            self.memory[record]=result
        elif tool=='lookup_knowledge':
            result=lookup_knowledge(args.get('query',''),args.get('limit',4))
        elif tool=='audit': result=audit(record)
        elif tool=='scan':
            result=analyze(record,args.get('mode','fast'),args.get('region'),self.trace)
            self.memory[record]=result
        elif tool=='explain':
            previous=self.memory.get(record)
            if previous is None:raise ValueError('请先完成检测，再解释其中的候选；解释不会自动重跑检测。')
            cid=args.get('candidate_id','D01').upper()
            c=next((c for c in previous['candidates'] if c['id']==cid),None)
            if c is None: raise ValueError(f'{cid} 不在当前检测结果中，请先查看候选列表。')
            n=len(load(record)[0]); lo=max(0,c['coarse_start']-160); hi=min(n-1,c['coarse_end']+160)
            result=dict(candidate=c,prediction_run_id=previous.get('run_id'),image=render_image(record,lo,hi,[c],(previous.get('run_id') or 'fixed')+'_'+cid),
                        explanation=f"{cid} 在 {c['start_s']}–{c['end_s']} 秒有局部响应。支持工具：{', '.join(c['sources'])}。三轴 RMS 为 {c['axis_response_rms']}（{c.get('axis_response_kind','原始磁信号幅值')}）。这说明存在待核查信号，不能据此确定断丝数量或真实断口长度。")
            if previous.get('mode') in ('adaptive','agentic','tfo','pyramid_tfo'):
                result['processed_image']=previous['image']
                result['filter_segments']=[s for s in previous.get('filter_segments',[]) if s['sample_range'][0]<=c['refined_end'] and s['sample_range'][1]>=c['refined_start']]
                result['agentic_segments']=[s for s in previous.get('agentic_segments',[]) if s['sample_range'][0]<=c['refined_end'] and s['sample_range'][1]>=c['refined_start']]
                result['literature_segments']=[s for s in previous.get('literature_segments',[]) if s['sample_range'][0]<=c['refined_end'] and s['sample_range'][1]>=c['refined_start']]
                result['explanation']+='已返回对应滤波分段与处理前后证据图；这些是工程算法输出，不代表抖动已被完全排除。'
            if c.get('confidence'):
                confidence=c['confidence']
                result['explanation']+=f"证据可信程度：{confidence['level']}；工程证据指数{confidence['evidence_index']}（不是缺陷概率）。"+'；'.join(confidence['uncertainties'])+'。'+confidence['review_action']
        elif tool=='compare':
            fast=analyze(record,'fast',trace=self.trace); balanced=analyze(record,'balanced',trace=self.trace)
            result=dict(fast=fast,balanced=balanced,description='双工具模式执行补充扫描；增加的候选需要核查，不能直接当成已找回漏检。')
            self.memory[record]=balanced
        else:
            previous=self.memory.get(record)
            if previous is None:raise ValueError('请先完成检测，再评估当前保存的结果；评估不会自动重跑检测。')
            if previous.get('scope_samples') is not None:raise ValueError('当前是局部结果，请先完成整条检测再做全记录标注评估。')
            from evaluate_agent import evaluate_record
            measured=evaluate_record(record,previous)
            result=dict(record=record,mode=previous['mode'],prediction_run_id=previous.get('run_id'),
                prediction_reused=True,refined=measured['events']['0.5'],evaluation=measured,
                annotation_convention='0-based inclusive row indices; confirm with dataset author')
        if tool in ('scan','adaptive_detect','agentic_detect','tfo_detect','pyramid_tfo_detect','compare'):
            from reporting import attach_reports
            attach_reports(result,record,getattr(self,'localization_axis',None))
        self.trace.append(dict(tool=tool,record=record,arguments=args,elapsed_s=round(time.time()-started,3)))
        notify('tool_complete',tool=tool,tool_elapsed_s=round(time.time()-started,3))
        return result
    def ask(self,record,question,mode='fast'):
        self.trace=[]; q=question.strip(); cid=re.search(r'(?<![A-Za-z0-9])D\d{1,3}(?![A-Za-z0-9])',q,re.I)
        candidate_id=cid.group() if cid else None
        if candidate_id is None and any(w in q for w in ['解释','候选','刚才']):
            ordinal=re.search(r'第\s*(\d+|[一二三四五六七八九十])\s*(?:个|处)',q)
            previous=self.memory.get(record)
            if ordinal:
                value=ordinal.group(1);number=int(value) if value.isdigit() else '一二三四五六七八九十'.index(value)+1
                if previous is None or not 1<=number<=len(previous['candidates']):raise ValueError('对应候选不存在，请先检测并选择当前候选编号。')
                candidate_id=previous['candidates'][number-1]['id']
            elif '解释' in q and '刚才' in q:
                if previous is None or len(previous['candidates'])!=1:raise ValueError('请指定当前候选编号，例如解释D01。')
                candidate_id=previous['candidates'][0]['id']
        region=None
        region_match=re.search(r'(\d+(?:\.\d+)?)\s*(?:到|至|[-~—–])\s*(\d+(?:\.\d+)?)\s*(?:秒|s\b)',q,re.I)
        if region_match:
            a,b=map(float,region_match.groups())
            if b<=a: raise ValueError('时间区间结束应大于开始。')
            n=len(load(record)[0]); region=[int(a*500),min(n-1,int(b*500))]
            if region[0]>=n: raise ValueError('所选区间超出记录。')
        if knowledge_question(q):
            result=self.call('lookup_knowledge',record,query=q)
            answer='\n\n'.join(c['title']+'：'+c['body']+'\n限制：'+c['limitations']+
                    '\n出处：'+'；'.join(reference_label(r) for r in c['references'])
                    for c in result['knowledge_items']) or result['note']
        elif any(w in q for w in ['评估','准确率','F1','召回','标注']):
            result=self.call('evaluate',record,mode=mode)
            s=result['refined']
            from evaluate_agent import pct
            answer=f"复用当前检测结果，按 IoU≥0.5 一对一匹配：TP={s['tp']}，FP={s['fp']}，FN={s['fn']}；精确率 {pct(s['precision'])}、召回率 {pct(s['recall'])}、F1 {pct(s['f1'])}。这是本条开发记录的结果，不能代表独立测试集性能。"
        elif any(w in q for w in ['比较','对比','多种','漏检']):
            result=self.call('compare',record)
            answer=f"单工具得到 {len(result['fast']['candidates'])} 个候选，双工具得到 {len(result['balanced']['candidates'])} 个候选。双工具包含额外的时间能量扫描；具体结果见下方。"
        elif any(w in q for w in ['截面','环周位置','方位展示']):
            result=self.call('inspect_cross_section',record,candidate_id=candidate_id)
            answer='已复用当前候选生成截面图并检查单点偶极子深度模型；通过检查时展示假设几何下的条件估计，拒绝时保留原因。真实深度与毫米坐标未经标定。'
        elif candidate_id:
            result=self.call('explain',record,candidate_id=candidate_id)
            answer=result['explanation']
        elif any(w in q for w in ['数据格式','采样率','检查数据','多少通道']):
            result=self.call('audit',record); answer=f"{result['samples']} 个采样点，三轴各 32 通道，名义采样率 500 Hz。"+' '.join(result['warnings'])
        elif any(w in q for w in ['识别工况','分析工况','噪声分析']) and '检测' not in q:
            result=self.call('inspect_conditions',record)
            answer='已按分段识别信号噪声形态，建议路线：'+', '.join(s['segment_id']+'='+s['recommended_algorithm'] for s in result['condition_segments'])+'。实际速度和提离未知，以上不是机械工况的确诊。'
        elif any(w in q for w in ['米','长度位置','钢绳位置']) and load(record)[3] is None:
            result=self.call('audit',record)
            answer='这条记录缺少采样点到钢绳长度的实测映射。目前可以检测并输出时间/采样点区间；米制定位需要补充编码器位置或经验证的位移映射。'
        else:
            if mode in ('tfo','pyramid_tfo') or any(w in q.lower() for w in ['tfo','目标特征','金字塔']):
                method='pyramid_tfo_detect' if mode=='pyramid_tfo' or '金字塔' in q else 'tfo_detect'
                result=self.call(method,record,region=region)
                answer=f"已执行文献工程工具 {method}，得到{len(result['candidates'])}个待复核响应。SSR未知时不会推断速度；这不是论文原实验性能复现。"
            elif mode=='agentic':
                result=self.call('agentic_detect',record,region=region)
                m=result['metrics']
                answer=f"检测得到{len(result['candidates'])}个待复核候选；独立视觉成功分段={m['vlm_completed']}，直接图像规划分段={m['direct_image_segments']}，重规划分段={m['replan_count']}，回退率={m['fallback_rate']:.1%}。状态：{result['processing_status']}。验证通过不代表缺陷已确诊，请查看实际执行方案与证据。"
            elif mode=='adaptive' or any(w in q for w in ['自适应','滤波','工况']):
                self.call('inspect_conditions',record)
                result=self.call('adaptive_detect',record,region=region)
                answer=f"已根据分段噪声特征选择滤波，得到{len(result['candidates'])}个待复核候选。实际路线："+', '.join(s['segment_id']+'='+s['used'] for s in result['filter_segments'])+'。完整判断依据、参数和回退见下方。'
            else:
                result=self.call('scan',record,mode=mode,region=region)
                answer=f"完成{'指定时间范围' if region else '整条记录'}检测，发现 {len(result['candidates'])} 个待复核候选。已调用：{', '.join(result['tools'])} → 合并 → 局部边界细化。结果中的范围表示漏磁响应范围。"
        from reporting import readable
        return dict(answer=readable(answer),result=result,trace=self.trace,planner='本地规则调度',
                    supported_queries='整条检测、5–10秒检测、解释D01、比较两种检测、评估、检查数据')
