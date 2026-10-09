"""Condition-aware, paper-inspired algorithm reproduction on calibrated float images.

Array convention: axes x axial samples x circumferential channels.
No annotation access. Paper coefficients not reported are explicit engineering choices.
"""
from pathlib import Path
import sys,json,time
import numpy as np
vendor=Path(__file__).parent/'vendor'
if vendor.exists() and sys.version_info[:2]==(3,12) and sys.platform=='win32':sys.path.insert(0,str(vendor))
try:
    from scipy import ndimage,signal
    from scipy.interpolate import CubicSpline
except ImportError as e:
    raise ImportError('自适应滤波需要SciPy，请安装requirements.txt依赖。') from e

ALGORITHMS=('baseline','paper2021','paper2023')
VERSION='2.0-experimental'
POLICY=dict(segment_samples=2048,context_samples=256,strong_shake=.45,shake=.20,
            broad_coverage=.65,moderate_coverage=.45,strand=.12,overlap=.35,retention_floor=.12,threshold_mad=5.,minimum_samples=12,merge_gap=15)

def spans(n,step=2048):
    items=[[a,min(n-1,a+step-1)] for a in range(0,n,step)]
    if len(items)>1 and items[-1][1]-items[-1][0]+1<256:
        last=items.pop();items[-1][1]=last[1]
    return items

def mean(x,w,axis=0):
    return ndimage.uniform_filter1d(x,max(1,int(w)),axis=axis,mode='reflect')

def zero_phase_highpass(x,w,axis=0):
    # Causal Hsub(z)=I-MA followed by reverse filtering, paper Eq.(5).
    values=np.moveaxis(x,axis,0);w=max(2,min(int(w),len(values)-1))
    b=np.full(w,-1./w);b[0]+=1
    zi=np.cumsum(b[:0:-1])[::-1].reshape((w-1,)+(1,)*(values.ndim-1))
    pad=min(3*(w-1),len(values)-1)
    padded=np.pad(values,[(pad,pad)]+[(0,0)]*(values.ndim-1),mode='reflect')
    first,_=signal.lfilter(b,[1.],padded,axis=0,zi=zi*padded[0])
    reverse=first[::-1]
    second,_=signal.lfilter(b,[1.],reverse,axis=0,zi=zi*reverse[0])
    return np.moveaxis(second[::-1][pad:-pad],0,axis)

def calibrate(raw):
    centered=raw-np.median(raw[:min(64,len(raw))],axis=0,keepdims=True)
    residual=centered-mean(centered,201)
    scale=1.4826*np.median(np.abs(residual-np.median(residual,axis=0)),axis=0)
    # Common scale per magnetic axis preserves relative circumferential amplitudes.
    scales=np.maximum(np.median(scale,axis=1),1.)
    return np.transpose(centered/scales[None,:,None],(1,0,2)),scales

def condition_features(z):
    z=np.asarray(z,float)
    n=z.shape[1]; c=z.shape[2]
    residual=z-mean(z,min(201,n),axis=1)
    amplitude=np.max(np.abs(residual),axis=0)
    magnitude=np.sqrt(np.mean(amplitude**2,axis=1))
    busy=magnitude>np.quantile(magnitude,.8)
    coverage_per_time=np.mean(amplitude>.35*np.maximum(np.max(amplitude,axis=1,keepdims=True),1e-9),axis=1)
    coverage=float(np.median(coverage_per_time[busy])) if np.any(busy) else 0.
    spatial=np.fft.rfft(residual,axis=2)
    power=np.mean(np.abs(spatial)**2,axis=(0,1))
    shake=float(power[1]/max(np.sum(power[1:]),1e-12))
    low=spatial.copy();low[:,:,2:]=0;low[:,:,0]=0
    shake_time=np.sqrt(np.mean(np.fft.irfft(low,n=c,axis=2)**2,axis=(0,2)))
    local=spatial.copy();local[:,:,:2]=0
    local_time=np.sqrt(np.mean(np.fft.irfft(local,n=c,axis=2)**2,axis=(0,2)))
    active=local_time>np.quantile(local_time,.85)
    overlap=float(np.mean(shake_time[active]>np.quantile(shake_time,.85))) if np.any(active) else 0.
    # Estimate a diagonal stripe ridge. Temporal frequency and channel harmonic
    # must BOTH be nonzero; broadband defect spots do not imply a strand ridge.
    f2=np.mean(np.abs(np.fft.fft2(residual,axes=(1,2)))**2,axis=0)
    ft=np.fft.fftfreq(n); allowed=(np.abs(ft)>=2/max(n,1))&(np.abs(ft)<=.2)
    ks=list(range(2,min(c//2,11)))
    ridge=f2[:,ks].copy();ridge[~allowed]=0
    ii,jj=np.unravel_index(np.argmax(ridge),ridge.shape)
    k=ks[jj]; f=float(ft[ii]);total=float(np.sum(f2[:,ks]))
    band=float(np.sum(f2[max(0,ii-1):min(n,ii+2),k]))
    strand=min(1.,band/max(total,1e-12))
    period=1/abs(f) if f else 0.
    slope=-k/(f*c) if f else 0. # native axial samples per circumferential channel
    broad=coverage>=POLICY['broad_coverage']
    moderate=coverage>=POLICY['moderate_coverage']
    if broad and (shake>=POLICY['strong_shake'] or (shake>=POLICY['shake'] and overlap>=POLICY['overlap'])):
        route='paper2023';label='跨通道背景较强或与局部响应共现'
    elif strand>=POLICY['strand'] or (moderate and shake>=POLICY['shake']):
        route='paper2021';label='周期斜纹或中等跨通道背景'
    else:route='baseline';label='上述噪声线索不足，保留基线'
    return dict(shaking_energy_fraction=round(shake,4),circumferential_coverage=round(coverage,4),strand_ridge_fraction=round(strand,4),
                local_background_cooccurrence=round(overlap,4),strand_period_samples=round(period,2),
                strand_harmonic=k,stripe_slope_samples_per_channel=round(slope,4),
                recommended_algorithm=route,signal_condition=label,
                interpretation='低阶能量/覆盖率是抖动线索，宽缺陷也可能产生；共现不等于已确认缺陷重叠，指标不是概率')

def inspect_conditions(record):
    import core
    raw,*_=core.load(record); z,scales=calibrate(raw)
    segments=[]
    for i,(a,b) in enumerate(spans(len(raw)),1):
        f=condition_features(z[:,a:b+1,:]);f.update(segment_id=f'S{i:02d}',sample_range=[a,b],time_range_s=[a/500,b/500])
        segments.append(f)
    return dict(record=record,condition_segments=segments,axis_scaling=scales.tolist(),policy=POLICY,
                available_algorithms=list(ALGORITHMS),version=VERSION,
                physical_condition='未知实际速度、提离和振幅；不由信号直接推断机械工况',
                geometry_assumption='通道1–32按环周顺序等间距；需与硬件确认，三轴方向不自动赋义',
                annotations_used=False)

def circum_spline(y,channels):
    c=y.shape[1]
    closed=np.concatenate([y,y[:,:1]],axis=1)
    return CubicSpline(np.arange(c+1),closed,axis=1,bc_type='periodic')(np.arange(channels)*c/channels)

def kalman_baseline(y,window):
    # Scalar random-walk Kalman per channel. Q/R not specified by P2023.
    gain=2/(window+1);q=gain*gain/max(1-gain,1e-6);p=np.ones(y.shape[1]);state=np.median(y[:min(32,len(y))],axis=0)
    out=np.empty_like(y)
    for t in range(len(y)):
        p+=q;k=p/(p+1);state+=k*(y[t]-state);p*=1-k;out[t]=state
    return y-out,dict(Q_over_R=q,initial_P=1.,initial_state='前32点逐通道中位数；工程初始化')

def diagonal_filter(y,slope,window=10):
    # Shear along the measured stripe; filter resampled traces, then invert shear.
    t,c=np.indices(y.shape,dtype=float)
    aligned=ndimage.map_coordinates(y,[t+slope*c,c],order=1,mode='nearest')
    filtered=zero_phase_highpass(aligned,window,axis=1)
    return ndimage.map_coordinates(filtered,[t-slope*c,c],order=1,mode='nearest')

def morph(y,size,operation):
    # Wrap ring coordinates explicitly; reflect axial borders.
    pad=size[1]+2
    padded=np.pad(y,[(0,0),(pad,pad)],mode='wrap')
    fn=ndimage.grey_closing if operation=='close' else ndimage.grey_opening
    return fn(padded,size=size,mode='reflect')[:,pad:-pad]

def paper_filter(y,algorithm,features,reference=False):
    """Native-coordinate output + explicit provenance; reference constants optional."""
    period=features.get('strand_period_samples',0)
    window=20 if reference else int(np.clip(3*period if period else 201,81,501))
    parameters=dict(baseline_window=window,parameter_mode='paper_reference' if reference else 'native_adaptation')
    if algorithm=='baseline':
        return np.abs(signal.hilbert(y-mean(y,201),axis=0)),parameters,['移动均值去趋势','Hilbert包络']
    if algorithm=='paper2021':
        out=zero_phase_highpass(y,window)
        out=circum_spline(out,100)
        operations=['双向等效零相位去趋势','周期样条环周插值→100']
        if features['strand_ridge_fraction']>=POLICY['strand']:
            slope=features['stripe_slope_samples_per_channel']*y.shape[1]/100
            out=diagonal_filter(out,slope,10);parameters.update(diagonal_window=10,stripe_slope=slope)
            operations.append('测量斜纹方向→剪切重采样→零相位滤波→反剪切')
        else:operations.append('斜纹证据不足，跳过方向滤波')
        out=np.abs(signal.hilbert(out,axis=0))
        width=50 if reference else 51 # Fig.14 span50; odd51 is an explicit native adaptation.
        bg=ndimage.median_filter(out,size=(1,width),mode='wrap')
        out=np.maximum(out-bg,0)
        out=circum_spline(out,y.shape[1])
        parameters.update(circumferential_samples=100,median_span=width)
        return np.maximum(out,0),parameters,operations+['轴向Hilbert包络','环周中值背景相减','恢复原始通道尺度']
    if algorithm!='paper2023':raise ValueError('未知滤波算法')
    out,kparams=kalman_baseline(y,window);parameters.update(kparams)
    out=circum_spline(out,360)
    ti=np.arange(len(y));tnew=np.arange((len(y)-1)*5+1)/5
    out=CubicSpline(ti,out,axis=0)(tnew)
    pos=np.maximum(out,0);neg=np.maximum(-out,0)
    parts=[]
    for part in (pos,neg):
        reconstructed=morph(part,(1,120),'close')
        background=morph(reconstructed,(1,120),'open')
        # Fig.12 and standard white top-hat: original minus opened background.
        residue=np.maximum(reconstructed-background,0)
        parts.append(morph(residue,(20,20),'open'))
    combined=parts[0]-parts[1]
    envelope=np.abs(signal.hilbert(combined,axis=0))[::5]
    envelope=circum_spline(envelope,y.shape[1])
    parameters.update(circumferential_samples=360,axial_factor=5,line_SE=120,square_SE=20,
                      top_hat_sign='closed_image - opened_background (Fig.12)')
    return np.maximum(envelope,0),parameters,['Kalman去趋势','环周360/轴向5倍样条插值','正负分组',
                  '竖线闭运算重构股纹','竖线开运算估计抖动','白顶帽相减','方形开运算',
                  '合并正负→轴向Hilbert包络','恢复原始采样/通道尺度']

def localized_score(z):
    ft=np.fft.rfft(z,axis=2);ft[:,:,:2]=0
    return np.max(np.abs(np.fft.irfft(ft,n=z.shape[2],axis=2)),axis=(0,2))

def preservation_guard(before,after):
    # Protect only distinct local landmarks, not all samples above an amplitude cut.
    score=localized_score(before);med=np.median(score);mad=1.4826*np.median(np.abs(score-med))
    peaks,_=signal.find_peaks(score,prominence=max(4*mad,.08*np.max(score),1e-8),distance=40)
    ratios=[]
    base=np.max(np.abs(signal.hilbert(before,axis=1)),axis=(0,2));post=np.max(after,axis=(0,2))
    for p in peaks:
        a=max(0,p-20);b=min(len(score),p+21)
        ratios.append(float(np.max(post[a:b])/max(np.max(base[a:b]),1e-9)))
    return dict(landmark_count=len(peaks),minimum_retention=min(ratios) if ratios else None,
                passes=not ratios or min(ratios)>=POLICY['retention_floor'],
                interpretation='无标注的局部响应保留检查；地标可能含噪声，不是缺陷召回率')

def apply_adaptive(raw,conditions,strategy='auto',routes=None):
    if strategy not in ('auto',)+ALGORITHMS:raise ValueError('未知自适应策略')
    if routes is not None:
        if not isinstance(routes,list) or len(routes)>len(conditions['condition_segments']):raise ValueError('路线列表无效')
        ids={s['segment_id'] for s in conditions['condition_segments']};chosen={}
        for r in routes:
            if not isinstance(r,dict) or set(r)!={'segment_id','algorithm'} or r['segment_id'] not in ids or r['algorithm'] not in ALGORITHMS or r['segment_id'] in chosen:
                raise ValueError('路线仅允许已有分段与已注册算法，且不能重复')
            chosen[r['segment_id']]=r['algorithm']
    else:chosen={}
    z,scales=calibrate(raw);out=np.zeros_like(z);weights=np.zeros(len(raw));logs=[]
    for item in conditions['condition_segments']:
        a,b=item['sample_range'];lo=max(0,a-POLICY['context_samples']);hi=min(len(raw),b+1+POLICY['context_samples'])
        selected=chosen.get(item['segment_id'],item['recommended_algorithm'] if strategy=='auto' else strategy)
        block=z[:,lo:hi,:];baseline=block-mean(block,201,axis=1)
        began=time.time();filtered=[];parameters=[];operations=[]
        for axis in block:
            y,p,ops=paper_filter(axis,selected,item)
            filtered.append(y);parameters.append(p);operations=ops
        filtered=np.array(filtered);guard=preservation_guard(baseline[:,a-lo:b-lo+1,:],filtered[:,a-lo:b-lo+1,:])
        used=selected
        if not guard['passes'] and selected!='baseline':
            filtered=np.abs(signal.hilbert(baseline,axis=1));used='baseline'
        weight=np.ones(hi-lo)
        if a>lo:weight[:a-lo]=np.linspace(0,1,a-lo,endpoint=False)
        if hi>b+1:weight[b+1-lo:]=np.linspace(1,0,hi-b-1,endpoint=False)
        out[:,lo:hi,:]+=filtered*weight[None,:,None];weights[lo:hi]+=weight
        logs.append(dict(segment_id=item['segment_id'],sample_range=[a,b],recommended=item['recommended_algorithm'],
            selected=selected,used=used,selection_source='model_routes' if item['segment_id'] in chosen else 'condition_policy' if strategy=='auto' else 'explicit_strategy',
            reason=item['signal_condition'],parameters=parameters[0],operations=operations,
            preservation_guard=guard,fallback_reason='局部响应保留不足，回退基线' if used!=selected else None,
            elapsed_s=round(time.time()-began,3)))
    out/=np.maximum(weights[None,:,None],1e-9)
    return out,logs,z,scales

def detect_envelope(envelope):
    import core
    score=np.sqrt(mean(np.max(envelope,axis=(0,2))**2,21))
    med=float(np.median(score));mad=float(1.4826*np.median(np.abs(score-med)))
    threshold=max(med+POLICY['threshold_mad']*mad,.05*float(np.max(score)),1e-6)
    intervals=core.masks_to_intervals(score>threshold,POLICY['minimum_samples'],POLICY['merge_gap'])
    return intervals,score,threshold

def adaptive_detect(record,strategy='auto',routes=None,region=None,trace=None):
    import core
    audit=core.audit(record);raw,*_=core.load(record)
    if region is not None and (len(region)!=2 or any(type(v)is not int for v in region) or region[0]<0 or region[1]>=len(raw) or region[1]<region[0]):raise ValueError('采样点范围无效')
    conditions=inspect_conditions(record)
    envelope,segments,z,scales=apply_adaptive(raw,conditions,strategy,routes)
    intervals,score,threshold=detect_envelope(envelope);candidates=[]
    for a,b in intervals:
        segment_sources=sorted({s['used'] for s in segments if s['sample_range'][0]<=b and s['sample_range'][1]>=a})
        cid=f'D{len(candidates)+1:02d}'
        c=dict(id=cid,start=a,end=b,coarse_start=a,coarse_end=b,refined_start=a,refined_end=b,
            refinement_used=False,refinement_reason='自适应滤波后阈值响应边界；未经实际断口校准',
            start_s=round(a/500,4),end_s=round(b/500,4),center_sample=(a+b)//2,
            sources=['adaptive_'+s for s in segment_sources],axis_response_rms=np.sqrt(np.mean(envelope[:,a:b+1,:]**2,axis=(1,2))).round(3).tolist(),
            active_channels_1based=(np.flatnonzero(np.max(envelope[:,a:b+1,:],axis=(0,1))>threshold)+1).tolist(),
            status='待复核',extent_kind='处理后漏磁响应范围，非真实损伤长度')
        pos=core.load(record)[3]
        if pos is not None:c['position_m']=[float(np.min(pos[a:b+1])),float(np.max(pos[a:b+1]))]
        candidates.append(c)
    if region is not None:candidates=[c for c in candidates if c['refined_end']>=region[0] and c['refined_start']<=region[1]]
    folder=core.RUNS/record;folder.mkdir(exist_ok=True)
    artifact='adaptive_'+strategy
    np.savez_compressed(folder/(artifact+'_signal.npz'),envelope=envelope.astype(np.float32),score=score,axis_scale=scales)
    image=render_comparison(record,z,envelope,candidates,folder/(artifact+'.png'))
    nodes=[dict(id='condition_analysis',title='信号工况与噪声识别',status='completed',output=conditions),
           dict(id='adaptive_filter',title='分段选择滤波与响应保留检查',status='completed',output=segments),
           dict(id='adaptive_detection',title='处理后包络阈值检测与坐标映射',status='completed',output=dict(threshold=threshold))]
    if trace is not None:
        for node in nodes:trace.append(dict(tool=node['id'],record=record,arguments={'strategy':strategy},elapsed_s=round(sum(s['elapsed_s'] for s in segments),3) if node['id']=='adaptive_filter' else 0.))
    result=dict(record=record,mode='adaptive',strategy=strategy,audit=audit,candidates=candidates,image=image,
        conditions=conditions,filter_segments=segments,thresholds={'adaptive_envelope':threshold},params=POLICY,
        tools=sorted({'adaptive_'+s['used'] for s in segments}),scope_samples=region,
        signal_artifact='/runs/'+record+'/'+artifact+'_signal.npz',stage='算法级工程复现与三轴迁移，尚无独立测试性能结论',
        stitching='上下文重叠线性融合，避免硬切换；接缝附近可能混合邻段算法输出',
        score_kind='响应强度，不是概率',workflow=dict(version=VERSION,status='completed',nodes=nodes,decisions=dict(annotations_used=False)),
        limitations=['阵列等间距环周顺序尚待硬件确认','实际速度/提离未知','Kalman参数等未公开，已明示工程选择','原论文数据不可用，不能声称精确复现实验指标'])
    (folder/(artifact+'_result.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    return result

def render_comparison(record,before,after,candidates,path):
    from PIL import Image,ImageDraw,ImageFont
    img=Image.new('RGB',(1200,710),'#102030');d=ImageDraw.Draw(img)
    try:font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17)
    except OSError:font=ImageFont.load_default()
    d.text((50,12),record+' | 左：输入响应（去显示基线）  右：自适应处理后包络',font=font,fill='white')
    residual=before-mean(before,201,axis=1)
    # Same numeric scale across panels; zero background is dark, not amplified independently.
    level=max(float(np.quantile(np.abs(residual),.995)),1e-6)
    for ai in range(3):
        y=55+ai*200
        for side,matrix in enumerate((residual,after)):
            values=np.clip(matrix[ai].T/level,-1,1)
            rgb=np.stack([25+225*np.maximum(values,0),30+100*np.minimum(np.abs(values),1),40+210*np.maximum(-values,0)],axis=2).astype('uint8')
            panel=Image.fromarray(rgb).resize((530,160));x=55+side*585;img.paste(panel,(x,y))
            for c in candidates:
                a=x+int(c['refined_start']/(before.shape[1]-1)*529);b=x+int(c['refined_end']/(before.shape[1]-1)*529)
                d.rectangle((a,y,b,y+159),outline='#ffe36d',width=1)
            d.text((x,y+165),f'{"XYZ"[ai]} · 通道1–32 · 0–{(before.shape[1]-1)/500:.3f}名义秒',font=font,fill='white')
    d.text((55,675),'同一显示尺度；检测用完整浮点矩阵。黄色框为处理后响应候选，需实物复核。',font=font,fill='white')
    img.save(path)
    return '/runs/'+record+'/'+path.name
