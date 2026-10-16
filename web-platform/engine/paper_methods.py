"""Source-traceable TFO / pyramid TFO engineering adaptations.

No annotations or cloud APIs are accessible here. Three-axis/32-channel use,
context margins, peak-envelope safeguards and plateau tie-breaking are explicit
adaptations, not claims of reproducing the authors' data or reported accuracy.
"""
import json,time,uuid
from pathlib import Path
import numpy as np
from adaptive_signal import mean,circum_spline
from scipy import ndimage,signal
from scipy.interpolate import CubicSpline
import core
from visual_condition import render

POLICY=dict(segment_samples=200,context_samples=32,image_channels=200,baseline_window=201,
    tfo_kernel=9,pyramid_base_kernel=5,sensitivity=5,gamma=2.,proposal_factor=7.,threshold_step=.01,
    min_plateau_steps=5,min_component_pixels=20,temporal_merge_gap=15)
SOURCES={'tfo':'P2024TFO','pyramid_tfo':'P2025PYRAMID'}

def extended_prewitt(size):
    if type(size)is not int or size<3 or size>19 or size%2==0:raise ValueError('模板尺寸须为3–19的奇数')
    # Time x channels: opposite temporal lobes, zero central row.
    return np.sign(np.arange(size)-size//2)[:,None]*np.ones((1,size))

def peak_envelope(image):
    result=np.empty_like(image,dtype=float);n=len(image)
    if n<3:return np.maximum(image,0)
    for channel in range(image.shape[1]):
        column=image[:,channel];peaks=signal.find_peaks(column)[0]
        knots=np.unique(np.r_[0,peaks,n-1])
        values=CubicSpline(knots,column[knots],bc_type='natural')(np.arange(n)) if len(knots)>2 else np.interp(np.arange(n),knots,column[knots])
        # Prevent spline undershoot from deleting measured feature energy.
        result[:,channel]=np.maximum(column,np.maximum(values,0))
    return result

def enhance(image,size=9,*,boundary='constant'):
    response=np.abs(ndimage.convolve(image,extended_prewitt(size),mode=boundary,cval=0.))
    return peak_envelope(response**POLICY['gamma'])

def spatial_parameters(sampling_hz=None,speed_m_s=None):
    if sampling_hz is None or speed_m_s is None:
        return dict(status='unknown',mu=None,kernel=5,weights=[1/3]*3,effective_weights=[1/3]*3,
            fusion='equal_weights_engineering_fallback',note='无实测速度/采样率；不推断SSR，不声称复现SSR自适应')
    if not np.isfinite([sampling_hz,speed_m_s]).all() or sampling_hz<=0 or speed_m_s<0:raise ValueError('实测采样率或速度无效')
    # Paper extreme SSR = 250 / 1.5 samples/m. Values outside its range are flagged.
    ratio=(250/1.5)*speed_m_s/sampling_hz;mu=float(np.clip(ratio,0,1))
    weights=[mu**2,2*mu*(1-mu),(1-mu)**2]
    size=int(np.ceil(5+5*(1-mu)));size+=size%2==0
    effective=[weights[0],(1-weights[0])*weights[1],(1-weights[0])*(1-weights[1])]
    return dict(status='measured' if 0<=ratio<=1 else 'outside_paper_range_clipped',mu=mu,unclipped_mu=float(ratio),
        sampling_hz=float(sampling_hz),speed_m_s=float(speed_m_s),kernel=size,weights=weights,effective_weights=effective,
        fusion='Eq14_recursive_coarsest_initialization',note='按式14递推；最粗层初始化和奇数模板调整为显式工程选择，非式11–13直接加权')

def pyramid_enhance(image,context):
    layers=[image]
    for _ in range(2):layers.append(ndimage.gaussian_filter(layers[-1],1.,mode='nearest')[::2,::2])
    enhanced=[enhance(layer,context['kernel'],boundary='nearest') for layer in layers]
    restored=[]
    for layer in enhanced:
        rows=np.linspace(0,len(layer)-1,len(image));cols=np.linspace(0,layer.shape[1]-1,image.shape[1])
        restored.append(ndimage.map_coordinates(layer,np.meshgrid(rows,cols,indexing='ij'),order=1,mode='nearest'))
    weights=context['effective_weights']
    fused=sum(w*layer for w,layer in zip(weights,restored))
    return fused,[list(layer.shape) for layer in layers]

def plateau_regions(image):
    image=np.asarray(image,dtype=float)
    if image.ndim!=2 or not image.size or not np.isfinite(image).all() or np.any(image<0):raise ValueError('特征图须为非负、有限的二维矩阵')
    if np.ptp(image)==0:return [],dict(status='flat_feature',threshold=None,counts=[],plateaus=[],max_feature=float(image.max()))
    maximum=float(np.max(image));normalized=image/maximum if maximum>0 else np.zeros_like(image)
    thresholds=np.arange(0,1.001,.01);counts=[]
    for threshold in thresholds:
        counts.append(int(ndimage.label(normalized>threshold,structure=np.ones((3,3)))[1]))
    runs=[];start=0
    for end in range(1,len(counts)+1):
        if end==len(counts) or counts[end]!=counts[start]:
            if counts[start]>0 and end-start>=POLICY['min_plateau_steps']:
                runs.append((counts[start],end-start,start,end-1))
            start=end
    if not runs:return [],dict(status='no_stable_plateau',threshold=None,counts=counts,plateaus=[],max_feature=maximum)
    selected=max(runs,key=lambda run:(run[0],run[1],run[2]))
    threshold=float(thresholds[max(1,selected[2])])
    labels,count=ndimage.label(normalized>threshold,structure=np.ones((3,3)))
    sizes=np.bincount(labels.ravel());objects=ndimage.find_objects(labels);regions=[]
    for i,box in enumerate(objects,1):
        if box is None or sizes[i]<POLICY['min_component_pixels']:continue
        rows,cols=box
        regions.append(dict(start=int(rows.start),end=int(rows.stop-1),channel_pixels=[int(cols.start),int(cols.stop-1)],area=int(sizes[i])))
    return regions,dict(status='selected',threshold=threshold,counts=counts,
        plateaus=[dict(region_count=r[0],steps=r[1],range=[float(thresholds[r[2]]),float(thresholds[r[3]])]) for r in runs],
        selected_region_count=selected[0],component_count=count,min_component_pixels=POLICY['min_component_pixels'],max_feature=maximum,
        selection_rule='largest component count among >=5-step plateaus; longest/high-threshold tie-break; low endpoint')

def measured_context(position,raw_time,a,b):
    if position is None or raw_time is None:return spatial_parameters()
    time_steps=np.diff(raw_time[a:b+1]);position_steps=np.diff(position[a:b+1])
    if not len(time_steps) or not np.isfinite(time_steps).all() or np.any(time_steps<=0):return spatial_parameters()
    return spatial_parameters(1/float(np.median(time_steps)),float(np.median(np.abs(position_steps/time_steps))))

def literature_detect(record,method='tfo',region=None):
    if method not in SOURCES:raise ValueError('未知文献方法')
    raw,_,raw_time,position=core.load(record);n=len(raw)
    if region is not None and (not isinstance(region,list) or len(region)!=2 or any(type(v)is not int for v in region) or not 0<=region[0]<=region[1]<n):raise ValueError('采样点范围无效')
    began=time.time();run_id=uuid.uuid4().hex[:12];folder=core.RUNS/record;folder.mkdir(exist_ok=True)
    tensor=np.transpose(raw,(1,0,2));residual=tensor-mean(tensor,201,axis=1)
    scale=max(float(np.max(np.abs(residual))),1e-9);normalized=residual/scale
    features=np.zeros((3,n,200),dtype=np.float32);segments=[]
    for index,a in enumerate(range(0,n,200),1):
        b=min(n-1,a+199);lo=max(0,a-32);hi=min(n,b+33);context=measured_context(position,raw_time,a,b)
        shapes=None
        for axis in range(3):
            image=circum_spline(normalized[axis,lo:hi],200)
            if method=='tfo':enhanced=enhance(image,9)
            else:enhanced,shapes=pyramid_enhance(image,context)
            features[axis,a:b+1]=enhanced[a-lo:b-lo+1]
        segments.append(dict(id=f'L{index:03d}',sample_range=[a,b],context=context,layer_shapes=shapes))
    # The proposal mean is full-record, before restricting requested output region.
    proposal_threshold=7*float(np.mean(features));detections=[];traces=[]
    for segment in segments:
        a,b=segment['sample_range'];axis_traces=[]
        for axis in range(3):
            image=features[axis,a:b+1]
            if float(image.max())<=max(proposal_threshold,1e-12):
                axis_traces.append(dict(axis='XYZ'[axis],status='below_proposal'));continue
            regions,trace=plateau_regions(image);axis_traces.append(dict(trace,axis='XYZ'[axis]))
            for box in regions:detections.append(dict(start=a+box['start'],end=a+box['end'],axis='XYZ'[axis],region=box))
        traces.append(dict(segment,axis_traces=axis_traces))
    merged=[]
    for box in sorted(detections,key=lambda box:box['start']):
        if merged and box['start']<=merged[-1]['end']+16:
            merged[-1]['end']=max(merged[-1]['end'],box['end']);merged[-1]['evidence'].append(box)
        else:merged.append(dict(start=box['start'],end=box['end'],evidence=[box]))
    candidates=[]
    for i,box in enumerate(merged,1):
        a,b=box['start'],box['end']
        if region is not None and (b<region[0] or a>region[1]):continue
        c=dict(id=f'D{i:02d}',start=a,end=b,coarse_start=a,coarse_end=b,refined_start=a,refined_end=b,
            start_s=a/500,end_s=b/500,center_sample=(a+b)//2,sources=[method],status='文献方法响应，待复核',refinement_used=False,
            axis_response_rms=np.sqrt(np.mean(features[:,a:b+1]**2,axis=(1,2))).round(3).tolist(),
            axis_response_kind='无量纲目标特征图RMS，不是原始磁信号幅值',extent_kind='特征连通区域时间投影，非实际损伤长度',component_evidence=box['evidence'])
        if position is not None:c['position_m']=[float(np.min(position[a:b+1])),float(np.max(position[a:b+1]))]
        candidates.append(c)
    path=folder/(run_id+'_'+method+'.png');rendering=render(features,path,detrend=False,display_range=(0,max(float(features.max()),1e-9)))
    rendering['axis']='horizontal time; vertical 200 interpolated circumferential pixels from 32 sensors'
    artifact=folder/(run_id+'_'+method+'.npz');np.savez_compressed(artifact,features=features)
    result=dict(record=record,run_id=run_id,mode=method,version='3.4-experimental',audit=core.audit(record),candidates=candidates,
        source_ids=[SOURCES[method]],literature_segments=traces,tools=['extended_prewitt','gamma2','cubic_peak_envelope','connected_component_plateau']+(['gaussian_pyramid','multiscale_fusion'] if method!='tfo' else []),
        params=dict(POLICY),thresholds=dict(segment_proposal=proposal_threshold),scope_samples=region,annotations_used=False,
        image='/runs/'+record+'/'+path.name,rendering=rendering,signal_artifact='/runs/'+record+'/'+artifact.name,elapsed_s=round(time.time()-began,3),
        metrics=dict(model_call_count=0,unknown_ssr_segments=sum(s['context']['status']=='unknown' for s in segments)),
        detector='experimental literature component localization',implementation_kind='engineering adaptation, not verified paper reproduction',
        adaptations=['3轴分别处理后时间区间融合，原文16通道单方向数据','环周周期样条到200通道','201点去趋势和32点上下文为当前数据适配',
            '三轴共享全记录归一化/提议均值','峰值样条采用自然边界并保护非负上包络','平台至少5步、面积至少20像素、区间间隙15点为明确工程规则',
            '高斯抗混叠下采样、双线性恢复尺度','未知SSR等权多尺度；实测SSR按式14递推、最粗层初始化及奇数核调整'],
        limitations=['未证明优于当前基线或复现论文性能','特征幅值不可与Hilbert信号幅值直接比较','SSR不能从图像或IMU猜测','论文高低SSR叙述和式5/式14边界定义需作者确认'])
    (folder/(run_id+'_'+method+'.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    return result
