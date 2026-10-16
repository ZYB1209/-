"""Annotation-free spatial fusion and hysteresis interval extraction.

Signal tensors use axes x samples x ring channels. All thresholds and interval
bounds are computed from signals, never labels or per-record tuned overrides.
"""
import numpy as np
import core

CONSENSUS_POLICY=dict(match_iou=.3,adaptive_weight=.25,max_shift_samples=40,max_shift_fraction=.25,
    eligible_segment_status='accepted',version='reference-consensus-v1')

def reference_candidates(record):
    result=core.temporal_channel(record)
    candidates=[]
    for interval in result['intervals']:
        candidate=core.refine(record,dict(start=interval['start'],end=interval['end'],sources=['reference_spatial_fusion']))
        candidate['reference_peak']=interval['peak'];candidates.append(candidate)
    return candidates,result['score'],result['threshold']

def fuse_candidates(reference,adaptive,segments,*,n_samples):
    """Keep all reference candidates; bounded boundary changes need verified support.

    Unmatched adaptive responses remain review evidence, not additional primary
    detections. Annotation access is deliberately absent from this function.
    """
    if type(n_samples)is not int or n_samples<1:raise ValueError('记录长度无效')
    def bounds(candidate):
        a,b=candidate['refined_start'],candidate['refined_end']
        if type(a)is not int or type(b)is not int or not 0<=a<=b<n_samples:raise ValueError('候选边界无效')
        return [a,b]
    primary_bounds=[bounds(c) for c in reference];adaptive_bounds=[bounds(c) for c in adaptive]
    eligible=[i for i,p in enumerate(adaptive_bounds) if any(s['status']=='accepted' and
        s['sample_range'][0]<=sum(p)//2<=s['sample_range'][1] for s in segments)]
    pairs=core.match_intervals(primary_bounds,[adaptive_bounds[i] for i in eligible],CONSENSUS_POLICY['match_iou'])
    matched={i:eligible[j] for i,j in pairs};output=[]
    for index,candidate in enumerate(reference):
        original=primary_bounds[index];final=list(original);proposal=None
        if index in matched:
            proposal=adaptive_bounds[matched[index]]
            limit=min(CONSENSUS_POLICY['max_shift_samples'],int((original[1]-original[0]+1)*CONSENSUS_POLICY['max_shift_fraction']))
            final=[int(round(v+np.clip(CONSENSUS_POLICY['adaptive_weight']*(proposal[k]-v),-limit,limit))) for k,v in enumerate(original)]
            final=[max(0,final[0]),min(n_samples-1,final[1])]
            if final[0]>final[1]:final=list(original)
        changed=final!=original
        item=dict(candidate,refined_start=final[0],refined_end=final[1],
            sources=list(candidate.get('sources',[]))+(['verified_adaptive_support'] if proposal is not None else []),
            boundary_consensus=dict(reference_range=original,adaptive_range=proposal,final_range=final,
                changed=changed,policy=dict(CONSENSUS_POLICY)),
            refinement_used=candidate.get('refinement_used',False) or changed,
            refinement_reason='通过验证的自适应响应参与受限边界融合' if changed else candidate.get('refinement_reason','保留参考边界'),
            status='参考与自适应响应支持，待复核' if proposal is not None else '参考响应候选，待复核')
        output.append(item)
    used=set(matched.values())
    supplementary=[dict(c,primary=False,review_reason='自适应独有响应；不自动加入主候选') for i,c in enumerate(adaptive) if i not in used]
    return output,supplementary,dict(policy=dict(CONSENSUS_POLICY),reference_count=len(reference),
        adaptive_count=len(adaptive),eligible_adaptive_count=len(eligible),matched_count=len(matched),
        boundary_changes=sum(c['boundary_consensus']['changed'] for c in output),
        primary_count=len(output),supplementary_count=len(supplementary),annotations_used=False)

def response_score(tensor,*,spatial=True,fusion='geometric',rms_window=51):
    x=np.asarray(tensor,dtype=float)
    if type(rms_window)is not int or rms_window<1:raise ValueError('RMS窗口须为正整数')
    if x.ndim!=3 or x.shape[0]!=3 or x.shape[2]!=32 or x.shape[1]<3 or not np.isfinite(x).all():
        raise ValueError('响应融合要求有限的3轴×采样点×32通道矩阵')
    if spatial:
        spectrum=np.fft.rfft(x,axis=2);spectrum[:,:,:2]=0
        distance=np.minimum(np.arange(32),32-np.arange(32))
        kernel=np.exp(-distance**2/(2*2.5**2));kernel-=kernel.mean();kernel/=np.sqrt(np.sum(kernel**2))
        response=np.abs(np.fft.irfft(spectrum*np.fft.rfft(kernel),n=32,axis=2))
    else:response=np.abs(x)
    near=np.maximum.reduce([np.roll(response,k,axis=2) for k in range(-2,3)])
    if fusion=='geometric':fused=np.prod(np.maximum(near,1e-8),axis=0)**(1/3)
    elif fusion=='two_axis':fused=np.partition(near,1,axis=0)[1]
    elif fusion=='max':fused=np.max(near,axis=0)
    else:raise ValueError('未知多轴融合方式')
    return np.sqrt(core.smooth(np.max(fused,axis=1)**2,rms_window))

def extract_intervals(score,*,high_method='quantile',high_quantile=.925,threshold_mad=5.,low_ratio=.75,
                      minimum_samples=40,seed_samples=12,merge_gap=25,max_expand=150,max_expand_fraction=None):
    score=np.asarray(score,dtype=float)
    if score.ndim!=1 or not len(score) or not np.isfinite(score).all() or np.any(score<0):raise ValueError('响应曲线无效')
    if not np.isfinite([high_quantile,low_ratio,threshold_mad]).all() or not 0<high_quantile<1 or not 0<low_ratio<=1 or threshold_mad<=0:raise ValueError('检测阈值参数无效')
    for value in (minimum_samples,seed_samples):
        if type(value)is not int or value<1:raise ValueError('最短区间须为正整数')
    for value in (merge_gap,max_expand):
        if type(value)is not int or value<0:raise ValueError('合并/扩展范围须为非负整数')
    if max_expand_fraction is not None and not 0<=max_expand_fraction<=1:raise ValueError('相对扩展上限须在0到1之间')
    baseline=float(np.median(score));mad=float(1.4826*np.median(np.abs(score-baseline)))
    if high_method=='quantile':high=float(np.quantile(score,high_quantile))
    elif high_method=='mad':high=baseline+threshold_mad*mad
    else:raise ValueError('未知阈值方式')
    high=max(high,1e-6);low=baseline+low_ratio*max(high-baseline,0)
    above=score>high
    if high>=float(score.max()) and high>baseline:above=score>=high
    seeds=core.masks_to_intervals(above,seed_samples,merge_gap)
    expanded=[]
    for a,b in seeds:
        expansion=max_expand if max_expand_fraction is None else min(max_expand,int((b-a+1)*max_expand_fraction))
        left,right=a,b
        while left>max(0,a-expansion) and score[left-1]>low:left-=1
        while right<min(len(score)-1,b+expansion) and score[right+1]>low:right+=1
        if expanded and left<=expanded[-1][1]+merge_gap+1:expanded[-1][1]=max(right,expanded[-1][1])
        else:expanded.append([int(left),int(right)])
    intervals=[span for span in expanded if span[1]-span[0]+1>=minimum_samples]
    return intervals,dict(high=float(high),low=float(low),baseline=baseline,mad=mad,seed_count=len(seeds),
        method=high_method,high_quantile=high_quantile,threshold_mad=threshold_mad,low_ratio=low_ratio,
        minimum_samples=minimum_samples,seed_samples=seed_samples,merge_gap=merge_gap,max_expand=max_expand,
        max_expand_fraction=max_expand_fraction,
        annotations_used=False)
