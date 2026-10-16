"""Unsupervised engineering diagnostics, not calibrated safety probabilities."""
import numpy as np
from adaptive_signal import preservation_guard,signal,mean,detect_envelope,localized_score

VERSION='anchored-v1'
LIMITS=dict(max_peak_shift_samples=15,max_dynamic_range_ratio=10.,max_background_increase=.1)

def landmark_shifts(reference,after,peaks):
    """Track a reference axis and nearby ring channels, not a switching global max."""
    n=reference.shape[1];details=[]
    for peak in peaks:
        a=max(0,int(peak)-20);b=min(n,int(peak)+21)
        axis,local,channel=np.unravel_index(np.argmax(reference[:,a:b,:]),reference[:,a:b,:].shape)
        anchor=a+int(local)
        channels=(int(channel)+np.arange(-2,3))%reference.shape[2]
        pre=np.max(reference[axis,a:b][:,channels],axis=1)
        post=np.max(after[axis,a:b][:,channels],axis=1)
        pre_peak=a+int(np.argmax(pre));post_peak=a+int(np.argmax(post))
        # A maximum on the search boundary is not an identifiable local reference peak.
        identifiable=a<pre_peak<b-1 and float(pre.max())>1e-9
        details.append(dict(seed_sample=int(peak),axis=int(axis),channel=int(channel)+1,
            search_range=[a,b-1],reference_peak_sample=pre_peak,processed_peak_sample=post_peak,
            shift_samples=abs(post_peak-pre_peak) if identifiable else None,
            retention=float(post.max()/max(pre.max(),1e-9)),identifiable=identifiable))
    return details

def verify(before,after):
    numerical=after.shape==before.shape and np.isfinite(after).all() and np.isfinite(before).all() and np.all(after>=0)
    if not numerical:return dict(verdict='reject',warnings=['数值或输出形状异常'],numerical_safety=False)
    reference=np.abs(signal.hilbert(before,axis=1)); guard=preservation_guard(before,after)
    score=localized_score(before); med=np.median(score);mad=1.4826*np.median(abs(score-med))
    peaks,_=signal.find_peaks(score,prominence=max(4*mad,.08*np.max(score),1e-8),distance=40)
    post=np.max(after,axis=(0,2));pre=np.max(reference,axis=(0,2));legacy_shifts=[]
    for peak in peaks:
        a=max(0,peak-20);b=min(len(score),peak+21)
        # Compare envelope peaks on both sides, not raw/localized peak vs envelope.
        legacy_shifts.append(int(abs(int(np.argmax(post[a:b]))-int(np.argmax(pre[a:b])))))
    landmarks=landmark_shifts(reference,after,peaks)
    shifts=[x['shift_samples'] for x in landmarks if x['identifiable']]
    active=score>np.quantile(score,.85); background=~active
    reduction=1-float(np.mean(after[:,background,:]**2)/max(np.mean(reference[:,background,:]**2),1e-12)) if np.any(background) else None
    ratio=float(np.quantile(after,.995)/max(np.quantile(reference,.995),1e-12))
    warnings=[]
    if not guard['passes']:warnings.append('局部响应保留不足')
    if shifts and max(shifts)>LIMITS['max_peak_shift_samples']:warnings.append('局部地标时间漂移超过15点')
    if ratio>LIMITS['max_dynamic_range_ratio'] or (np.max(reference)>1e-8 and np.max(after)<1e-12):warnings.append('动态范围异常或非零输入变为全零')
    if reduction is not None and reduction<-LIMITS['max_background_increase']:warnings.append('非局部背景能量增加')
    pre_count=len(detect_envelope(reference)[0]);post_count=len(detect_envelope(after)[0])
    return dict(local_response_preservation=guard['minimum_retention'],landmark_count=guard['landmark_count'],
        peak_shift_samples=max(shifts) if shifts else None,background_reduction=reduction,
        legacy_global_peak_shift_samples=max(legacy_shifts) if legacy_shifts else None,
        landmark_details=landmarks,verifier_version=VERSION,
        output_dynamic_range_ratio=ratio,candidate_count_change=post_count-pre_count,
        numerical_safety=True,warnings=warnings,verdict='reject' if warnings else 'accept',
        interpretation='同轴邻近通道地标/非局部能量代理；无法定位的参考峰不计漂移。不是实际缺陷位置或召回率，不保证宽缺陷保留')
