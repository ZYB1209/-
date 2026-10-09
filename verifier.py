"""Unsupervised engineering diagnostics, not calibrated safety probabilities."""
import numpy as np
from adaptive_signal import preservation_guard,signal,mean,detect_envelope,localized_score

def verify(before,after):
    numerical=after.shape==before.shape and np.isfinite(after).all()
    if not numerical:return dict(verdict='reject',warnings=['数值或输出形状异常'],numerical_safety=False)
    reference=np.abs(signal.hilbert(before,axis=1)); guard=preservation_guard(before,after)
    score=localized_score(before); med=np.median(score);mad=1.4826*np.median(abs(score-med))
    peaks,_=signal.find_peaks(score,prominence=max(4*mad,.08*np.max(score),1e-8),distance=40)
    post=np.max(after,axis=(0,2));pre=np.max(reference,axis=(0,2));shifts=[]
    for peak in peaks:
        a=max(0,peak-20);b=min(len(score),peak+21)
        # Compare envelope peaks on both sides, not raw/localized peak vs envelope.
        shifts.append(int(abs(int(np.argmax(post[a:b]))-int(np.argmax(pre[a:b])))))
    active=score>np.quantile(score,.85); background=~active
    reduction=1-float(np.mean(after[:,background,:]**2)/max(np.mean(reference[:,background,:]**2),1e-12)) if np.any(background) else None
    ratio=float(np.quantile(after,.995)/max(np.quantile(reference,.995),1e-12))
    warnings=[]
    if not guard['passes']:warnings.append('局部响应保留不足')
    if shifts and max(shifts)>15:warnings.append('局部地标时间漂移超过15点')
    if ratio>10 or (np.max(reference)>1e-8 and np.max(after)<1e-12):warnings.append('动态范围异常或非零输入变为全零')
    if reduction is not None and reduction<-.1:warnings.append('非局部背景能量增加')
    pre_count=len(detect_envelope(reference)[0]);post_count=len(detect_envelope(after)[0])
    return dict(local_response_preservation=guard['minimum_retention'],landmark_count=guard['landmark_count'],
        peak_shift_samples=max(shifts) if shifts else None,background_reduction=reduction,
        output_dynamic_range_ratio=ratio,candidate_count_change=post_count-pre_count,
        numerical_safety=True,warnings=warnings,verdict='reject' if warnings else 'accept',
        interpretation='无标注地标/非局部能量代理；背景下降也可能误去掉宽响应，不保证真实缺陷保留')
