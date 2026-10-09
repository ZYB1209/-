"""Eight bounded operators, axes x time x channels. Presets remain unchanged in v2."""
import numpy as np
from adaptive_signal import mean,signal,ndimage,kalman_baseline,diagonal_filter,circum_spline,morph
from parameter_guard import TOOLS

def execute(z, plan):
    x=np.array(z,dtype=float,copy=True); envelope=None; score=None; logs=[]
    p=plan['actual_parameters']
    for tool in plan['selected_pipeline']:
        if tool==TOOLS[0]:
            x=np.array([kalman_baseline(a,p['baseline_window'])[0] for a in x]) if p['baseline_mode']=='kalman' else x-mean(x,p['baseline_window'],axis=1)
        elif tool==TOOLS[1]:
            x=np.array([diagonal_filter(a,p['stripe_slope'],p['diagonal_window']) for a in x])
        elif tool==TOOLS[2]:
            x=x-ndimage.median_filter(x,size=(1,1,p['median_span']),mode='wrap')
        elif tool==TOOLS[3]:
            axes=[]
            for a in x:
                hi=circum_spline(a,360); parts=[]
                for part in (np.maximum(hi,0),np.maximum(-hi,0)):
                    closed=morph(part,(1,p['line_SE']),'close')
                    residual=np.maximum(closed-morph(closed,(1,p['line_SE']),'open'),0)
                    # Cleanup operates on native time; temporal interpolation is deliberately not implicit.
                    parts.append(morph(residual,(max(1,p['cleanup_SE']//5),p['cleanup_SE']),'open'))
                axes.append(circum_spline(parts[0]-parts[1],a.shape[1]))
            x=np.array(axes)
        elif tool==TOOLS[4]:
            c=x.shape[2];d=np.minimum(np.arange(c),c-np.arange(c));kernel=np.exp(-d*d/(2*p['matched_sigma']**2));kernel-=kernel.mean();kernel/=np.sum(np.abs(kernel))
            x=np.fft.irfft(np.fft.rfft(x,axis=2)*np.fft.rfft(kernel)[None,None,:],n=c,axis=2)
        elif tool==TOOLS[5]: envelope=np.abs(signal.hilbert(x,axis=1))
        elif tool==TOOLS[6]: score=np.max(envelope,axis=(0,2))
        elif tool==TOOLS[7]:
            envelope=np.sqrt(np.maximum(mean(envelope**2,p['temporal_window'],axis=1),0))
            score=np.max(envelope,axis=(0,2))
        else: raise ValueError('未注册 DSP 工具')
        if not np.isfinite(x).all() or (score is not None and not np.isfinite(score).all()):
            raise ValueError('DSP产生非有限数值')
        logs.append(dict(tool=tool,parameters=p,shape=list(x.shape)))
    return envelope,score,logs
