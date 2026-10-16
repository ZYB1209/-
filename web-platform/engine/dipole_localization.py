"""Conditional center-section point-dipole inversion. No physical calibration claim."""
import numpy as np
from adaptive_signal import signal  # Initializes the repository's bundled SciPy path.
from scipy.optimize import least_squares

POLICY=dict(version='center-dipole-v1',relative_residual_max=.25,minimum_active_channels=6,
    sign_coherence_min=.8,maximum_depth_span_fraction=.5,depth_profile_points=65,angle_profile_points=720,
    description='单个轴向点偶极子、同心圆环、中心截面与已知假设提离下的条件反演；未经有限元和实测标定。')

def shape(theta,r,angle,S):
    return np.maximum(S*S+r*r-2*S*r*np.cos(theta-angle),1e-18)**(-1.5)

def fit_center_section(y,noise,config):
    y=np.asarray(y,float);noise=np.asarray(noise,float)
    if y.shape!=(32,) or noise.shape!=(32,) or not np.isfinite(y).all() or not np.isfinite(noise).all() or np.any(noise<0):raise ValueError('偶极子拟合需32通道有限有符号信号和噪声尺度')
    R=config['rope_diameter_mm']/2000;h=config['lift_off_mm']/1000;S=R+h
    if h<=0:return dict(status='rejected',accepted=False,reasons=['点源模型暂要求正提离，避免零距离奇点'],model_depth_mm=None,model_xy_mm=None)
    sign=1 if config['direction']=='counterclockwise' else -1
    theta=np.deg2rad(config['channel1_angle_deg']+sign*np.arange(32)*11.25)
    scale=float(np.sqrt(np.mean(y*y)));floor=max(scale*1e-5,1e-12)
    sigma=np.maximum(noise,max(float(np.median(noise))*.2,floor))
    weights=np.minimum((np.median(sigma)/sigma)**2,25);weights/=np.mean(weights)
    active=np.abs(y)>=2*sigma;count=int(active.sum())
    coherence=float(max(np.sum(y[active]>0),np.sum(y[active]<0))/max(count,1))
    if scale<1e-12:return dict(status='rejected',accepted=False,reasons=['截面信号近零'],model_depth_mm=None,model_xy_mm=None)
    yn=y/scale
    def project(r,angle):
        f=shape(theta,r,angle,S);f=f/np.max(f)
        A=float(np.sum(weights*yn*f)/np.sum(weights*f*f))
        return (A*f-yn)*np.sqrt(weights),A,f
    def residual(x):return project(x[0]*R,x[1])[0]
    fits=[]
    peak=theta[int(np.argmax(np.abs(y)))];initials=[]
    # The linear negative-power transform provides a model-consistency initial estimate.
    if count>=6:
        ids=active;P=np.abs(y[ids]);Y=P**(-2/3)
        coef=np.linalg.lstsq(np.stack([np.ones(ids.sum()),np.cos(theta[ids]),np.sin(theta[ids])],axis=1),Y,rcond=None)[0]
        alpha,beta,gamma=coef;kappa=float(np.hypot(beta,gamma)/alpha) if alpha>0 else float('inf')
        if 0<=kappa<1:
            r=S*kappa/(1+np.sqrt(1-kappa*kappa))
            if 0<=r<=R:initials.append((float(r/R),float(np.arctan2(-gamma,-beta))))
    initials += [(r,float(peak+offset)) for r in (.15,.5,.85,.98) for offset in (0,np.pi/2,np.pi,-np.pi/2)]
    for initial in initials:
        fit=least_squares(residual,initial,bounds=([0,-4*np.pi],[1,4*np.pi]),max_nfev=150,ftol=1e-10,xtol=1e-10,gtol=1e-10)
        if fit.success and np.isfinite(fit.fun).all():fits.append(fit)
    if not fits:return dict(status='rejected',accepted=False,reasons=['多初值求解未收敛'],model_depth_mm=None,model_xy_mm=None)
    best=min(fits,key=lambda f:np.dot(f.fun,f.fun));r=float(best.x[0]*R);angle=float(best.x[1]%(2*np.pi))
    error,A,f=project(r,angle);prediction=A*f*scale;cost=float(np.dot(error,error))
    relative=float(np.sqrt(cost/np.sum(weights*yn*yn)))
    depths=np.linspace(0,R,POLICY['depth_profile_points']);angles=np.linspace(0,2*np.pi,POLICY['angle_profile_points'],endpoint=False)
    # Include the continuous optimum in the angular grid, preserving its profile cost.
    angles=np.r_[angles,angle];profile=[]
    for depth in depths:
        F=shape(theta[None,:],R-depth,angles[:,None],S);F/=np.max(F,axis=1,keepdims=True)
        amplitudes=np.sum(weights*yn*F,axis=1)/np.sum(weights*F*F,axis=1)
        costs=np.sum(weights*(amplitudes[:,None]*F-yn)**2,axis=1);profile.append(float(costs.min()))
    depths=np.r_[depths,R-r];profile=np.r_[profile,cost]
    noise_fraction=float(np.sqrt(np.mean(weights*sigma*sigma))/scale)
    tolerance=3.84*noise_fraction**2
    compatible=depths[profile<=cost+tolerance+1e-9]
    depth_range=[float(compatible.min()*1000),float(compatible.max()*1000)]
    reasons=[]
    if count<POLICY['minimum_active_channels']:reasons.append('可用环周通道不足，弱侧幅值不能约束深度')
    if coherence<POLICY['sign_coherence_min']:reasons.append('有效通道有符号响应混杂，不符合中心截面单点偶极子同号形态')
    if relative>POLICY['relative_residual_max']:reasons.append('整圈原始信号拟合残差过大，单磁源模型失配')
    if depth_range[1]-depth_range[0]>R*1000*POLICY['maximum_depth_span_fraction']:reasons.append('噪声尺度下可相容深度范围过宽，深度不可稳定辨识')
    if r/R<.02:reasons.append('等效磁源接近绳中心，周向角不可辨识')
    if best.x[0]>.9999:reasons.append('拟合贴近绳表面约束边界，应检查几何、提离与背景')
    accepted=not reasons;depth_mm=float((R-r)*1000);angle_deg=float(np.rad2deg(angle))
    # A is a nuisance amplitude in CSV units*m^3; uncalibrated CSV values cannot yield a physical magnetic moment.
    fraw=shape(theta,r,angle,S);amplitude=float(np.sum(weights*y*fraw)/np.sum(weights*fraw*fraw))
    return dict(status='conditional_estimate' if accepted else 'rejected',accepted=accepted,reasons=reasons,
        model_depth_mm=round(depth_mm,4) if accepted else None,
        model_angle_deg=round(angle_deg,3) if accepted else None,
        model_xy_mm=[round(float(r*1000*np.cos(angle)),4),round(float(r*1000*np.sin(angle)),4)] if accepted else None,
        best_fit_diagnostic=dict(depth_mm=round(depth_mm,4),angle_deg=round(angle_deg,3),radial_fraction=round(r/R,6)),
        relative_residual=round(relative,6),active_channels=count,sign_coherence=round(coherence,4),
        compatible_depth_range_mm=[round(v,4) for v in depth_range],
        range_kind='估计噪声尺度与单点模型下的剖面可相容范围；不是已标定置信区间，也不包含几何系统误差',
        multi_start_count=len(initials),converged_starts=len(fits),amplitude_csv_units_m3=amplitude,
        signed_observation=y.round(8).tolist(),predicted_profile=prediction.round(8).tolist(),noise_scale=sigma.round(8).tolist(),
        calibrated=False,physical_depth_mm=None,physical_xy_mm=None,broken_wires=None,policy=POLICY)

def candidate_depth(candidate,residual,config,others,center_sample=None):
    a,b=candidate['refined_start'],candidate['refined_end'];axis='XYZ'.index(config['axial_axis']);n=len(residual)
    mask=np.ones(n,bool)
    for c in [candidate]+list(others):mask[max(0,c['refined_start']-16):min(n,c['refined_end']+17)]=False
    lo=max(0,a-512);hi=min(n,b+513);ids=np.flatnonzero(mask[lo:hi])+lo
    if len(ids)<64:return dict(status='rejected',accepted=False,reasons=['相邻非候选背景不足，不能估计有符号背景与噪声'],model_depth_mm=None,model_xy_mm=None)
    bg=residual[ids,axis,:];baseline=np.median(bg,axis=0);noise=1.4826*np.median(np.abs(bg-baseline),axis=0)
    block=residual[a:b+1,axis,:]-baseline
    if center_sample is not None and (type(center_sample) is not int or not a<=center_sample<=b):raise ValueError('固定截面必须位于候选范围内')
    center=a+int(np.argmax(np.mean(block*block,axis=1))) if center_sample is None else center_sample
    left=max(a,center-1);right=min(b+1,center+2)
    observation=np.mean(residual[left:right,axis,:]-baseline,axis=0)
    fit=fit_center_section(observation,noise,config)
    overlap=[c['id'] for c in others if c is not candidate and c['refined_start']<=b and c['refined_end']>=a]
    # Overlapping detector proposals can describe the same event; they do not establish multiple physical sources.
    fit['overlapping_proposals']=overlap
    fit['overlap_note']='重叠检测提议可能是同一事件的重复算法证据；物理磁源数量需复核，不能按提议数量认定多缺陷' if overlap else None
    fit.update(center_sample=center,input_axis=config['axial_axis'],axis_source=config.get('axial_axis_source'),center_selection=(f"候选区间{config['axial_axis']}分量去背景信号能量最大截面" if center_sample is None else '固定采样截面对照')+'，邻近最多3点平均；未验证等于真实缺陷中心',
        background_samples=int(len(ids)),geometry_source=config['geometry_source'],
        assumptions=['单个轴向点磁偶极子','同心圆环与等间距通道','小断口间隙','选取截面假设为缺陷中心','自由空间等效源，未显式考虑钢丝绳磁性边界','几何、提离、增益和角度零点未经实测标定'])
    return fit
