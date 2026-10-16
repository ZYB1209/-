"""Strict DSP templates; model selects discrete controls, never arbitrary code/numbers."""
import numpy as np

TOOLS = ('remove_slow_baseline','suppress_directional_strand_pattern',
 'remove_circumferential_background','suppress_broad_morphological_background',
 'enhance_local_channel_response','extract_hilbert_envelope',
 'fuse_multiaxis_response','aggregate_temporal_response')
TEMPLATES = {
 'A': [TOOLS[0],TOOLS[5],TOOLS[6],TOOLS[7]],
 'B': [TOOLS[0],TOOLS[1],TOOLS[2],TOOLS[5],TOOLS[6],TOOLS[7]],
 'C': [TOOLS[0],TOOLS[3],TOOLS[5],TOOLS[6],TOOLS[7]],
 'D': [TOOLS[0],TOOLS[4],TOOLS[5],TOOLS[6],TOOLS[7]]}
CONTROLS = {'baseline_mode':('moving_average','kalman'), 'strength':('mild','medium','strong'),
 'period_scale':('2x','3x','4x'),'span':('small','medium','large'),
 'structure_scale':('small','medium','large'),'cleanup_strength':('mild','medium'),
 'window':('short','medium','long')}

def guard(plan, features, n, channels=32):
    if not isinstance(plan,dict) or set(plan)-{'template','controls','disabled','reason','risk_notes','verification_focus'}:
        raise ValueError('计划字段不合法')
    template=plan.get('template'); controls=plan.get('controls',{}); disabled=plan.get('disabled',[])
    if template not in TEMPLATES or not isinstance(controls,dict) or set(controls)-set(CONTROLS):
        raise ValueError('未知模板或参数')
    for k,v in controls.items():
        if v not in CONTROLS[k]: raise ValueError('参数必须为注册的离散档位：'+k)
    if not isinstance(disabled,list) or len(set(disabled))!=len(disabled) or any(x not in TEMPLATES[template][1:-3] for x in disabled):
        raise ValueError('只能关闭模板中的可选抑制/增强工具')
    if n<32 or channels!=32: raise ValueError('v3工具要求至少32点和32环周通道')
    period=features.get('strand_period_samples',0) if features.get('strand_ridge_fraction',0)>=.12 else 0
    scale=int(controls.get('period_scale','3x')[0])
    base=np.clip(scale*period if period else 201,81,501)
    strength=controls.get('strength','medium'); factor={'mild':1.3,'medium':1.,'strong':.75}[strength]
    params=dict(baseline_window=int(np.clip(base*factor,3,min(501,n-1))),
        baseline_mode=controls.get('baseline_mode','moving_average'),
        diagonal_window={'mild':6,'medium':10,'strong':14}[strength],
        median_span={'small':9,'medium':17,'large':25}[controls.get('span','medium')],
        line_SE={'small':60,'medium':90,'large':120}[controls.get('structure_scale','medium')],
        cleanup_SE={'mild':8,'medium':20}[controls.get('cleanup_strength','mild')],
        temporal_window=min({'short':11,'medium':21,'long':41}[controls.get('window','medium')],n),
        orientation='measured',stripe_slope=features.get('stripe_slope_samples_per_channel',0),
        matched_sigma=2.5)
    if template=='B' and features.get('strand_ridge_fraction',0)<.12 and TOOLS[1] not in disabled:
        disabled=disabled+[TOOLS[1]]
    return dict(template=template,selected_pipeline=[x for x in TEMPLATES[template] if x not in disabled],
        requested_controls=controls,actual_parameters=params,disabled=disabled,
        reason=str(plan.get('reason',''))[:600],risk_notes=plan.get('risk_notes',[]),
        verification_focus=plan.get('verification_focus',[]),guard_version='v3.0')
