"""Circumferential response analysis and conditional center-section depth inversion."""
import html,json,math,secrets
from pathlib import Path
import numpy as np

DEFAULT_GEOMETRY=dict(axial_axis='X',axial_axis_source='user_confirmed',rope_diameter_mm=32.,lift_off_mm=2.,
    channel1_angle_deg=0.,direction='counterclockwise',channel_count=32,channel_order_confirmed=False,
    geometry_source='assumed',angle_calibrated=False,
    note='直径32mm、提离2mm、通道1为0°且逆时针等间距编号均为示意假设；X为用户确认的轴向分量。')

def geometry():
    config=dict(DEFAULT_GEOMETRY)
    path=Path(__file__).with_name('.local_cross_section.json')
    if path.exists():config.update(json.loads(path.read_text(encoding='utf-8')))
    if config['axial_axis'] not in ('X','Y','Z') or config['direction'] not in ('counterclockwise','clockwise'):raise ValueError('截面轴向或编号方向无效')
    for key,low,high in (('rope_diameter_mm',1,200),('lift_off_mm',0,100),('channel1_angle_deg',-360,360)):
        value=config[key]
        if type(value) not in (int,float) or not math.isfinite(value) or not low<=value<=high:raise ValueError('截面示意参数超出范围')
    # A settings file cannot establish physical calibration or unlock depth estimates.
    config.update(geometry_source='assumed',angle_calibrated=False,channel_order_confirmed=False,channel_count=32)
    config['note']=f"直径{config['rope_diameter_mm']}mm、提离{config['lift_off_mm']}mm、通道1为{config['channel1_angle_deg']}°且{'逆时针' if config['direction']=='counterclockwise' else '顺时针'}等间距编号均为示意假设；{config['axial_axis']}为轴向分量。"
    return config

def estimate_profile(profile,config):
    """Interpolate the maximum and its cyclic neighbors; report angular spread, never a CI."""
    from adaptive_signal import CubicSpline
    p=np.asarray(profile,dtype=float)
    if p.shape!=(32,) or not np.isfinite(p).all() or np.any(p<0):raise ValueError('截面响应要求32个有限非负值')
    baseline=float(np.median(p));peak=int(np.argmax(p));prominence=float(p[peak]-baseline);step=360/32
    if prominence<=max(float(p[peak])*1e-6,1e-12):
        return dict(status='no_direction',peak_channel_1based=None,channel_relative_angle_deg=None,
            assumed_angle_deg=None,response_spread_deg=None,lobes=[],peak_to_median_ratio=None,
            interpretation='环周分布近似均匀，不能给出主响应方位')
    xp=np.array([-step,0.,step]);values=p[[(peak-1)%32,peak,(peak+1)%32]]
    grid=np.linspace(-step,step,181);interpolated=np.maximum(CubicSpline(xp,values,bc_type='natural')(grid),0)
    relative=float((peak*step+grid[int(np.argmax(interpolated))])%360)
    sign=1 if config['direction']=='counterclockwise' else -1
    physical=lambda angle:float((config['channel1_angle_deg']+sign*angle)%360)
    half=baseline+.5*prominence;left=right=0
    while left<16 and p[(peak-left-1)%32]>=half:left+=1
    while right<16 and left+right<31 and p[(peak+right+1)%32]>=half:right+=1
    spread=min(360.,(left+right+1)*step)
    local=[i for i in range(32) if p[i]>=p[(i-1)%32] and p[i]>p[(i+1)%32] and p[i]>=baseline+.7*prominence]
    chosen=[]
    for i in sorted(local,key=lambda i:-p[i]):
        if all(min((i-j)%32,(j-i)%32)>=3 for j in chosen):chosen.append(i)
    lobes=[dict(channel_1based=i+1,assumed_angle_deg=round(physical(i*step),2),relative_height=round(float((p[i]-baseline)/prominence),3)) for i in chosen]
    ambiguous=len(lobes)>1 or spread>=90
    return dict(status='ambiguous' if ambiguous else 'relative_direction',peak_channel_1based=peak+1,
        channel_relative_angle_deg=round(relative,2),assumed_angle_deg=round(physical(relative),2),
        response_spread_deg=round(spread,2),lobes=lobes,
        peak_to_median_ratio=round(float(p[peak]/baseline),3) if baseline>1e-12 else None,
        interpolation='最强通道及其环周相邻通道的自然三次样条，0.125°显示采样；不是实测角分辨率',
        interpretation='存在多个强环周峰或宽分布，方位不唯一' if ambiguous else '主响应方位可作人工复核线索，尚非真实缺陷坐标')

def localize(candidate,residual,config,others=()):
    a,b=candidate['refined_start'],candidate['refined_end'];n=len(residual)
    if type(a)is not int or type(b)is not int or not 0<=a<=b<n:raise ValueError('截面候选范围无效')
    lo=max(0,a-64);hi=min(n,b+65);block=residual[lo:hi]
    actual=np.ptp(block,axis=0)
    robust=np.quantile(block,.95,axis=0)-np.quantile(block,.05,axis=0)
    axis='XYZ'.index(config['axial_axis']);estimate=estimate_profile(robust[axis],config)
    overlap=[c['id'] for c in others if c is not candidate and c['refined_start']<hi and c['refined_end']>=lo]
    notes=['环周通道顺序、角度零点和编号方向未实测确认。',
        '保留通道相对幅值：对原始数据做201点移动均值去趋势，未做逐通道MAD均衡。',
        f"方位取{config['axial_axis']}分量每通道95%–5%稳健峰峰值；论文使用轴向峰峰值，本实现作抗离群工程适配，未复现其精度。",
        '环周半峰响应宽度是信号扩散范围，不是角度置信区间或实际缺陷宽度。',
        '通道增益与提离差异尚未标定，可能改变主峰位置。',
        '真实径向深度与截面坐标未经标定；红色射线仅指响应方向；条件模型深度单独记录，不估计断丝数量。']
    if overlap:notes.append('用于提取双极性波形的±64点上下文包含其它候选：'+','.join(overlap)+'；方位可能混合。')
    if lo==0 or hi==n:notes.append('响应提取上下文触及记录边缘。')
    from dipole_localization import candidate_depth
    depth=candidate_depth(candidate,residual,config,others)
    notes.append('新增整圈有符号中心截面偶极子反演；结果仅在假设几何和单磁源模型下成立，不是实测缺陷深度。')
    notes.extend(depth.get('reasons',[]))
    return dict(candidate_id=candidate['id'],sample_range=[a,b],time_range_s=[candidate['start_s'],candidate['end_s']],
        amplitude_scope_samples=[lo,hi-1],axial_axis=config['axial_axis'],**estimate,
        geometry=dict(config),depth_mm=None,defect_xy_mm=None,broken_wires=None,physical_angle_deg=None,
        axis_profiles_peak_to_peak=actual.round(6).tolist(),axis_profiles_robust_peak_to_peak=robust.round(6).tolist(),
        amplitude_units='原始CSV磁通道单位（未经磁感应强度标定）',
        depth_model=depth,calibrated=False,annotations_used=False,notes=notes,source=dict(id='P2025UCA',pdf_pages=[4,5],method='环周相邻通道插值定位'))

def render_svg(data):
    """Cross-section ray has no radial endpoint, so it cannot imply an estimated depth."""
    e=lambda s:html.escape(str(s),quote=True)
    cfg=data['geometry'];cx,cy=260,285;ratio=1+2*cfg['lift_off_mm']/cfg['rope_diameter_mm'];rope=min(155,180/ratio);sensor=rope*ratio
    def point(angle,r):v=math.radians(angle);return cx+r*math.cos(v),cy-r*math.sin(v)
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1150" height="650" viewBox="0 0 1150 650"><rect width="1150" height="610" fill="#fff"/><g font-family="Microsoft YaHei,Arial" fill="#243746">',
        f'<text x="24" y="34" font-size="23">{e(data["candidate_id"])} · 环周响应方位与截面示意</text>',
        f'<text x="24" y="66" font-size="15">本次拟合分量{data["axial_axis"]}；示意直径{cfg["rope_diameter_mm"]}mm / 提离{cfg["lift_off_mm"]}mm；方向映射尚未标定</text>',
        f'<circle cx="{cx}" cy="{cy}" r="{rope}" fill="#eef6f8" stroke="#486774" stroke-width="2"/>',
        f'<circle cx="{cx}" cy="{cy}" r="{sensor}" fill="none" stroke="#678f98" stroke-dasharray="5 5"/>',
        f'<line x1="{cx-rope}" y1="{cy}" x2="{cx+rope}" y2="{cy}" stroke="#cad9df"/><line x1="{cx}" y1="{cy-rope}" x2="{cx}" y2="{cy+rope}" stroke="#cad9df"/>']
    sign=1 if cfg['direction']=='counterclockwise' else -1
    axial=data['axis_profiles_robust_peak_to_peak']['XYZ'.index(data['axial_axis'])];maximum=max(axial) or 1
    for i,p in enumerate(axial):
        angle=cfg['channel1_angle_deg']+sign*i*11.25;x,y=point(angle,sensor)
        intensity=min(1,p/maximum);color=f'rgb({int(80+165*intensity)},{int(170-100*intensity)},{int(180-110*intensity)})'
        svg.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" fill="{color}"/>')
        if i%4==0 or i+1==data['peak_channel_1based']:
            tx,ty=point(angle,sensor+20);svg.append(f'<text x="{tx:.2f}" y="{ty+5:.2f}" font-size="13" text-anchor="middle">S{i+1}</text>')
    angle=data['assumed_angle_deg']
    depth=data.get('depth_model',{})
    if depth.get('accepted'):
        px,py=depth['model_xy_mm'];scale=rope/(cfg['rope_diameter_mm']/2);x=cx+px*scale;y=cy-py*scale
        svg.append(f'<path d="M {x:.2f} {y-7:.2f} L {x-7:.2f} {y+6:.2f} L {x+7:.2f} {y+6:.2f} Z" fill="#773dab"/><text x="24" y="510" font-size="15" fill="#773dab">紫色三角：假设模型位置；中心埋深{depth["model_depth_mm"]:.2f}mm（未标定）</text>')
    if angle is not None:
        width=min(359.9,data['response_spread_deg']);start=angle-width/2;end=angle+width/2
        x1,y1=point(start,rope);x2,y2=point(end,rope)
        svg.append(f'<path d="M {cx} {cy} L {x1:.2f} {y1:.2f} A {rope} {rope} 0 {int(width>180)} 0 {x2:.2f} {y2:.2f} Z" fill="#ee9341" opacity=".14"/>')
        x,y=point(angle,sensor+6);svg.append(f'<line x1="{cx}" y1="{cy}" x2="{x:.2f}" y2="{y:.2f}" stroke="#cd503d" stroke-width="3"/>')
        for lobe in data['lobes'][1:]:
            x,y=point(lobe['assumed_angle_deg'],rope);svg.append(f'<line x1="{cx}" y1="{cy}" x2="{x:.2f}" y2="{y:.2f}" stroke="#cc8650" stroke-width="2" stroke-dasharray="5 4"/>')
    svg.append('<text x="24" y="541" font-size="17">红线：主响应方向；浅色扇区：环周响应扩散范围</text><text x="24" y="572" font-size="17">真实位置未标定；模型点不代表已确认的实物缺陷</text>')
    svg.append('<text x="565" y="105" font-size="19">各通道稳健峰峰值（保持通道相对幅值）</text>')
    colors=['#bd4939','#168c88','#426ab6'];allvalues=data['axis_profiles_robust_peak_to_peak'];maximum=max(max(v) for v in allvalues) or 1
    for k,values in enumerate(allvalues):
        points=' '.join(f'{570+i/31*515:.2f},{405-v/maximum*245:.2f}' for i,v in enumerate(values))
        svg.append(f'<polyline points="{points}" fill="none" stroke="{colors[k]}" stroke-width="2"/><text x="{605+k*135}" y="450" font-size="16" fill="{colors[k]}">{"XYZ"[k]}{"（拟合输入）" if "XYZ"[k]==data["axial_axis"] else "（参考）"}</text>')
    svg.append('<line x1="570" y1="405" x2="1085" y2="405" stroke="#a9bcc4"/><text x="565" y="428" font-size="14">S1</text><text x="1048" y="428" font-size="14">S32</text>')
    for j in range(4):
        v=maximum*j/3;y=405-v/maximum*245;svg.append(f'<text x="556" y="{y+4:.2f}" text-anchor="end" font-size="12">{v:.3g}</text>')
    label='无法给出方向' if angle is None else f'S{data["peak_channel_1based"]}主峰 / 示意方位{angle:.2f}° / 半峰响应宽{data["response_spread_deg"]:.2f}°'
    svg.append(f'<text x="565" y="490" font-size="17">{e(label)}</text><text x="565" y="520" font-size="15">{e(data["interpretation"])}</text><text x="24" y="623" font-size="14">角度均相对假设编号；每通道间隔11.25°，插值网格更细不代表定位精度提高。</text></g></svg>')
    return ''.join(svg)

def attach_cross_sections(result,record,axis=None):
    import core
    from progress_events import notify
    if axis is not None and axis not in ('X','Y','Z'):raise ValueError('定位试验分量无效')
    if 'cross_sections' in result and (axis is None or result['cross_sections']['geometry']['axial_axis']==axis):return
    candidates=result.get('candidates',[])+result.get('supplementary_candidates',[])
    raw,*_=core.load(record);residual=raw-core.smooth(raw,201);cfg=geometry()
    if axis is not None and axis!=cfg['axial_axis']:
        cfg.update(axial_axis=axis,axial_axis_source='trial_unconfirmed')
        cfg['note']=f"{axis}为本次定位试验输入，未据拟合效果认定其物理轴向；直径{cfg['rope_diameter_mm']}mm、提离{cfg['lift_off_mm']}mm及编号方向均为示意假设。"
    folder=core.RUNS/record;folder.mkdir(exist_ok=True);prefix='cross_sections_'+secrets.token_hex(6)
    notify('reporting',message='正在计算各候选的轴向环周响应和截面示意')
    localized=[]
    for c in candidates:
        data=localize(c,residual,cfg,candidates);name=prefix+'_'+c['id']+'.svg'
        data['image']='/runs/'+record+'/'+name;(folder/name).write_text(render_svg(data),encoding='utf-8')
        if data['depth_model'].get('signed_observation'):
            fitname=prefix+'_'+c['id']+'_model.svg';data['model_fit_image']='/runs/'+record+'/'+fitname
            (folder/fitname).write_text(render_fit_svg(data),encoding='utf-8')
        c['cross_section']=data;localized.append(data)
    name=prefix+'.html';url='/runs/'+record+'/'+name
    result['cross_sections']=dict(geometry=cfg,candidates=localized,html_url=url,calibrated=False,depth_available=False,
        conditional_depth_count=sum(c['depth_model']['accepted'] for c in localized),depth_model_version='center-dipole-v1',
        method='环周方位分析与有符号中心截面偶极子条件深度拟合')
    (folder/name).write_text(explorer_html(result['cross_sections']),encoding='utf-8')
    (folder/(prefix+'.json')).write_text(json.dumps(result['cross_sections'],ensure_ascii=False,indent=2),encoding='utf-8')

def explorer_html(data):
    e=lambda x:html.escape(str(x),quote=True)
    options=''.join(f'<option value="{i}">{e(c["candidate_id"])} · {c["time_range_s"][0]}–{c["time_range_s"][1]}秒</option>' for i,c in enumerate(data['candidates']))
    cards=[]
    for i,c in enumerate(data['candidates']):
        angle=c['assumed_angle_deg'];direction='无法给出方向' if angle is None else f'主响应通道S{c["peak_channel_1based"]}；示意方位{angle:.2f}°'
        depth=c.get('depth_model',{});summary='模型深度估计被拒绝：'+'；'.join(depth.get('reasons',[]))
        if depth.get('accepted'):summary=f"假设模型中心埋深{depth['model_depth_mm']:.2f}mm，方位{depth['model_angle_deg']:.2f}°；可相容深度范围{depth['compatible_depth_range_mm']}mm；相对残差{depth['relative_residual']:.3f}（未经实测标定）。"
        fitimage='<img alt="有符号观测与模型拟合对照" src="'+e(c['model_fit_image'])+'">' if c.get('model_fit_image') else ''
        cards.append(f'<section class="candidate" data-index="{i}"'+(' hidden' if i else '')+f'><h2>{e(c["candidate_id"])} · {e(direction)}</h2><p>采样点{c["sample_range"]}；真实深度与截面坐标：未标定。</p><p>{e(summary)}</p><img alt="候选{e(c["candidate_id"])}的截面响应示意" src="{e(c["image"])}">'+fitimage+'<details><summary>方法、参数与复核疑点</summary><ul>'+''.join('<li>'+e(s)+'</li>' for s in c['notes'])+'</ul><pre>'+e(json.dumps(c,ensure_ascii=False,indent=2))+'</pre></details></section>')
    return '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>候选截面方位展示</title><style>body{font:16px/1.75 system-ui;color:#243746;max-width:1200px;padding:24px;margin:auto;background:#f5f8fa}section{background:white;padding:20px;margin:20px 0;border-radius:14px}img{width:100%;height:auto}select{padding:10px;font:inherit}pre{white-space:pre-wrap;overflow-wrap:anywhere}.notice{background:#fff1d9;padding:16px;border-radius:12px}[hidden]{display:none}</style><h1>候选截面方位与条件深度估计</h1><a href="/">返回智能体</a><p class="notice">'+e(data['geometry']['note'])+'<br>紫色模型点仅在拟合与可辨识性检查通过时显示；真实位置未标定，不估计断丝数。</p><label for="candidate">选择候选：</label><select id="candidate">'+options+'</select>'+(''.join(cards) if cards else '<p>当前没有候选。</p>')+'<script>document.getElementById("candidate").onchange=function(){for(const section of document.querySelectorAll(".candidate")){section.hidden=section.dataset.index!==this.value}};</script></html>'

def render_fit_svg(data):
    fit=data['depth_model'];y=fit['signed_observation'];p=fit['predicted_profile'];low=min(y+p);high=max(y+p)
    span=max(high-low,1e-8);low-=.1*span;high+=.1*span
    def coords(v):return ' '.join(f'{90+i/31*950:.2f},{260-(value-low)/(high-low)*180:.2f}' for i,value in enumerate(v))
    status='条件估计通过，未经实测标定' if fit['accepted'] else '模型拒绝；曲线仅作失配诊断，不发布深度'
    return '<svg xmlns="http://www.w3.org/2000/svg" width="1150" height="345"><rect width="1150" height="345" fill="white"/><g font-family="Microsoft YaHei,Arial" fill="#243746"><text x="24" y="30" font-size="20">'+html.escape(data['candidate_id'])+' · 有符号中心截面观测与模型拟合</text><text x="24" y="58" font-size="15">'+status+f'；相对残差{fit["relative_residual"]:.3f}；中心采样点{fit["center_sample"]}</text><polyline points="'+coords(y)+'" fill="none" stroke="#168c88" stroke-width="2"/><polyline points="'+coords(p)+'" fill="none" stroke="#773dab" stroke-width="2" stroke-dasharray="6 4"/><text x="90" y="290" font-size="15">S1</text><text x="1010" y="290" font-size="15">S32</text><text x="360" y="320" font-size="16" fill="#168c88">绿色：去背景后的有符号观测</text><text x="720" y="320" font-size="16" fill="#773dab">紫色虚线：最佳模型曲线</text>'+''.join(f'<text x="78" y="{260-j/3*180+4:.2f}" text-anchor="end" font-size="13">{low+(high-low)*j/3:.3g}</text>' for j in range(4))+'</g></svg>'
