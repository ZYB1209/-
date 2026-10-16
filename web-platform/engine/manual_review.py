"""Traceable 500x500 candidate crops and aligned, native-resolution review signals."""
import html,json,secrets
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw

def bounds(candidate,n):
    a,b=candidate['refined_start'],candidate['refined_end']
    if type(a)is not int or type(b)is not int or not 0<=a<=b<n:raise ValueError('复检候选范围无效')
    margin=max(256,2*(b-a+1));return max(0,a-margin),min(n-1,b+margin)

def make_gray(matrix,clip):
    # Input samples x channels; image x=time, y=channel. Zero is middle gray.
    pixels=np.rint(127.5*(1+np.clip(np.asarray(matrix,float).T/clip,-1,1))).astype('uint8')
    return Image.fromarray(pixels).resize((500,500),Image.Resampling.NEAREST)

def band_pixels(a,b,lo,hi):
    # Cells, not just interval endpoints: inclusive samples occupy their complete image cells.
    length=hi-lo+1
    return max(0,int(np.floor((a-lo)/length*500))),min(499,int(np.ceil((b+1-lo)/length*500))-1)

def annotate(image,candidate,neighbors,lo,hi,peak_channel):
    marked=image.convert('RGB');draw=ImageDraw.Draw(marked)
    for c in neighbors:
        if c['id']==candidate['id']:continue
        a,b=band_pixels(max(lo,c['refined_start']),min(hi,c['refined_end']),lo,hi)
        draw.rectangle((a,1,b,498),outline='#7693a0',width=1)
    a,b=band_pixels(candidate['refined_start'],candidate['refined_end'],lo,hi)
    draw.rectangle((a,1,b,498),outline='#e33c37',width=3)
    cy=(peak_channel-.5)*500/32;cx=(a+b)/2
    draw.ellipse((cx-7,cy-7,cx+7,cy+7),outline='#ffc226',width=3)
    return marked

def attach_manual_review(result,record):
    import core
    from progress_events import notify
    if 'manual_review' in result:return
    raw,*_=core.load(record);n=len(raw);candidates=result.get('candidates',[])+result.get('supplementary_candidates',[])
    folder=core.RUNS/record;folder.mkdir(exist_ok=True);prefix='manual_review_'+secrets.token_hex(6)
    processed=None;input_signal=None;score=None;reference=None;artifact=result.get('signal_artifact')
    if artifact:
        path=(core.ROOT/artifact.lstrip('/')).resolve()
        if not path.is_relative_to(core.RUNS.resolve()):raise ValueError('复检处理证据路径无效')
        if path.is_file():
            with np.load(path,allow_pickle=False) as arrays:
                if 'envelope' in arrays:
                    value=arrays['envelope']
                    if value.shape!=(3,n,32) or not np.isfinite(value).all():raise ValueError('已保存包络形状或数值无效')
                    processed=value.transpose(1,0,2).copy()
                if 'axis_scale' in arrays:
                    scales=arrays['axis_scale']
                    if scales.shape==(3,) and np.isfinite(scales).all() and np.all(scales>0):
                        input_signal=(raw-np.median(raw[:min(64,n)],axis=0)[None,:,:])/scales[None,:,None]
                for key in ('score','reference_score'):
                    if key in arrays:
                        v=arrays[key]
                        if v.shape==(n,) and np.isfinite(v).all():
                            if key=='score':score=v.copy()
                            else:reference=v.copy()
    notify('reporting',message='正在生成500×500候选定位灰度图和对齐波形复检证据')
    items=[]
    for c in candidates:
        lo,hi=bounds(c,n);native=raw[lo:hi+1];centered=native-np.median(native,axis=0,keepdims=True)
        neighbors=[x for x in candidates if x['refined_start']<=hi and x['refined_end']>=lo]
        data=dict(candidate_id=c['id'],crop_samples=[lo,hi],selected_samples=[c['refined_start'],c['refined_end']],
            sample_indices=list(range(lo,hi+1)),time_s=np.round(np.arange(lo,hi+1)/500,6).tolist(),
            raw=np.round(native,8).tolist(),input_signal=None if input_signal is None else np.round(input_signal[lo:hi+1],8).tolist(),
            processed=None if processed is None else np.round(processed[lo:hi+1],8).tolist(),
            adaptive_score=None if score is None else np.round(score[lo:hi+1],8).tolist(),
            reference_score=None if reference is None else np.round(reference[lo:hi+1],8).tolist(),
            thresholds=result.get('thresholds'),neighbors=[dict(id=x['id'],samples=[x['refined_start'],x['refined_end']]) for x in neighbors])
        axes=[]
        for ai,axis in enumerate('XYZ'):
            peak=int(np.argmax(np.ptp(raw[c['refined_start']:c['refined_end']+1,ai,:],axis=0)))+1
            raw_clip=max(float(np.quantile(np.abs(centered[:,ai,:]),.995)),1e-9)
            states=[('raw',centered[:,ai,:],raw_clip,'原始磁通道（仅显示减去局部通道中位数）')]
            if input_signal is not None and processed is not None:
                clip=max(float(np.quantile(np.abs(input_signal[lo:hi+1,ai]),.995)),float(np.quantile(processed[lo:hi+1,ai],.995)),1e-9)
                states += [('input',input_signal[lo:hi+1,ai],clip,'检测输入：原始信号减首64点通道中位数，再除以该次轴标度'),
                    ('processed',processed[lo:hi+1,ai],clip,'该次实际处理后的Hilbert包络/融合包络；正值，不是原始有符号波形')]
            elif processed is not None:
                clip=max(float(np.quantile(processed[lo:hi+1,ai],.995)),1e-9)
                states.append(('processed',processed[lo:hi+1,ai],clip,'该次已保存的处理包络；未保存输入标度，前后图不能直接比较幅值'))
            images=[]
            for stage,matrix,clip,label in states:
                name=prefix+'_'+c['id']+'_'+axis+'_'+stage+'.png';plain=make_gray(matrix,clip)
                plain.save(folder/name);marked=name.replace('.png','_marked.png');annotate(plain,c,neighbors,lo,hi,peak).save(folder/marked)
                images.append(dict(stage=stage,label=label,gray_image='/runs/'+record+'/'+name,
                    marked_image='/runs/'+record+'/'+marked,clip=[-clip,clip],image_size=[500,500]))
            axes.append(dict(axis=axis,peak_channel_1based=peak,images=images))
        final_segments=[dict(segment_id=s['segment_id'],samples=s['sample_range'],status=s['status'],
            executed_pipeline=s['final_executed_pipeline'],fallback_reason=s.get('fallback_reason')) for s in result.get('agentic_segments',[]) if s['sample_range'][0]<=hi and s['sample_range'][1]>=lo]
        data['executed_segments']=final_segments
        name=prefix+'_'+c['id']+'.json';(folder/name).write_text(json.dumps(data,ensure_ascii=False),encoding='utf-8')
        item=dict(candidate_id=c['id'],selected_samples=data['selected_samples'],crop_samples=[lo,hi],time_range_s=[c['start_s'],c['end_s']],
            supplementary=c['id'] not in {x['id'] for x in result.get('candidates',[])},axes=axes,data_url='/runs/'+record+'/'+name,
            pixel_mapping=dict(x='floor(pixel_x*(crop_end-crop_start+1)/500)+crop_start',y='floor(pixel_y*32/500)+1',selected_band_pixels=band_pixels(c['refined_start'],c['refined_end'],lo,hi)),
            signal_artifact=artifact,run_id=result.get('run_id'))
        c['manual_review']=dict(data_url=item['data_url'],crop_samples=[lo,hi]);items.append(item)
    index=dict(record=record,run_id=result.get('run_id'),candidates=items,processing_available=processed is not None,
        method='500×500时间—环周通道灰度图，原始采样点波形与实际执行结果对齐',annotations_used=False)
    name=prefix+'.html';index['html_url']='/runs/'+record+'/'+name
    result['manual_review']=index;(folder/name).write_text(review_html(index),encoding='utf-8')
    (folder/(prefix+'_index.json')).write_text(json.dumps(index,ensure_ascii=False,indent=2),encoding='utf-8')

def review_html(index):
    payload=json.dumps(index,ensure_ascii=False).replace('<','\\u003c')
    script=r'''
const index=__INDEX__,el=id=>document.getElementById(id);let data=null,request=0;
for(const [i,c] of index.candidates.entries()){const o=document.createElement('option');o.value=i;o.textContent=c.candidate_id+' · '+c.time_range_s.join('–')+'秒'+(c.supplementary?'（补充响应）':'');el('candidate').append(o)}
for(let i=1;i<=32;i++){const o=document.createElement('option');o.value=i;o.textContent='S'+i;el('channel').append(o)}
function node(tag,text){const n=document.createElement(tag);if(text!==undefined)n.textContent=text;return n}
function plot(container,values,label,threshold=null){
 container.replaceChildren();container.append(node('h3',label));if(!values){container.append(node('p','该次未保存此信号，未重跑或伪造处理结果。'));return}
 const W=1040,H=240,x=i=>65+i/(values.length-1)*940;let lo=Math.min(...values),hi=Math.max(...values);if(threshold!==null){lo=Math.min(lo,threshold);hi=Math.max(hi,threshold)}let range=hi-lo||1;lo-=range*.1;hi+=range*.1;const y=v=>190-(v-lo)/(hi-lo)*150;
 const ns='http://www.w3.org/2000/svg',svg=document.createElementNS(ns,'svg');svg.setAttribute('viewBox',`0 0 ${W} ${H}`);svg.setAttribute('aria-label',label);svg.setAttribute('role','img');
 function add(tag,attrs,text){const n=document.createElementNS(ns,tag);for(const [k,v]of Object.entries(attrs))n.setAttribute(k,v);if(text!==undefined)n.textContent=text;svg.append(n)}
 const c=index.candidates[Number(el('candidate').value)],a=c.selected_samples[0]-c.crop_samples[0],b=c.selected_samples[1]-c.crop_samples[0];add('rect',{x:x(a),y:30,width:Math.max(2,x(b)-x(a)),height:165,fill:'#f5b59b',opacity:.35});
 for(let j=0;j<4;j++){const v=lo+(hi-lo)*j/3;add('text',{x:58,y:y(v)+4,'text-anchor':'end','font-size':12},v.toPrecision(4))}
 add('polyline',{points:values.map((v,i)=>x(i).toFixed(2)+','+y(v).toFixed(2)).join(' '),fill:'none',stroke:'#167f83','stroke-width':1.6});
 if(threshold!==null)add('line',{x1:65,x2:1005,y1:y(threshold),y2:y(threshold),stroke:'#c74c40','stroke-dasharray':'5 4'});
 for(let j=0;j<5;j++){const i=Math.round((values.length-1)*j/4);add('text',{x:x(i),y:220,'text-anchor':'middle','font-size':12},data.sample_indices[i]+' / '+data.time_s[i].toFixed(3)+'s')}
 container.append(svg);
}
function display(){
 if(!data)return;const c=index.candidates[Number(el('candidate').value)],ai=Number(el('axis').value),ch=Number(el('channel').value)-1,axis=c.axes[ai];
 el('summary').textContent=c.candidate_id+' · 预测采样点'+c.selected_samples.join('–')+'；上下文'+c.crop_samples.join('–')+'；横轴时间/采样点，纵轴S1–S32。红框仅为预测时间区间，不是缺陷像素轮廓；黄圈为该轴最强响应通道。';
 el('images').replaceChildren();for(const im of axis.images){const box=node('div');box.append(node('h3',im.label));const image=node('img');image.src=im.marked_image;image.alt=im.label+'预测区间图';image.width=500;image.height=500;box.append(image);const a=node('a','打开500×500标记图');a.href=im.marked_image;a.target='_blank';const b=node('a','保存无标记灰度图');b.href=im.gray_image;b.target='_blank';box.append(a,node('br'),b,node('p','显示范围 '+im.clip.map(v=>Number(v).toPrecision(4)).join('–')+'；0为中灰，前后输入与包络采用同一范围。'));el('images').append(box)}
 plot(el('rawPlot'),data.raw.map(row=>row[ai][ch]),'原始'+axis.axis+' / S'+(ch+1)+'波形（CSV原始单位，保留直流与符号）');
 plot(el('inputPlot'),data.input_signal?.map(row=>row[ai][ch]),'检测输入波形（该次轴标度单位）');
 plot(el('processedPlot'),data.processed?.map(row=>row[ai][ch]),'实际处理后包络（该次轴标度单位，与原始有符号信号含义不同）');
 plot(el('referencePlot'),data.reference_score,'参考检测响应曲线',(typeof data.thresholds?.reference==='number'?data.thresholds.reference:data.thresholds?.reference?.high)??null);
 plot(el('adaptivePlot'),data.adaptive_score,'自适应检测响应曲线',typeof data.thresholds?.envelope==='number'?data.thresholds.envelope:null);
 el('details').textContent=JSON.stringify({record:index.record,run_id:index.run_id,artifact:c.signal_artifact,selected:data.selected_samples,crop:data.crop_samples,thresholds:data.thresholds,executed_segments:data.executed_segments,pixel_mapping:c.pixel_mapping},null,2);
}
async function select(){const token=++request;data=null;el('images').replaceChildren();for(const id of ['rawPlot','inputPlot','processedPlot','referencePlot','adaptivePlot'])el(id).replaceChildren();el('status').textContent='读取已保存复检信号…';try{const c=index.candidates[Number(el('candidate').value)],r=await fetch(c.data_url);if(!r.ok)throw Error('读取失败');const d=await r.json();if(token!==request)return;data=d;el('channel').value=c.axes[Number(el('axis').value)].peak_channel_1based;display();el('status').textContent='已读取该次检测记录；没有调用模型。'}catch(e){el('status').textContent=e.message}}
el('candidate').onchange=select;el('axis').onchange=()=>{const c=index.candidates[Number(el('candidate').value)];el('channel').value=c.axes[Number(el('axis').value)].peak_channel_1based;display()};el('channel').onchange=display;
el('csv').onclick=()=>{if(!data)return;const ai=Number(el('axis').value),ch=Number(el('channel').value)-1;const rows=['sample_index,time_s,raw_csv_value,detector_input,processed_envelope,in_selected_interval'];for(let i=0;i<data.sample_indices.length;i++){const s=data.sample_indices[i];rows.push([s,data.time_s[i],data.raw[i][ai][ch],data.input_signal?.[i][ai][ch]??'',data.processed?.[i][ai][ch]??'',Number(s>=data.selected_samples[0]&&s<=data.selected_samples[1])].join(','))}const url=URL.createObjectURL(new Blob(['\ufeff'+rows.join('\r\n')],{type:'text/csv;charset=utf-8'})),a=document.createElement('a');a.href=url;a.download=index.record+'_'+data.candidate_id+'_'+'XYZ'[ai]+'_S'+(ch+1)+'.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};
if(index.candidates.length)select();else el('status').textContent='当前没有候选。';
'''.replace('__INDEX__',payload)
    return_url='/?result='+index['history_id'] if index.get('history_id') else '/'
    return '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>候选人工复检工作台</title><style>body{font:16px/1.7 system-ui;max-width:1600px;margin:auto;padding:24px;color:#243746;background:#f3f7f9}section{background:white;border-radius:14px;padding:20px;margin:20px 0}select,button{font:inherit;padding:9px;margin:8px}#images{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px}#images>div{max-width:500px}#images h3{min-height:100px}@media(max-width:900px){#images{grid-template-columns:1fr}#images h3{min-height:0}}img{max-width:100%;height:auto;border:1px solid #ccd}svg{width:100%;max-height:250px}pre{white-space:pre-wrap;overflow-wrap:anywhere}.notice{background:#fff1d9;padding:14px}a{margin-right:10px}</style><h1>'+html.escape(index['record'])+' · 候选人工复检工作台</h1><a href="'+html.escape(return_url,quote=True)+'">返回智能体（恢复检测结果）</a><p class="notice">预测区间为漏磁响应范围，未确认实物缺陷。灰度图只用于显示，图像缩放不改变原始采样点；原始波形、检测输入与实际处理包络分别标注单位和含义。</p><label>候选<select id="candidate"></select></label><label>分量<select id="axis"><option value="0">X</option><option value="1" selected>Y</option><option value="2">Z</option></select></label><label>通道<select id="channel"></select></label><button id="csv">导出当前通道原始/处理波形CSV</button><p id="status" role="status"></p><p id="summary"></p><section><h2>500×500灰度定位图</h2><div id="images"></div></section><section><h2>同采样位置对齐的波形 · 阴影为预测区间</h2><div id="rawPlot"></div><div id="inputPlot"></div><div id="processedPlot"></div><div id="referencePlot"></div><div id="adaptivePlot"></div></section><details><summary>实际处理方法、参数、阈值与坐标映射</summary><pre id="details"></pre></details><script>'+script+'</script></html>'
