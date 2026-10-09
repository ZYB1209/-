"""Canonical matrix rendering, independent VLM transport and schema validation."""
import base64,hashlib,json,os,time
from pathlib import Path
from urllib.request import Request,urlopen
from urllib.error import HTTPError,URLError
import numpy as np
from PIL import Image,ImageDraw
from api_config import crypt,validate
from adaptive_signal import mean

FILE=Path(__file__).parent/'.local_vlm.json'
VERSION='mfl-visual-v2'
SCHEMA={
 'periodic_strand_pattern':{'level':['absent','weak','moderate','strong'],'orientation':['positive','negative','mixed','unclear']},
 'circumferential_background':{'level':['weak','moderate','strong'],'extent':['localized','medium','widespread']},
 'local_response':{'presence':['absent','possible','clear'],'extent':['narrow','medium','broad'],'multiplicity':['single','multiple','unclear']},
 'background_local_coexistence':{'level':['low','moderate','high']},
 'cross_axis_consistency':{'level':['low','moderate','high']},
 'overall_condition':{'label':['clean','strand_dominant','broad_background','mixed','ambiguous']}}

def config():
    doc=json.loads(FILE.read_text('utf-8')) if FILE.exists() else {}
    key=os.getenv('MFL_VLM_KEY','')
    if not key and doc.get('key_encrypted'):key=crypt(base64.b64decode(doc['key_encrypted']),True).decode()
    return dict(base_url=os.getenv('MFL_VLM_BASE') or doc.get('base_url',''),model=os.getenv('MFL_VLM_MODEL') or doc.get('model',''),api_key=key)

def public_config():
    c=config();return dict(base_url=c['base_url'],model=c['model'],configured=all(c.values()),has_key=bool(c['api_key']))

def save_config(base_url,model,api_key=''):
    base,model=validate(base_url,model);old=config();key=str(api_key).strip() or (old['api_key'] if base==old['base_url'] else '')
    if not key or '\n' in key or '\r' in key:raise ValueError('请填写合法视觉模型密钥')
    temp=FILE.with_suffix('.tmp');temp.write_text(json.dumps(dict(base_url=base,model=model,key_encrypted=base64.b64encode(crypt(key.encode())).decode())),encoding='utf-8');temp.replace(FILE)
    return public_config()

def validate_observation(data):
    if not isinstance(data,dict) or set(data)!=set(SCHEMA)|{'confidence','evidence'}:raise ValueError('视觉输出字段不符合 schema')
    for group,fields in SCHEMA.items():
        if not isinstance(data[group],dict) or set(data[group])!=set(fields):raise ValueError('视觉输出分组字段不合法')
        for key,choices in fields.items():
            if data[group][key] not in choices:raise ValueError('视觉输出枚举不合法')
    confidence=data['confidence']
    if not isinstance(confidence,dict) or set(confidence)!={'value'} or type(confidence['value']) not in (int,float) or not 0<=confidence['value']<=1:raise ValueError('视觉置信度字段无效')
    evidence=data['evidence']
    if not isinstance(evidence,list) or not 1<=len(evidence)<=6 or any(not isinstance(s,str) or len(s)>300 for s in evidence):raise ValueError('视觉证据无效')
    return data

def render(z,path):
    residual=z-mean(z,min(201,z.shape[1]),axis=1)
    level=max(float(np.quantile(abs(residual),.995)),1e-6)
    img=Image.new('RGB',(1024,360),'#102030');draw=ImageDraw.Draw(img)
    for i,axis in enumerate('XYZ'):
        values=np.clip(residual[i].T/level,-1,1)
        rgb=np.stack([25+225*np.maximum(values,0),30+100*abs(values),40+210*np.maximum(-values,0)],axis=2).astype('uint8')
        img.paste(Image.fromarray(rgb).resize((960,96),Image.Resampling.BILINEAR),(56,i*120+18))
        draw.text((4,i*120+50),'B'+axis,fill='white')
    img.save(path)
    return dict(normalization='common tri-axis abs q.995 of MA201 residual',scale=level,
        colormap='fixed signed red/blue',resize='bilinear 960x96 per axis',axis='horizontal time; vertical channels 1-32',annotations_used=False)

def request(c,image_path):
    base,model=validate(c['base_url'],c['model'])
    example={group:{key:choices[0] for key,choices in fields.items()} for group,fields in SCHEMA.items()}
    example.update(confidence={'value':0.5},evidence=['用中文简述可观察的视觉证据'])
    instruction=('Describe visually observable MFL signal conditions ONLY. Do not recommend methods, infer speed, lift-off, mechanical shaking amplitude, defect counts or damage severity. '
        'BX/BY/BZ are separate axes, horizontal=time and vertical=channels. Return ONLY ONE JSON object. '
        'Exact structure example (values are placeholders, replace with your observations): '+json.dumps(example,ensure_ascii=False)+
        '. All seven top-level groups including overall_condition MUST appear; no additional fields. '
        'For each attribute choose ONE enum STRING, never return the enum list. Allowed values: '+json.dumps(SCHEMA)+
        '. confidence MUST be an object {"value":number}, not a scalar; range 0..1, self-reported and uncalibrated. '
        'evidence MUST be 1..6 short strings in Chinese. Use compact JSON, no blank lines. Close every brace including the final root brace.')
    payload=dict(model=model,messages=[dict(role='user',content=[dict(type='text',text=instruction),
        dict(type='image_url',image_url=dict(url='data:image/png;base64,'+base64.b64encode(image_path.read_bytes()).decode()))])],max_tokens=1200,stream=False,temperature=0,response_format={'type':'json_object'})
    req=Request(base if base.endswith('/chat/completions') else base+'/chat/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json','Authorization':'Bearer '+c['api_key']})
    try:
        for attempt in range(3):
            try:
                with urlopen(req,timeout=90) as response:data=json.load(response)
                break
            except HTTPError:raise
            except (URLError,TimeoutError):
                if attempt==2:raise
                time.sleep(.5)
        content=data['choices'][0]['message']['content'].strip()
        if content.startswith('```'):content=content.split('\n',1)[1].rsplit('```',1)[0].strip()
        observation,repair=parse_content(content)
        return dict(observation=observation,output_repair=repair)
    except HTTPError as e:raise ValueError(f'视觉 API HTTP {e.code}，检查模型图片输入能力与账号权限') from None
    except (URLError,TimeoutError):raise ValueError('视觉 API 连接失败或超时') from None
    except (KeyError,IndexError,TypeError,AttributeError,json.JSONDecodeError):raise ValueError('视觉 API 返回格式不兼容') from None

def inspect(record,item,z,folder,transport=None,use_cache=True):
    path=folder/(item['segment_id']+'_visual.png');meta=render(z,path);c=config()
    result=dict(segment_id=item['segment_id'],image_refs=['/runs/'+record+'/'+path.name],rendering=meta,
        prompt_version=VERSION,model=c['model'],confidence_kind='VLM self-report; not calibrated',annotations_used=False)
    if not all(c.values()) and transport is None:return dict(result,status='disabled',visual_condition=None,note='视觉模型尚未配置，未发送图片')
    digest=hashlib.sha256(path.read_bytes()+json.dumps([VERSION,c['base_url'],c['model']]).encode()).hexdigest()
    cache=folder/('visual_'+digest+'.json')
    if use_cache and cache.exists():return dict(json.loads(cache.read_text('utf-8')),cached=True)
    try:
        response=(transport or request)(c,path)
        observation=validate_observation(response if transport is not None else response['observation'])
        result.update(status='completed',visual_condition=observation,cached=False,
            output_repair=None if transport is not None else response['output_repair'])
        cache.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    except ValueError as e:result.update(status='failed',visual_condition=None,note=str(e))
    return result

def parse_content(content):
    try:return validate_observation(json.loads(content)),None
    except json.JSONDecodeError:
        # Only a single missing root closing brace may be repaired. No values are inferred.
        if content.lstrip().startswith('{') and not content.rstrip().endswith('}'):
            return validate_observation(json.loads(content+'}')),'single_missing_root_closing_brace'
        raise

def conflicts(numeric,visual):
    if visual['status']!='completed':return []
    v=visual['visual_condition'];notes=[]
    if v['periodic_strand_pattern']['level']=='strong' and numeric['strand_ridge_fraction']<.08:notes.append('视觉强斜纹与数值频谱脊偏低冲突')
    if v['overall_condition']['label']=='clean' and numeric['recommended_algorithm']!='baseline':notes.append('视觉干净与数值噪声线索冲突')
    if v['circumferential_background']['extent']=='widespread' and numeric['circumferential_coverage']<.4:notes.append('视觉宽背景与环周覆盖偏低冲突')
    return notes
