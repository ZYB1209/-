"""Per-user Windows DPAPI storage; secrets are never returned to the browser."""
import base64,ctypes,json,os
from ctypes import wintypes
from pathlib import Path
from urllib.parse import urlparse

FILE=Path(__file__).parent/'.local_api.json'
class Blob(ctypes.Structure):
    _fields_=[('size',wintypes.DWORD),('data',ctypes.POINTER(ctypes.c_ubyte))]

def crypt(data,decrypt=False):
    if os.name!='nt': raise ValueError('本版本密钥持久化使用 Windows 加密；其他系统请使用 MFL_API_KEY 环境变量。')
    buffer=ctypes.create_string_buffer(data); source=Blob(len(data),ctypes.cast(buffer,ctypes.POINTER(ctypes.c_ubyte))); out=Blob()
    lib=ctypes.windll.crypt32
    if decrypt: ok=lib.CryptUnprotectData(ctypes.byref(source),None,None,None,None,1,ctypes.byref(out))
    else: ok=lib.CryptProtectData(ctypes.byref(source),'MFL Agent API key',None,None,None,1,ctypes.byref(out))
    if not ok: raise ValueError('Windows 密钥加密/解密失败，请在当前用户下重新配置。')
    try: return ctypes.string_at(out.data,out.size)
    finally: ctypes.windll.kernel32.LocalFree(out.data)

def validate(base,model):
    base=str(base).strip().rstrip('/'); model=str(model).strip()
    u=urlparse(base)
    if not u.hostname or u.username or u.password or u.query or u.fragment:
        raise ValueError('Base URL 必须是有效服务地址，不能包含密钥或查询参数。')
    if u.scheme!='https' and not (u.scheme=='http' and u.hostname in ('127.0.0.1','localhost')):
        raise ValueError('云端 API 地址必须使用 HTTPS。')
    if not model or len(model)>200: raise ValueError('请填写服务提供的模型名称。')
    return base,model

def load_config():
    doc=json.loads(FILE.read_text(encoding='utf-8')) if FILE.exists() else {}
    key=os.getenv('MFL_API_KEY','') or os.getenv('OPENAI_API_KEY','')
    if not key and doc.get('key_encrypted'):
        key=crypt(base64.b64decode(doc['key_encrypted']),True).decode()
    return dict(base_url=os.getenv('MFL_API_BASE') or doc.get('base_url','https://api.deepseek.com'),
                model=os.getenv('MFL_MODEL') or doc.get('model','deepseek-flash'),api_key=key)

def public_config():
    c=load_config()
    return dict(base_url=c['base_url'],model=c['model'],configured=bool(c['base_url'] and c['model'] and c['api_key']),has_key=bool(c['api_key']))

def save_config(base_url,model,api_key=''):
    base,model=validate(base_url,model)
    existing=load_config()
    key=str(api_key).strip() or (existing['api_key'] if base==existing['base_url'] else '')
    if not key: raise ValueError('请填写该服务的 API Key。')
    if '\n' in key or '\r' in key: raise ValueError('密钥不能包含换行。')
    doc=dict(base_url=base,model=model,key_encrypted=base64.b64encode(crypt(key.encode())).decode())
    temp=FILE.with_suffix('.tmp');temp.write_text(json.dumps(doc,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(FILE)
    return public_config()
