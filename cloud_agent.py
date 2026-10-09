"""OpenAI-compatible Chat Completions transport, with tool-call IDs preserved."""
import json,re,time
from urllib.request import Request,urlopen
from urllib.error import HTTPError,URLError
from urllib.parse import urlparse
from ollama_agent import OllamaAgent
from api_config import load_config,validate

def chat_messages(messages):
    out=[]; pending=[]
    for mi,msg in enumerate(messages):
        if msg['role']=='tool':
            if not pending: raise ValueError('工具响应缺少对应调用')
            cid,name=pending.pop(0)
            if name!=msg['tool_name']: raise ValueError('工具响应顺序不一致')
            out.append(dict(role='tool',tool_call_id=cid,content=msg['content']))
        elif msg['role']=='assistant':
            m=dict(role='assistant',content=msg.get('content') or None)
            if msg.get('tool_calls'):
                calls=[]
                for ci,c in enumerate(msg['tool_calls']):
                    f=c['function'];cid=c.get('id') or f'call_{mi}_{ci}'
                    args=f.get('arguments',{})
                    calls.append(dict(id=cid,type='function',function=dict(name=f['name'],arguments=args if isinstance(args,str) else json.dumps(args))))
                    pending.append((cid,f['name']))
                m['tool_calls']=calls
            out.append(m)
        else: out.append(dict(role=msg['role'],content=msg['content']))
    if pending: raise ValueError('部分工具调用没有响应')
    return out

class CloudAgent(OllamaAgent):
    def __init__(self,config=None,transport=None):
        self.config=config or load_config()
        super().__init__(self.config.get('model'),transport)
        self.planner_label='云端 API 工具调度：'
    def ask(self,record,question,mode='fast'):
        if not all(self.config.get(k) for k in ('base_url','model','api_key')):
            raise ValueError('请先在左侧填写并保存 Base URL、模型名称和 API Key，再测试云端模型。')
        return super().ask(record,question,mode)
    def request(self,messages,tools):
        c=self.config
        if not c.get('api_key'): raise ValueError('尚未配置 API Key，请先保存 API 设置。')
        base,model=validate(c['base_url'],c['model'])
        url=base if base.endswith('/chat/completions') else base+'/chat/completions'
        payload=dict(model=model,messages=chat_messages(messages),tools=tools,stream=False,max_tokens=1800)
        if urlparse(base).hostname=='api.deepseek.com':
            # DeepSeek thinking mode rejects tool_choice=required; use explicit non-thinking tool routing.
            payload['thinking']={'type':'disabled'}
        # First turn must use a tool; subsequent turns may finish or call another.
        payload['tool_choice']='auto' if any(m['role']=='tool' for m in messages) else 'required'
        req=Request(url,data=json.dumps(payload,ensure_ascii=False).encode(),headers={'Content-Type':'application/json','Authorization':'Bearer '+c['api_key']})
        try:
            for attempt in range(3):
                try:
                    with urlopen(req,timeout=90) as response:data=json.load(response)
                    break
                except HTTPError:raise
                except (URLError,TimeoutError):
                    if attempt==2:raise
                    time.sleep(.5)
            msg=data['choices'][0]['message']
            if not isinstance(msg,dict): raise ValueError('响应格式不是 Chat Completions')
            return msg
        except HTTPError as e:
            reasons={401:'密钥无效或已过期',403:'账号没有该模型权限',404:'服务地址或模型名称不正确',429:'余额不足或请求受限'}
            # Never return provider response bodies: some proxies echo credentials/requests.
            raise ValueError(f"API 请求失败（HTTP {e.code}）：{reasons.get(e.code,'请检查服务状态与工具调用支持')}。") from None
        except (URLError,TimeoutError): raise ValueError('无法连接模型服务或请求超时，请检查网络与 API 地址。') from None
        except (KeyError,IndexError,json.JSONDecodeError): raise ValueError('服务响应格式不兼容，请确认支持 Chat Completions 和工具调用。') from None
