"""OpenAI-compatible Chat Completions transport, with tool-call IDs preserved."""
import json,os,re,time
from urllib.request import Request,urlopen
from urllib.error import HTTPError,URLError
from urllib.parse import urlparse
from ollama_agent import OllamaAgent
from api_config import load_config,validate
from progress_events import notify

class ModelCallBudgetExceeded(RuntimeError):
    pass

class ModelServiceAccessDenied(ValueError):
    """Authentication/access rejection; re-planning cannot repair it."""
    pass

def read_stream(response):
    """Collect SSE tool argument fragments; never log private reasoning text."""
    content=[];calls={};usage=None;finished=False;reasoning_chars=0;event=[];last_activity=0.
    def consume(lines):
        nonlocal usage,finished,reasoning_chars,last_activity
        data='\n'.join(lines)
        if data=='[DONE]':finished=True;return
        chunk=json.loads(data)
        if chunk.get('error'):raise ValueError('模型流式响应报告错误')
        if chunk.get('usage'):usage=chunk['usage']
        for choice in chunk.get('choices',[]):
            if choice.get('index',0)!=0:continue
            if choice.get('finish_reason') in ('length','content_filter'):
                raise ValueError('模型输出被截断或过滤，没有完整工具计划')
            if choice.get('finish_reason'):finished=True
            delta=choice.get('delta',{})
            reasoning_chars+=len(delta.get('reasoning_content') or '')
            if delta.get('content'):
                content.append(delta['content']);notify('model_text',text=delta['content'])
            if time.monotonic()-last_activity>=.5:
                notify('model_activity',phase='planning' if delta.get('tool_calls') else 'receiving',reasoning_chars=reasoning_chars)
                last_activity=time.monotonic()
            for part in delta.get('tool_calls',[]):
                index=part.get('index',0)
                call=calls.setdefault(index,dict(id='',type='function',function=dict(name='',arguments='')))
                if part.get('id'):call['id']=part['id']
                function=part.get('function',{})
                for field in ('name','arguments'):
                    call['function'][field]+=function.get(field) or ''
    for raw in response:
        line=raw.decode('utf-8').rstrip('\r\n')
        if line.startswith('data:'):event.append(line[5:].lstrip())
        elif not line and event:consume(event);event=[]
    if event:consume(event)
    if not finished:raise ValueError('模型流式连接提前结束，没有完成标记')
    message=dict(role='assistant',content=''.join(content))
    if calls:
        ordered=[calls[i] for i in sorted(calls)]
        if any(not c['function']['name'] or not c['function']['arguments'] for c in ordered):
            raise ValueError('模型返回不完整的工具调用')
        message['tool_calls']=ordered
    return message,usage,reasoning_chars

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
        self.dsp_planner_mode=os.getenv('MFL_PLANNER_MODE','separated')
        self.call_history=[];self.max_model_calls=None
        self.access_denied_http=None
        self.enable_thinking=os.getenv('MFL_ENABLE_THINKING','false').lower() in ('true','1','yes')
        if self.dsp_planner_mode not in ('separated','unified'):
            raise ValueError('MFL_PLANNER_MODE 必须为 separated 或 unified')
    def ask(self,record,question,mode='fast'):
        if not all(self.config.get(k) for k in ('base_url','model','api_key')):
            raise ValueError('请先在左侧填写并保存 Base URL、模型名称和 API Key，再测试云端模型。')
        return super().ask(record,question,mode)
    def perform_request(self,req,stream):
        if self.max_model_calls is not None and len(self.call_history)>=self.max_model_calls:
            raise ModelCallBudgetExceeded('模型调用预算已用完，已停止新请求；未完成记录不得作为完整评估')
        began=time.monotonic();entry=dict(status='running',model=self.model)
        self.call_history.append(entry);self.last_usage=None;self.last_reasoning_chars=0
        notify('model_start',model=self.model,request_index=len(self.call_history))
        try:
            with urlopen(req,timeout=90) as response:
                if stream:
                    msg,self.last_usage,self.last_reasoning_chars=read_stream(response)
                else:
                    data=json.load(response);msg=data['choices'][0]['message'];self.last_usage=data.get('usage')
            if not isinstance(msg,dict):raise ValueError('响应格式不是 Chat Completions')
            entry.update(status='returned',usage=self.last_usage,reasoning_chars=self.last_reasoning_chars)
            notify('model_end',status='returned',usage=self.last_usage)
            return msg
        except Exception as exc:
            entry.update(status='failed',error_type=type(exc).__name__)
            if isinstance(exc,HTTPError):entry['http_status']=exc.code
            notify('model_end',status='failed',error_type=type(exc).__name__,http_status=entry.get('http_status'))
            raise
        finally:entry['elapsed_s']=round(time.monotonic()-began,3)
    def request(self,messages,tools):
        if self.access_denied_http is not None:
            raise ModelServiceAccessDenied(f'模型服务曾返回HTTP {self.access_denied_http}，本次会话已停止新请求；需修复配置后新建会话。')
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
        if model.lower().startswith('qwen'):
            thinking=self.enable_thinking
            payload.update(stream=True,stream_options={'include_usage':True},enable_thinking=thinking,max_tokens=8192 if thinking else 4096)
            if thinking:payload['tool_choice']='auto'
        req=Request(url,data=json.dumps(payload,ensure_ascii=False).encode(),headers={'Content-Type':'application/json','Authorization':'Bearer '+c['api_key']})
        try:
            for attempt in range(3):
                try:
                    return self.perform_request(req,payload['stream'])
                except HTTPError:raise
                except (URLError,TimeoutError):
                    if attempt==2:raise
                    time.sleep(.5)
        except HTTPError as e:
            reasons={401:'密钥无效或已过期',403:'账号没有该模型权限',404:'服务地址或模型名称不正确',429:'余额不足或请求受限'}
            # Never return provider response bodies: some proxies echo credentials/requests.
            if e.code in (401,403):
                self.access_denied_http=e.code
                raise ModelServiceAccessDenied(f'API 请求失败（HTTP {e.code}）：服务拒绝访问；本次会话停止新请求。') from None
            raise ValueError(f"API 请求失败（HTTP {e.code}）：{reasons.get(e.code,'请检查服务状态与工具调用支持')}。") from None
        except (URLError,TimeoutError): raise ValueError('无法连接模型服务或请求超时，请检查网络与 API 地址。') from None
        except (KeyError,IndexError,json.JSONDecodeError): raise ValueError('服务响应格式不兼容，请确认支持 Chat Completions 和工具调用。') from None
