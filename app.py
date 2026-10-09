"""Local-only web interface. Run: python app.py"""
import json, mimetypes, secrets, threading, webbrowser
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse,unquote
from core import Agent, ROOT, RUNS, records, TOOL_DESCRIPTIONS
from ollama_agent import OllamaAgent
from cloud_agent import CloudAgent
from api_config import public_config,save_config

SESSIONS={}; LOCK=threading.Lock()
class Handler(BaseHTTPRequestHandler):
    def send(self,code,body,kind='application/json; charset=utf-8'):
        raw=body if isinstance(body,bytes) else json.dumps(body,ensure_ascii=False).encode()
        self.send_response(code); self.send_header('Content-Type',kind)
        self.send_header('Cache-Control','no-store'); self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('Content-Length',str(len(raw))); self.end_headers(); self.wfile.write(raw)
    def do_GET(self):
        path=unquote(urlparse(self.path).path)
        if path in ('/工作流.png','/工作流说明.md','/钢丝绳漏磁专家知识.md','/自适应滤波算法复现与验证.md'):
            target=ROOT/path.lstrip('/')
            return self.send(200,target.read_bytes(),mimetypes.guess_type(target.name)[0] or 'text/plain; charset=utf-8')
        if path=='/api/workflow':
            from workflow import workflow_catalog
            return self.send(200,workflow_catalog())
        if path=='/knowledge':
            from expert_knowledge import knowledge_html
            return self.send(200,knowledge_html(),'text/html; charset=utf-8')
        if path=='/api/knowledge':
            from expert_knowledge import load_knowledge
            data=load_knowledge()
            return self.send(200,dict(version=data['version'],cards=len(data['cards']),sources=data['sources']))
        if path=='/workflow':return self.send(200,(ROOT/'workflow.html').read_bytes(),'text/html; charset=utf-8')
        if path=='/workflow-v3':return self.send(200,(ROOT/'workflow_v3.html').read_bytes(),'text/html; charset=utf-8')
        if path=='/api/config':
            try: return self.send(200,public_config())
            except ValueError as e: return self.send(400,{'error':str(e)})
        if path=='/api/vlm-config':
            from visual_condition import public_config as vlm_config
            return self.send(200,vlm_config())
        if path=='/api/records':
            return self.send(200,dict(records=[dict(id=r['record_id'],type=r['source_type'],group=r['group']) for r in records()],tools=TOOL_DESCRIPTIONS))
        if path=='/': return self.send(200,(ROOT/'index.html').read_bytes(),'text/html; charset=utf-8')
        if path.startswith('/runs/'):
            target=(ROOT/path.lstrip('/')).resolve()
            if not target.is_relative_to(RUNS.resolve()) or not target.is_file(): return self.send(404,{'error':'文件不存在'})
            return self.send(200,target.read_bytes(),mimetypes.guess_type(target.name)[0] or 'application/octet-stream')
        self.send(404,{'error':'接口不存在'})
    def do_POST(self):
        if self.path not in ('/api/ask','/api/config','/api/vlm-config','/api/visual-test'): return self.send(404,{'error':'接口不存在'})
        # Reject cross-origin browser requests to this local service.
        if self.headers.get('Origin') not in (None,'http://127.0.0.1:8765','http://localhost:8765','http://127.0.0.1:8767','http://localhost:8767'):
            return self.send(403,{'error':'只接受本地页面请求'})
        try:
            size=int(self.headers.get('Content-Length','0'))
            if size<1 or size>20000: raise ValueError('请求长度异常')
            req=json.loads(self.rfile.read(size))
            if self.path=='/api/visual-test':
                from agentic_dsp import inspect_visual_condition
                return self.send(200,inspect_visual_condition(req['record'],'S01'))
            if self.path=='/api/vlm-config':
                from visual_condition import save_config as save_vlm
                with LOCK:
                    result=save_vlm(req.get('base_url',''),req.get('model',''),req.get('api_key',''))
                    SESSIONS.clear()
                return self.send(200,result)
            if self.path=='/api/config':
                with LOCK:
                    result=save_config(req.get('base_url',''),req.get('model',''),req.get('api_key',''))
                    SESSIONS.clear()
                return self.send(200,result)
            session=req.get('session') or secrets.token_hex(16)
            backend=req.get('backend','rules')
            if backend not in ('rules','ollama','cloud'): raise ValueError('未知调度方式')
            key=(session,backend)
            with LOCK:
                if key not in SESSIONS:
                    if len(SESSIONS)>=100: SESSIONS.pop(next(iter(SESSIONS)))
                    SESSIONS[key]=CloudAgent() if backend=='cloud' else OllamaAgent() if backend=='ollama' else Agent()
                response=SESSIONS[key].ask(req['record'],str(req.get('question','检测整条记录')),req.get('mode','fast'))
            response['session']=session
            from datetime import datetime
            run_id=secrets.token_hex(8)
            archive=dict(created_at=datetime.now().isoformat(),record=req['record'],question=str(req.get('question','')),**response)
            output=RUNS/'sessions';output.mkdir(exist_ok=True)
            (output/(run_id+'.json')).write_text(json.dumps(archive,ensure_ascii=False,indent=2),encoding='utf-8')
            response['saved_result']='/runs/sessions/'+run_id+'.json'
            self.send(200,response)
        except (ValueError,KeyError,TypeError) as e: self.send(400,{'error':str(e)})
        except Exception as e:
            import traceback; traceback.print_exc()
            self.send(500,{'error':'分析失败：'+str(e)})
    def log_message(self,*args): pass

if __name__=='__main__':
    server=ThreadingHTTPServer(('127.0.0.1',8767),Handler)
    print('MFL Agent v3: http://127.0.0.1:8767',flush=True)
    import sys
    if '--no-browser' not in sys.argv:
        threading.Timer(1.,lambda:webbrowser.open('http://127.0.0.1:8767')).start()
    try: server.serve_forever()
    except KeyboardInterrupt: server.server_close()
