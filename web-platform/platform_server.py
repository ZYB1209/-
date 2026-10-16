"""Private single-lab annotation and inference service; serve behind HTTPS."""
from __future__ import annotations
import base64,csv,hashlib,hmac,io,json,math,os,re,secrets,sqlite3,sys,threading,time,uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse,parse_qs
import numpy as np
import pandas as pd
from PIL import Image

ROOT=Path(__file__).resolve().parent
HEX=re.compile(r'^[a-f0-9]{32}$')
FIELDS=[f'mag_{c}{a}' for a in 'xyz' for c in range(1,33)]
MAX_BYTES=int(os.getenv('MFL_UPLOAD_MB','64'))*1024**2

class Conflict(ValueError):pass

class Store:
    def __init__(self,root):
        self.root=Path(root).resolve();self.data=self.root/'dataset';self.runs=self.root/'runs'
        for p in (self.root,self.data,self.data/'data',self.runs):p.mkdir(parents=True,exist_ok=True)
        self.db=self.root/'platform.sqlite';self.lock=threading.RLock()
        with self.connect() as c:
            c.executescript('''CREATE TABLE IF NOT EXISTS records(id TEXT PRIMARY KEY,name TEXT NOT NULL,samples INTEGER NOT NULL,sha256 TEXT NOT NULL,created REAL NOT NULL,warnings TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS annotations(record TEXT NOT NULL,revision INTEGER NOT NULL,payload TEXT NOT NULL,reviewed INTEGER NOT NULL,author TEXT NOT NULL,created REAL NOT NULL,PRIMARY KEY(record,revision));
CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY,record TEXT NOT NULL,mode TEXT NOT NULL,status TEXT NOT NULL,created REAL NOT NULL,finished REAL,result TEXT,error TEXT,annotation_revision INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS evaluations(id TEXT PRIMARY KEY,job TEXT NOT NULL,revision INTEGER NOT NULL,result TEXT NOT NULL,created REAL NOT NULL);''')
            c.execute("UPDATE jobs SET status='interrupted',error='服务重启，任务未完成；可重新提交' WHERE status IN ('queued','running')")
        self.manifest()
    @contextmanager
    def connect(self):
        c=sqlite3.connect(self.db,timeout=30);c.row_factory=sqlite3.Row
        try:
            c.execute('PRAGMA journal_mode=WAL')
            with c:yield c
        finally:c.close()
    def record(self,r):
        if not HEX.fullmatch(r):raise ValueError('记录编号无效')
        with self.connect() as c:row=c.execute('SELECT * FROM records WHERE id=?',(r,)).fetchone()
        if row is None:raise ValueError('记录不存在')
        return dict(row)
    def records(self):
        with self.connect() as c:rows=c.execute('SELECT r.*,(SELECT max(revision) FROM annotations a WHERE a.record=r.id) revision FROM records r ORDER BY created DESC').fetchall()
        return [dict(row) for row in rows]
    def manifest(self):
        with self.lock:
            temp=self.data/'manifest.tmp'
            with temp.open('w',encoding='utf-8',newline='') as f:
                w=csv.DictWriter(f,['record_id','source_type','source_collection','group','data_path','annotation_path','row_count','data_sha256','annotation_sha256']);w.writeheader()
                for r in self.records():w.writerow(dict(record_id=r['id'],source_type='physical_acquisition',source_collection='web_upload',group='unassigned',data_path='data/'+r['id']+'.csv',annotation_path='',row_count=r['samples'],data_sha256=r['sha256'],annotation_sha256=''))
            temp.replace(self.data/'manifest.csv')
    def upload(self,stream,length,name,fs):
        if not math.isfinite(fs) or fs!=500:raise ValueError('当前继承算法只支持已确认的500Hz数据；其他采样率需先适配算法')
        if not 0<length<=MAX_BYTES:raise ValueError('CSV大小超过限制或为空')
        rid=uuid.uuid4().hex;path=self.data/'data'/(rid+'.csv');temp=path.with_suffix('.upload')
        try:
            with temp.open('wb') as f:
                left=length
                while left:
                    chunk=stream.read(min(left,1024**2))
                    if not chunk:raise ValueError('文件上传不完整')
                    f.write(chunk);left-=len(chunk)
            frame=pd.read_csv(temp,nrows=100001)
            if len(frame)>100000:raise ValueError('第一版单条记录最多100000点，请先按连续记录分批上传')
            if not 512<=len(frame):raise ValueError('记录至少需要512个采样点')
            if any(k not in frame for k in FIELDS):raise ValueError('需要mag_1x…mag_32x、mag_1y…mag_32y、mag_1z…mag_32z共96列')
            x=frame[FIELDS].to_numpy(float)
            if not np.isfinite(x).all():raise ValueError('磁通道有缺失或非有限值，请先修复')
            warnings=['名义时间按用户确认的500Hz换算；无位置表时不报告米制长度']
            if 'time(0.1ms)' in frame:
                t=frame['time(0.1ms)'].to_numpy(float)/10000;dt=np.diff(t)
                if not np.isfinite(t).all() or np.any(dt<=0) or abs(np.median(dt)-.002)>.0005:warnings.append('原始时间轴异常或与500Hz不符；标注以原始行号为准，请复核采样条件')
            else:warnings.append('未提供原始时间戳，采样率尚无独立核对')
            digest=hashlib.sha256(temp.read_bytes()).hexdigest();temp.replace(path)
            with self.lock:
                with self.connect() as c:c.execute('INSERT INTO records VALUES(?,?,?,?,?,?)',(rid,str(name)[:160],len(frame),digest,time.time(),json.dumps(warnings,ensure_ascii=False)))
                self.manifest()
            return self.record(rid)
        except Exception:
            temp.unlink(missing_ok=True)
            raise
    def labels(self,r,revision=None):
        self.record(r)
        with self.connect() as c:
            row=c.execute('SELECT * FROM annotations WHERE record=? '+('AND revision=?' if revision is not None else 'ORDER BY revision DESC LIMIT 1'),(r,revision) if revision is not None else (r,)).fetchone()
        if row is None:
            if revision is not None:raise ValueError('标注版本不存在')
            return dict(record=r,revision=0,items=[],reviewed=False,author='')
        return dict(record=r,revision=row['revision'],items=json.loads(row['payload']),reviewed=bool(row['reviewed']),author=row['author'])
    def save_labels(self,r,body):
        n=self.record(r)['samples'];items=body.get('items');base=body.get('revision');reviewed=body.get('reviewed',False)
        if type(base)is not int or type(reviewed)is not bool or not isinstance(items,list) or len(items)>2000:raise ValueError('标注数据无效')
        clean=[];ids=set()
        for a in items:
            if not isinstance(a,dict):raise ValueError('标注格式无效')
            s,e,c,d=(a.get(k) for k in ('start','end','channel_start','channel_end'))
            if not all(type(v)is int for v in (s,e,c,d)) or not 0<=s<=e<n or not 1<=c<=d<=32:raise ValueError('标注范围越界；采样点从0开始，通道从1开始')
            if a.get('axis') not in ('X','Y','Z','ALL'):raise ValueError('标注轴无效')
            aid=a.get('id') or uuid.uuid4().hex
            if not isinstance(aid,str) or not HEX.fullmatch(aid) or aid in ids:raise ValueError('标注编号无效或重复')
            ids.add(aid);kind=str(a.get('kind','未分类')).strip()[:80]
            clean.append(dict(id=aid,start=s,end=e,channel_start=c,channel_end=d,axis=a['axis'],kind=kind,note=str(a.get('note',''))[:1000]))
        with self.lock,self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            latest=db.execute('SELECT COALESCE(max(revision),0) FROM annotations WHERE record=?',(r,)).fetchone()[0]
            if latest!=base:raise Conflict('其他人已保存新版本；请先重新加载标注，避免覆盖')
            db.execute('INSERT INTO annotations VALUES(?,?,?,?,?,?)',(r,base+1,json.dumps(clean,ensure_ascii=False),int(reviewed),str(body.get('author','课题组'))[:80],time.time()))
        return self.labels(r)

class Service:
    def __init__(self,store,engine=True):
        self.store=store;self.pool=ThreadPoolExecutor(max_workers=1)
        if engine:
            os.environ['MFL_DATA_DIR']=str(store.data);os.environ['MFL_RUNS_DIR']=str(store.runs)
            sys.path.insert(0,str(ROOT/'engine'))
            import core
            self.core=core
    def job(self,j):
        if not HEX.fullmatch(j):raise ValueError('任务编号无效')
        with self.store.connect() as c:row=c.execute('SELECT * FROM jobs WHERE id=?',(j,)).fetchone()
        if row is None:raise ValueError('任务不存在')
        d=dict(row);d['result']=json.loads(d['result']) if d['result'] else None;return d
    def submit(self,r,mode,budget):
        self.store.record(r)
        if mode not in ('fast','rules','cloud_unified','tfo','pyramid_tfo'):raise ValueError('检测模式无效')
        if type(budget)is not int or not 1<=budget<=40:raise ValueError('调用上限须为1–40')
        if mode=='cloud_unified':
            from api_config import public_config
            if not public_config()['configured']:raise ValueError('云端尚未配置模型，请管理员设置服务器环境变量')
        with self.store.lock,self.store.connect() as c:
            if c.execute("SELECT count(*) FROM jobs WHERE status IN ('queued','running')").fetchone()[0]>=8:raise ValueError('任务队列已满，请稍后重试')
            j=uuid.uuid4().hex;c.execute('INSERT INTO jobs(id,record,mode,status,created,annotation_revision) VALUES(?,?,?,?,?,?)',(j,r,mode,'queued',time.time(),self.store.labels(r)['revision']))
        self.pool.submit(self.run,j,budget);return self.job(j)
    def run(self,j,budget):
        job=self.job(j);r=job['record']
        with self.store.connect() as c:c.execute("UPDATE jobs SET status='running' WHERE id=?",(j,))
        try:
            meta=self.store.record(r)
            if hashlib.sha256((self.store.data/'data'/(r+'.csv')).read_bytes()).hexdigest()!=meta['sha256']:raise ValueError('原始数据校验失败')
            if job['mode']=='fast':result=self.core.analyze(r,'fast')
            elif job['mode'] in ('tfo','pyramid_tfo'):
                from paper_methods import literature_detect
                result=literature_detect(r,job['mode'])
            else:
                from agentic_dsp import agentic_detect
                planner=None
                if job['mode']=='cloud_unified':
                    from cloud_agent import CloudAgent
                    agent=CloudAgent();agent.max_model_calls=budget;planner=agent.request
                result=agentic_detect(r,planner=planner,perception='fused' if planner else 'numeric',planner_mode='unified' if planner else 'separated')
            frozen=dict(result,platform_job_id=j,input_sha256=meta['sha256'],annotations_used=False)
            raw=json.dumps(frozen,ensure_ascii=False,allow_nan=False)
            (self.store.runs/(j+'.json')).write_text(raw,encoding='utf-8')
            with self.store.connect() as c:c.execute("UPDATE jobs SET status='completed',finished=?,result=? WHERE id=?",(time.time(),raw,j))
        except Exception as e:
            # Provider response bodies may contain sensitive data; do not return them.
            error=str(e)[:250] if isinstance(e,ValueError) and job['mode']!='cloud_unified' else '任务失败（'+type(e).__name__+'）；请管理员核对数据、模型服务或请求预算'
            with self.store.connect() as c:c.execute("UPDATE jobs SET status='failed',finished=?,error=? WHERE id=?",(time.time(),error,j))
    def score(self,j,revision):
        job=self.job(j)
        if job['status']!='completed':raise ValueError('请先完成检测')
        labels=self.store.labels(job['record'],revision)
        if not labels['reviewed']:raise ValueError('请先确认整条记录已完整复核；草稿标注不计算正式指标')
        from evaluate_agent import event_metrics,point_metrics
        result=job['result']
        pred=[(c['refined_start'],c['refined_end']) for c in result['candidates']]
        truth=[(a['start'],a['end']) for a in labels['items']]
        metrics=dict(events={str(t):event_metrics(pred,truth,t) for t in (.3,.5,.7)},points=point_metrics(pred,truth,self.store.record(job['record'])['samples']))
        report=dict(id=uuid.uuid4().hex,job_id=j,record=job['record'],annotation_revision=revision,labels_snapshot=labels,prediction_sha256=hashlib.sha256(json.dumps(result,sort_keys=True).encode()).hexdigest(),metrics=metrics,scope='时间区间一对一匹配；通道范围与缺陷类型保留在标注中，本版不计算空间或分类准确率',warning='每个框代表一个标注事件；不要跨轴重复框选同一缺陷。未标注区域按正常处理，须确认全记录复核完整。')
        with self.store.connect() as c:c.execute('INSERT INTO evaluations VALUES(?,?,?,?,?)',(report['id'],j,revision,json.dumps(report,ensure_ascii=False),time.time()))
        return report

class Handler(BaseHTTPRequestHandler):
    server_version='MFLPlatform'
    def send(self,code,body,kind='application/json; charset=utf-8',cookie=None):
        data=body if isinstance(body,bytes) else json.dumps(body,ensure_ascii=False,allow_nan=False).encode()
        self.send_response(code)
        for k,v in {'Content-Type':kind,'Content-Length':str(len(data)),'Cache-Control':'no-store','X-Content-Type-Options':'nosniff','Referrer-Policy':'same-origin','Content-Security-Policy':"default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' blob:; object-src 'none'; frame-ancestors 'none'; base-uri 'none'"}.items():self.send_header(k,v)
        if cookie:self.send_header('Set-Cookie',cookie)
        self.end_headers();self.wfile.write(data)
    def token(self):
        jar=SimpleCookie();jar.load(self.headers.get('Cookie',''));cookie=jar.get('mfl_session')
        value=cookie.value if cookie else '';parts=value.split('.')
        if len(parts)!=3:return None
        payload='.'.join(parts[:2]);expected=hmac.new(self.server.secret,payload.encode(),'sha256').hexdigest()
        if not hmac.compare_digest(expected,parts[2]):return None
        try:valid=int(parts[0])>time.time()
        except ValueError:valid=False
        return value if valid else None
    def csrf(self,token):return hmac.new(self.server.secret,('csrf:'+token).encode(),'sha256').hexdigest()
    def body(self):
        length=int(self.headers.get('Content-Length','0'))
        if not 0<length<=1024**2:raise ValueError('请求大小无效')
        d=json.loads(self.rfile.read(length))
        if not isinstance(d,dict):raise ValueError('请求格式无效')
        return d
    def do_GET(self):self.handle_request('GET')
    def do_POST(self):self.handle_request('POST')
    def do_PUT(self):self.handle_request('PUT')
    def handle_request(self,method):
        path=urlparse(self.path).path;q=parse_qs(urlparse(self.path).query)
        try:
            service=self.server.service;store=service.store
            if method=='GET' and path=='/healthz':return self.send(200,{'ok':True})
            if method=='GET' and path in ('/','/platform.js','/platform.css'):
                file=ROOT/'web'/('index.html' if path=='/' else path[1:]);kind={'/':'text/html; charset=utf-8','/platform.js':'text/javascript; charset=utf-8','/platform.css':'text/css; charset=utf-8'}[path]
                return self.send(200,file.read_bytes(),kind)
            if method!='GET':
                origin=self.headers.get('Origin')
                allowed=os.getenv('MFL_PUBLIC_URL','').rstrip('/') or ('http://'+self.headers.get('Host',''))
                if origin and origin!=allowed:return self.send(403,{'error':'页面来源不匹配'})
            if method=='POST' and path=='/api/login':
                now=time.time();ip=self.client_address[0]
                with self.server.guard:
                    attempts=[t for t in self.server.attempts.get(ip,[]) if now-t<60]
                    self.server.attempts[ip]=attempts
                    if len(attempts)>=6:return self.send(429,{'error':'尝试过于频繁，请一分钟后再试'})
                    attempts.append(now)
                password=str(self.body().get('password',''))
                if not hmac.compare_digest(password.encode(),self.server.password.encode()):return self.send(401,{'error':'口令错误'})
                payload=str(int(now+8*3600))+'.'+secrets.token_hex(16);token=payload+'.'+hmac.new(self.server.secret,payload.encode(),'sha256').hexdigest()
                return self.send(200,{'csrf':self.csrf(token)},cookie='mfl_session='+token+'; HttpOnly; SameSite=Strict; Path=/; Max-Age=28800'+self.server.secure_cookie)
            token=self.token()
            if not token:return self.send(401,{'error':'请先登录'})
            if method!='GET' and not hmac.compare_digest(self.csrf(token),self.headers.get('X-CSRF-Token','')):return self.send(403,{'error':'请求校验失败，请刷新后重试'})
            if path=='/api/session' and method=='GET':return self.send(200,{'csrf':self.csrf(token),'cloud_configured':bool(os.getenv('MFL_API_KEY'))})
            if path=='/api/logout' and method=='POST':return self.send(200,{'ok':True},cookie='mfl_session=; HttpOnly; SameSite=Strict; Path=/; Max-Age=0'+self.server.secure_cookie)
            if path=='/api/records' and method=='GET':return self.send(200,{'records':store.records()})
            if path=='/api/upload' and method=='PUT':return self.send(201,store.upload(self.rfile,int(self.headers.get('Content-Length','0')),q.get('name',['record.csv'])[0],float(self.headers.get('X-Sampling-Hz','0'))))
            match=re.fullmatch('/api/records/([a-f0-9]{32})/(labels|heatmap|jobs|export)',path)
            if match:
                r,action=match.groups();meta=store.record(r)
                if action=='labels':
                    if method=='GET':return self.send(200,store.labels(r))
                    if method=='POST':return self.send(200,store.save_labels(r,self.body()))
                if action=='heatmap' and method=='GET':return self.heatmap(meta,q)
                if action=='jobs':
                    if method=='POST':
                        d=self.body();return self.send(202,service.submit(r,d.get('mode','fast'),d.get('budget',12)))
                    with store.connect() as c:rows=c.execute('SELECT id,record,mode,status,created,finished,error,annotation_revision FROM jobs WHERE record=? ORDER BY created DESC LIMIT 40',(r,)).fetchall()
                    return self.send(200,{'jobs':[dict(x) for x in rows]})
                if action=='export' and method=='GET':
                    labels=store.labels(r);buf=io.StringIO();w=csv.writer(buf);w.writerow(['label','start_row_idx','end_row_idx','axis','channel_start','channel_end','defect_type','note','annotation_revision'])
                    for a in labels['items']:
                        safe=lambda v:"'"+v if v.startswith(('=','+','-','@')) else v
                        w.writerow(['anomaly',a['start'],a['end'],a['axis'],a['channel_start'],a['channel_end'],safe(a['kind']),safe(a['note']),labels['revision']])
                    return self.send(200,buf.getvalue().encode('utf-8-sig'),'text/csv; charset=utf-8')
            match=re.fullmatch('/api/jobs/([a-f0-9]{32})(/score)?',path)
            if match:
                j,action=match.groups()
                if method=='GET' and not action:return self.send(200,service.job(j))
                if method=='POST' and action:
                    d=self.body();revision=d.get('revision')
                    if type(revision)is not int or revision<1:raise ValueError('请选择已保存标注版本')
                    return self.send(200,service.score(j,revision))
            if method=='GET' and path.startswith('/runs/'):
                file=(store.runs/path[6:]).resolve()
                if not file.is_relative_to(store.runs) or not file.is_file() or file.suffix not in ('.png','.json'):return self.send(404,{'error':'结果不存在'})
                return self.send(200,file.read_bytes(),'image/png' if file.suffix=='.png' else 'application/json; charset=utf-8')
            return self.send(404,{'error':'接口不存在'})
        except Conflict as e:return self.send(409,{'error':str(e)})
        except (ValueError,KeyError,TypeError) as e:return self.send(400,{'error':str(e)[:250]})
        except (BrokenPipeError,ConnectionResetError):return
        except Exception:return self.send(500,{'error':'服务器处理失败；请管理员查看服务状态'})
    def heatmap(self,meta,q):
        n=meta['samples'];a=int(q.get('start',['0'])[0]);b=int(q.get('end',[str(n-1)])[0]);axis=q.get('axis',['X'])[0]
        if not 0<=a<=b<n or axis not in ('X','Y','Z'):raise ValueError('显示范围无效')
        x=self.server.service.core.image_tensor(meta['id'])['XYZ'.index(axis),a:b+1,:]
        width=min(1400,len(x));edges=np.linspace(0,len(x),width+1,dtype=int);values=[]
        for lo,hi in zip(edges[:-1],edges[1:]):
            block=x[lo:hi];values.append(np.take_along_axis(block,np.argmax(np.abs(block),axis=0)[None,:],axis=0)[0])
        z=np.asarray(values).T;level=max(float(np.quantile(np.abs(z),.995)),1e-9);v=np.clip(z/level,-1,1)
        rgb=np.stack([np.where(v>0,245,245+v*200),245-np.abs(v)*185,np.where(v<0,245,245-v*200)],axis=-1).astype('uint8')
        image=Image.fromarray(rgb);out=io.BytesIO();image.save(out,format='PNG');return self.send(200,out.getvalue(),'image/png')
    def log_message(self,*args):pass

def create_server(address,root,password,secret,secure=True,engine=True):
    if len(password)<12 or len(secret)<32:raise ValueError('平台口令至少12字符，会话密钥至少32字符；请通过服务器Secrets配置')
    server=ThreadingHTTPServer(address,Handler);server.service=Service(Store(root),engine)
    server.password=password;server.secret=secret.encode();server.secure_cookie='; Secure' if secure else ''
    server.guard=threading.Lock();server.attempts={};return server

if __name__=='__main__':
    server=create_server((os.getenv('HOST','0.0.0.0'),int(os.getenv('PORT','8080'))),os.getenv('MFL_STORAGE_DIR',str(ROOT/'storage')),os.environ.get('MFL_PLATFORM_PASSWORD',''),os.environ.get('MFL_SESSION_SECRET',''),os.getenv('MFL_COOKIE_SECURE','true').lower()=='true')
    print('MFL annotation platform ready',flush=True)
    try:server.serve_forever()
    finally:server.service.pool.shutdown(wait=False,cancel_futures=True);server.server_close()
