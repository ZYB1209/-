import hashlib,io,json,os,sys,tempfile,threading,time,unittest
from pathlib import Path
from urllib.request import Request,urlopen
from urllib.error import HTTPError
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from platform_server import Store,Service,Conflict,FIELDS,create_server

def sample():
    rng=np.random.default_rng(12);x=rng.normal(0,.01,(1024,96));x[450:570,3:7]+=.8
    x[450:570,35:39]+=.8;x[450:570,67:71]+=.8
    f=pd.DataFrame(x,columns=FIELDS);f['time(0.1ms)']=np.arange(1024)*20
    return f.to_csv(index=False).encode()
def label(start=450,end=570):return dict(start=start,end=end,channel_start=4,channel_end=7,axis='ALL',kind='断丝',note='人工复核')

class StorageTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=Path(__file__).parent);self.store=Store(self.tmp.name);self.raw=sample();self.r=self.store.upload(io.BytesIO(self.raw),len(self.raw),'sample.csv',500)
    def tearDown(self):self.tmp.cleanup()
    def test_upload_hash_and_private_manifest_without_labels(self):
        self.assertEqual(self.r['sha256'],hashlib.sha256(self.raw).hexdigest())
        meta=pd.read_csv(self.store.data/'manifest.csv',keep_default_na=False).iloc[0]
        self.assertEqual(meta['annotation_path'],'');self.assertEqual(meta['row_count'],1024)
    def test_annotation_version_and_lost_update_protection(self):
        v1=self.store.save_labels(self.r['id'],dict(revision=0,items=[label()],reviewed=True))
        with self.assertRaises(Conflict):self.store.save_labels(self.r['id'],dict(revision=0,items=[],reviewed=False))
        self.store.save_labels(self.r['id'],dict(revision=1,items=[],reviewed=True))
        self.assertEqual(len(self.store.labels(self.r['id'],1)['items']),1)
        self.assertEqual(self.store.labels(self.r['id'])['revision'],2)
        self.assertEqual(v1['items'][0]['channel_start'],4)
    def test_invalid_coordinates_or_sampling_rejected(self):
        for bad in [dict(label(),end=1024),dict(label(),channel_start=0),dict(label(),start=1.5)]:
            with self.assertRaises(ValueError):self.store.save_labels(self.r['id'],dict(revision=0,items=[bad]))
        with self.assertRaises(ValueError):self.store.upload(io.BytesIO(self.raw),len(self.raw),'other.csv',2000)
    def test_evaluation_freezes_label_revision_and_does_not_call_model(self):
        service=Service(self.store);service.core.DATA=self.store.data;service.core.RUNS=self.store.runs
        service.core.load.cache_clear();service.core.image_tensor.cache_clear()
        try:
            j='b'*32;result=dict(record=self.r['id'],candidates=[dict(refined_start=450,refined_end=570)])
            with self.store.connect() as c:c.execute('INSERT INTO jobs(id,record,mode,status,created,result,annotation_revision) VALUES(?,?,?,?,?,?,?)',(j,self.r['id'],'fast','completed',time.time(),json.dumps(result),0))
            self.store.save_labels(self.r['id'],dict(revision=0,items=[label()],reviewed=False))
            with self.assertRaises(ValueError):service.score(j,1)
            self.store.save_labels(self.r['id'],dict(revision=1,items=[label()],reviewed=True))
            first=service.score(j,2)
            self.store.save_labels(self.r['id'],dict(revision=2,items=[label(50,100)],reviewed=True))
            second=service.score(j,3)
            self.assertEqual(first['metrics']['events']['0.5']['tp'],1)
            self.assertEqual(second['metrics']['events']['0.5']['tp'],0)
            self.assertEqual(first['labels_snapshot']['items'][0]['start'],450)
            self.assertEqual(first['prediction_sha256'],second['prediction_sha256'])
        finally:service.pool.shutdown()
    def test_restart_marks_unfinished_jobs_interrupted(self):
        with self.store.connect() as c:c.execute('INSERT INTO jobs(id,record,mode,status,created,annotation_revision) VALUES(?,?,?,?,?,?)',('c'*32,self.r['id'],'fast','running',time.time(),0))
        restored=Store(self.store.root)
        with restored.connect() as c:self.assertEqual(c.execute('SELECT status FROM jobs').fetchone()[0],'interrupted')

class HTTPTests(unittest.TestCase):
    def test_authentication_csrf_upload_labels_and_prediction(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            server=create_server(('127.0.0.1',0),tmp,'test-password-123','test-secret-'+'x'*40,secure=False)
            server.service.core.DATA=server.service.store.data;server.service.core.RUNS=server.service.store.runs
            server.service.core.load.cache_clear();server.service.core.image_tensor.cache_clear()
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            base='http://127.0.0.1:'+str(server.server_port);cookie='';csrf=''
            def call(path,body=None,method=None,headers=None):
                h={'Cookie':cookie,'X-CSRF-Token':csrf,**(headers or {})}
                if isinstance(body,dict):body=json.dumps(body).encode();h['Content-Type']='application/json'
                response=urlopen(Request(base+path,data=body,method=method,headers=h),timeout=40)
                return response,json.load(response)
            try:
                with self.assertRaises(HTTPError) as error:call('/api/records')
                self.assertEqual(error.exception.code,401)
                response,data=call('/api/login',{'password':'test-password-123'})
                cookie=response.headers['Set-Cookie'].split(';')[0];csrf=data['csrf']
                with self.assertRaises(HTTPError) as error:call('/api/logout',{},headers={'X-CSRF-Token':'bad'})
                self.assertEqual(error.exception.code,403)
                _,r=call('/api/upload?name=synthetic.csv',sample(),'PUT',{'X-Sampling-Hz':'500'})
                _,a=call('/api/records/'+r['id']+'/labels',dict(revision=0,items=[label()],reviewed=True))
                self.assertEqual(a['revision'],1)
                _,j=call('/api/records/'+r['id']+'/jobs',dict(mode='fast',budget=1))
                deadline=time.time()+35
                while time.time()<deadline:
                    _,j=call('/api/jobs/'+j['id'])
                    if j['status'] not in ('queued','running'):break
                    time.sleep(.1)
                self.assertEqual(j['status'],'completed',j.get('error'))
                self.assertFalse(j['result']['annotations_used'])
                _,score=call('/api/jobs/'+j['id']+'/score',dict(revision=1))
                self.assertEqual(score['annotation_revision'],1)
                self.assertIn('0.5',score['metrics']['events'])
                with self.assertRaises(HTTPError):call('/runs/../../platform.sqlite')
                call('/api/logout',{})
            finally:
                server.shutdown();server.server_close();server.service.pool.shutdown()

if __name__=='__main__':unittest.main()
