"""HTTP integration test runs client/server in one process (sandbox-local)."""
import json,threading,unittest
from http.server import ThreadingHTTPServer
from urllib.request import Request,urlopen
from app import Handler

class HTTPTests(unittest.TestCase):
    def test_end_to_end(self):
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        base=f'http://127.0.0.1:{server.server_port}'
        def ask(question,session=''):
            req=Request(base+'/api/ask',data=json.dumps(dict(record='s1a0-1',question=question,session=session)).encode(),headers={'Content-Type':'application/json'})
            with urlopen(req,timeout=15) as response: return json.load(response)
        try:
            with urlopen(base+'/api/records',timeout=15) as r:self.assertEqual(len(json.load(r)['records']),57)
            result=ask('检测整条记录'); self.assertTrue(result['trace'])
            with urlopen(base+result['result']['image'],timeout=15) as r:self.assertEqual(r.headers['Content-Type'],'image/png')
            candidate=result['result']['candidates'][0]['id']
            followup=ask('解释'+candidate,result['session'])
            self.assertEqual(followup['result']['candidate']['id'],candidate)
        finally:server.shutdown();server.server_close()

if __name__=='__main__':unittest.main()
