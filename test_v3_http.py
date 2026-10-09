import json,threading,unittest
from http.server import ThreadingHTTPServer
from urllib.request import Request,urlopen
from unittest.mock import patch
from app import Handler

class V3HTTPTests(unittest.TestCase):
    def test_v3_page_visual_status_and_detection(self):
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        base=f'http://127.0.0.1:{server.server_port}'
        try:
            with urlopen(base) as response:
                page=response.read().decode();self.assertIn('saveVlm',page);self.assertIn('value="agentic"',page);self.assertIn('/workflow-v3',page)
            with urlopen(base+'/workflow-v3') as response:self.assertEqual(response.status,200)
            with patch('visual_condition.config',return_value=dict(base_url='',model='',api_key='')):
                req=Request(base+'/api/visual-test',data=json.dumps(dict(record='s1a0-2')).encode(),headers={'Content-Type':'application/json'})
                with urlopen(req) as response:self.assertEqual(json.load(response)['status'],'disabled')
                req=Request(base+'/api/ask',data=json.dumps(dict(record='s1a0-2',question='检测整条记录',mode='agentic',backend='rules')).encode(),headers={'Content-Type':'application/json'})
                with urlopen(req,timeout=60) as response:data=json.load(response)
                self.assertEqual(data['result']['version'],'3.0-experimental');self.assertTrue(data['result']['agentic_segments'])
                with urlopen(base+data['result']['image']) as response:self.assertEqual(response.headers['Content-Type'],'image/png')
        finally:server.shutdown();server.server_close()

if __name__=='__main__':unittest.main()
