import io,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from cloud_agent import CloudAgent,chat_messages
from api_config import validate

class CloudTests(unittest.TestCase):
    def test_tool_ids_preserved(self):
        messages=[{'role':'assistant','tool_calls':[{'id':'a1','function':{'name':'audit','arguments':{}}}]},
                  {'role':'tool','tool_name':'audit','content':'{}'}]
        out=chat_messages(messages)
        self.assertEqual(out[0]['tool_calls'][0]['id'],out[1]['tool_call_id'])
    def test_real_tools_with_simulated_provider(self):
        replies=iter([{'choices':[{'message':{'role':'assistant','content':None,'tool_calls':[
            {'id':'scan_1','type':'function','function':{'name':'scan','arguments':'{"mode":"fast"}'}}]}}]},
            {'choices':[{'message':{'role':'assistant','content':'已检测，候选待复核。'}}]}])
        payloads=[]
        def provider(req,timeout):
            payloads.append(json.loads(req.data))
            return io.BytesIO(json.dumps(next(replies)).encode())
        config=dict(base_url='https://example.com/v1',model='test-model',api_key='test-key-not-real')
        with patch('cloud_agent.urlopen',provider):
            result=CloudAgent(config).ask('s1a0-1','检测整条记录')
        self.assertTrue(result['result']['candidates'])
        self.assertIn('云端',result['planner'])
        self.assertEqual(payloads[0]['tool_choice'],'required')
        self.assertEqual(payloads[1]['messages'][-1]['tool_call_id'],'scan_1')
    def test_no_cloud_key_in_public_config(self):
        import api_config
        file=Path(__file__).parent/'_test_api_config.json'
        self.assertFalse(file.exists())
        try:
            with patch.object(api_config,'FILE',file),patch.dict('os.environ',{'MFL_API_KEY':'','OPENAI_API_KEY':'','MFL_API_BASE':'','MFL_MODEL':''}):
                api_config.save_config('https://example.com/v1','model','test-key-not-real')
                disk=api_config.FILE.read_text()
                self.assertNotIn('test-key-not-real',disk)
                self.assertNotIn('api_key',api_config.public_config())
                self.assertTrue(api_config.public_config()['configured'])
        finally:
            file.unlink(missing_ok=True)
            file.with_suffix('.tmp').unlink(missing_ok=True)
    def test_no_remote_cleartext_keys(self):
        with self.assertRaises(ValueError):validate('http://example.com/v1','model')
    def test_deepseek_tool_routing_disables_thinking(self):
        payloads=[]
        def provider(req,timeout):
            payloads.append(json.loads(req.data))
            return io.BytesIO(json.dumps({'choices':[{'message':{'role':'assistant','content':'ok'}}]}).encode())
        config=dict(base_url='https://api.deepseek.com',model='deepseek-flash',api_key='test-key-not-real')
        with patch('cloud_agent.urlopen',provider):
            CloudAgent(config).request([{'role':'user','content':'检查数据'}],[])
        self.assertEqual(payloads[0]['thinking'],{'type':'disabled'})

if __name__=='__main__':unittest.main()
