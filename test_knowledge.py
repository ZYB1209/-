import json, threading, unittest
from http.server import ThreadingHTTPServer
from urllib.request import urlopen
from unittest.mock import patch
from expert_knowledge import load_knowledge, lookup_knowledge
from core import Agent
from cloud_agent import CloudAgent
from app import Handler

class KnowledgeTests(unittest.TestCase):
    def test_sources_and_discrepancy_retrieval(self):
        data=load_knowledge()
        self.assertEqual(len(data['cards']),25)
        self.assertEqual({s['id'] for s in data['sources']},{'P2021','P2023'})
        for c in data['cards']:
            for r in c['references']:
                self.assertTrue(all(1<=p<=11 for p in r['pdf_pages']))
        result=lookup_knowledge('SNR信噪比267%如何计算',1)
        self.assertEqual(result['knowledge_items'][0]['id'],'K21')
        self.assertIn('1.093',result['knowledge_items'][0]['limitations'])

    def test_no_match_and_argument_bounds(self):
        self.assertEqual(lookup_knowledge('unsupported_zzzz')['knowledge_items'],[])
        for args in [('',4),('抖动',7),('抖动',True),('抖动','4'),('x'*601,4)]:
            with self.assertRaises(ValueError):lookup_knowledge(*args)

    def test_offline_domain_question_does_not_scan(self):
        agent=Agent()
        with patch('core.analyze',side_effect=AssertionError('knowledge must not scan')):
            result=agent.ask('s1a0-1','根据论文，为什么抖动容易被当成缺陷？')
        self.assertEqual(result['trace'][0]['tool'],'lookup_knowledge')
        self.assertIn('P202',result['answer'])
        with patch('core.evaluate',side_effect=AssertionError('literature must not read annotations')):
            result=agent.ask('s1a0-1','根据2023论文，标注与误报的关系是什么？')
        self.assertEqual(result['trace'][0]['tool'],'lookup_knowledge')

    def test_llm_receives_background_and_keeps_detection_and_citations(self):
        captured=[]
        replies=iter([
            dict(role='assistant',content='',tool_calls=[dict(id='e',function=dict(name='explain',arguments={'candidate_id':'D01'}))]),
            dict(role='assistant',content='',tool_calls=[dict(id='k',function=dict(name='lookup_knowledge',arguments={'query':'抖动与缺陷的图像形态','limit':2}))]),
            dict(role='assistant',content='这是待复核候选，文献形态仅提供线索。')])
        def transport(messages,tools):
            captured.append(json.loads(messages[1]['content']))
            self.assertIn('lookup_knowledge',[t['function']['name'] for t in tools])
            return next(replies)
        agent=CloudAgent(dict(model='mock',base_url='https://example.com',api_key='test'),transport)
        agent.memory['s1a0-1']={'candidates':[]}
        def fake_call(name,record,**args):
            if name=='explain':return dict(candidate=dict(id='D01'))
            return lookup_knowledge(**args)
        with patch.object(agent,'call',side_effect=fake_call):
            result=agent.ask('s1a0-1','解释D01并排查抖动')
        self.assertEqual(result['result']['candidate']['id'],'D01')
        self.assertEqual(len(result['knowledge_evidence']),2)
        self.assertEqual(captured[0]['expert_background']['source_ids'],['P2021','P2023'])

    def test_knowledge_web_routes(self):
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        threading.Thread(target=server.serve_forever,daemon=True).start()
        try:
            base=f'http://127.0.0.1:{server.server_port}'
            with urlopen(base+'/api/knowledge') as r:self.assertEqual(json.load(r)['cards'],25)
            with urlopen(base+'/knowledge') as r:
                html=r.read().decode('utf-8')
                self.assertIn('K21',html)
                self.assertIn('尚未',html)
        finally:server.shutdown();server.server_close()

if __name__=='__main__':unittest.main()
