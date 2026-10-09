import json,unittest
from cloud_agent import CloudAgent

class ConversationTests(unittest.TestCase):
    def test_model_receives_previous_dialogue_and_candidates(self):
        prompts=[]
        replies=iter([
          {'role':'assistant','content':'','tool_calls':[{'id':'a','function':{'name':'scan','arguments':{}}}]},
          {'role':'assistant','content':'检测得到候选D01。'},
          {'role':'assistant','content':'','tool_calls':[{'id':'b','function':{'name':'explain','arguments':{'candidate_id':'D01'}}}]},
          {'role':'assistant','content':'第一个候选的证据已提取。'}])
        def transport(messages,tools):
            prompts.append(json.loads(messages[1]['content']))
            return next(replies)
        config=dict(base_url='https://example.com/v1',model='mock',api_key='test-key-not-real')
        agent=CloudAgent(config,transport)
        first=agent.ask('s1a0-1','检测整条记录')
        second=agent.ask('s1a0-1','解释刚才的第一个')
        self.assertEqual(second['result']['candidate']['id'],'D01')
        self.assertEqual(len(prompts[2]['recent_dialogue']),2)
        self.assertTrue(prompts[2]['previous_candidates'])

if __name__=='__main__':unittest.main()
