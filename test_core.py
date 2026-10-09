import unittest
import numpy as np
from unittest.mock import patch
from core import masks_to_intervals,physical_source,match_intervals,interval_iou,Agent,load

class Tests(unittest.TestCase):
    def test_rotation_sources(self):
        self.assertEqual(physical_source('s1a270from0zhen-1'),'s1a0zhen-1')
        self.assertEqual(physical_source('s2a180from90-2'),'s2a90-2')
    def test_inclusive_coordinates(self):
        self.assertEqual(interval_iou((0,9),(0,9)),1)
        self.assertEqual(interval_iou((0,9),(10,19)),0)
        self.assertEqual(masks_to_intervals(np.array([False,True,True,False]),1,0),[[1,2]])
    def test_no_duplicate_matching(self):
        self.assertEqual(len(match_intervals([(0,99),(0,99)],[(0,99)],.5)),1)
    def test_maximum_matching(self):
        self.assertEqual(len(match_intervals([(0,19),(0,9)],[(0,9),(10,19)],.5)),2)
    def test_detector_does_not_read_annotations(self):
        import pandas as pd
        original=pd.read_csv
        def guarded(path,*args,**kwargs):
            if 'annotation' in str(path): raise AssertionError('Inference attempted GT access')
            return original(path,*args,**kwargs)
        load.cache_clear()
        with patch.object(pd,'read_csv',guarded):
            response=Agent().ask('s1a0-1','检测整条记录')
            self.assertIn('candidates',response['result'])
    def test_followup_and_region(self):
        agent=Agent(); r=agent.ask('s1a0-1','检测整条记录')
        if r['result']['candidates']:
            cid=r['result']['candidates'][0]['id']
            self.assertEqual(agent.ask('s1a0-1','解释'+cid)['result']['candidate']['id'],cid)
        r=agent.ask('s1a0-1','检测5–10秒')
        self.assertEqual(r['result']['scope_samples'],[2500,5000])
    def test_no_fabricated_meters(self):
        r=Agent().ask('s1a0-1','缺陷在几米')
        self.assertFalse(r['result']['has_position'])
        self.assertIn('缺少',r['answer'])
    def test_llm_tool_loop_with_mock(self):
        from ollama_agent import OllamaAgent
        replies=iter([{'role':'assistant','content':'','tool_calls':[{'function':{'name':'audit','arguments':{}}}]},
                      {'role':'assistant','content':'已检查数据'}])
        agent=OllamaAgent('mock',lambda messages,tools:next(replies))
        r=agent.ask('s1a0-1','检查数据')
        self.assertEqual(r['trace'][0]['tool'],'audit')
    def test_llm_no_ungrounded_detection(self):
        from ollama_agent import OllamaAgent
        agent=OllamaAgent('mock',lambda messages,tools:{'role':'assistant','content':'有缺陷'})
        with self.assertRaises(ValueError): agent.ask('s1a0-1','检测')

if __name__=='__main__': unittest.main()
