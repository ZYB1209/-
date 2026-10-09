import tempfile,unittest,json
from pathlib import Path
from unittest.mock import patch
import numpy as np
from test_adaptive import synthetic
from parameter_guard import guard
from dsp_tools import execute
from adaptive_signal import condition_features
from agentic_dsp import process_block,model_plan,agentic_detect
from visual_condition import render,SCHEMA,validate_observation,inspect,conflicts
from visual_condition import parse_content

class V3Tests(unittest.TestCase):
    def test_json_repair_never_invents_attribute_values(self):
        data={k:{f:choices[0] for f,choices in fields.items()} for k,fields in SCHEMA.items()}
        data.update(confidence={'value':.5},evidence=['visible bands'])
        text=json.dumps(data)
        observation,repair=parse_content(text[:-1])
        self.assertEqual(observation,data);self.assertEqual(repair,'single_missing_root_closing_brace')
        with self.assertRaises(ValueError):parse_content(text[:-2])
        incomplete=dict(data);incomplete.pop('local_response')
        with self.assertRaises(ValueError):parse_content(json.dumps(incomplete))

    def test_all_templates_and_bounded_parameters(self):
        z=synthetic('strand',768);features=condition_features(z)
        for template in 'ABCD':
            p=guard(dict(template=template,controls={'strength':'strong'}),features,768)
            out,score,logs=execute(z,p)
            self.assertEqual(out.shape,z.shape);self.assertTrue(np.isfinite(out).all());self.assertGreaterEqual(len(logs),4)
        for plan in [dict(template='X'),dict(template='A',controls={'strength':100}),dict(template='A',disabled=['extract_hilbert_envelope'])]:
            with self.assertRaises(ValueError):guard(plan,features,768)

    def test_schema_and_canonical_images(self):
        z=synthetic('shake',768)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent/'runs') as folder:
            a=Path(folder)/'a.png';b=Path(folder)/'b.png';render(z,a);render(z,b);self.assertEqual(a.read_bytes(),b.read_bytes())
        data={k:{f:choices[0] for f,choices in fields.items()} for k,fields in SCHEMA.items()}
        data.update(confidence={'value':.5},evidence=['visible bands'])
        validate_observation(data);data['speed']=1
        with self.assertRaises(ValueError):validate_observation(data)

    def test_replan_and_final_fallback_are_bounded(self):
        z=synthetic('quiet',768);f=condition_features(z)
        with patch('agentic_dsp.verify',return_value=dict(verdict='reject',warnings=['forced rejection'])):
            out,log=process_block(z,f,dict(status='disabled'))
        self.assertEqual(log['attempts'],2);self.assertEqual(log['status'],'fallback');self.assertEqual(log['final_executed_pipeline']['template'],'preset_baseline')

    def test_model_receives_verifier_and_can_revise(self):
        z=synthetic('quiet',768);f=condition_features(z);seen=[]
        def transport(messages,tools):
            seen.append(json.loads(messages[1]['content']))
            return dict(tool_calls=[dict(function=dict(name='submit_dsp_plan',arguments=json.dumps(dict(template='A',controls={},reason='test'))))])
        with patch('agentic_dsp.verify',side_effect=[dict(verdict='reject',warnings=['attenuated']),dict(verdict='accept')]):
            _,log=process_block(z,f,dict(status='disabled'),transport)
        self.assertEqual(log['status'],'accepted');self.assertEqual(log['attempts'],2);self.assertEqual(seen[1]['verifier_feedback']['warnings'],['attenuated'])

    def test_vlm_disabled_does_not_fake_observations(self):
        z=synthetic('quiet',768)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent/'runs') as folder,patch('visual_condition.config',return_value=dict(base_url='',model='',api_key='')):
            result=inspect('test',{'segment_id':'S01'},z,Path(folder))
        self.assertEqual(result['status'],'disabled');self.assertIsNone(result['visual_condition'])

    def test_inference_annotation_isolation_and_followup(self):
        import core,pandas as pd
        original=pd.read_csv
        def guarded(path,*args,**kwargs):
            if 'annotations' in str(path):raise AssertionError('GT leak')
            return original(path,*args,**kwargs)
        with patch('pandas.read_csv',side_effect=guarded),patch('visual_condition.config',return_value=dict(base_url='',model='',api_key='')):
            agent=core.Agent();response=agent.ask('s1a0-2','检测整条记录','agentic')
            result=response['result'];self.assertEqual(result['version'],'3.0-experimental');self.assertFalse(result['annotations_used'])
            if result['candidates']:
                follow=agent.ask('s1a0-2','解释D01','agentic');self.assertIn('agentic_segments',follow['result'])

if __name__=='__main__':unittest.main()
