import unittest
from unittest.mock import patch
import numpy as np
from adaptive_signal import condition_features,paper_filter,zero_phase_highpass,spans,apply_adaptive,ALGORITHMS
from core import Agent
from cloud_agent import CloudAgent

def synthetic(kind,n=1024):
    rng=np.random.default_rng(47);t=np.arange(n)[:,None];c=np.arange(32)[None,:]
    x=rng.normal(0,.02,(3,n,32))
    if kind=='strand':x+=np.sin(2*np.pi*(t/64+6*c/32))[None,:,:]
    if kind=='shake':x+=3*np.exp(-((t-450)/50)**2)[None,:,:]*np.cos(2*np.pi*c/32)[None,:,:]
    # Local bipolar flaw, including weak-amplitude case under periodic texture.
    distance=np.minimum(np.abs(c-12),32-np.abs(c-12))
    flaw=.5*((t-650)/8)*np.exp(-((t-650)/8)**2)*np.exp(-(distance/1.5)**2)
    x+=flaw[None,:,:]
    return x

class AdaptiveTests(unittest.TestCase):
    def test_condition_routes_differ_with_controlled_noise(self):
        self.assertEqual(condition_features(synthetic('quiet'))['recommended_algorithm'],'baseline')
        self.assertEqual(condition_features(synthetic('strand'))['recommended_algorithm'],'paper2021')
        self.assertEqual(condition_features(synthetic('shake'))['recommended_algorithm'],'paper2023')

    def test_eq5_frequency_response(self):
        t=np.arange(3000);w=20;frequency=.017
        x=np.sin(2*np.pi*frequency*t)
        out=zero_phase_highpass(x,w)
        b=np.full(w,-1/w);b[0]+=1
        expected=abs(np.sum(b*np.exp(-2j*np.pi*frequency*np.arange(w))))**2
        ratio=np.linalg.norm(out[500:-500])/np.linalg.norm(x[500:-500])
        self.assertAlmostEqual(ratio,expected,places=3)

    def test_directional_filter_removes_strand_preserves_weak_local_response(self):
        z=synthetic('strand');f=condition_features(z)
        out,p,ops=paper_filter(z[0],'paper2021',f)
        self.assertEqual(out.shape,z[0].shape)
        self.assertTrue(np.isfinite(out).all())
        self.assertGreater(float(np.max(out[625:675,8:17])),float(np.quantile(out[150:450],.99))*2)
        self.assertIn('测量斜纹方向→剪切重采样→零相位滤波→反剪切',ops)

    def test_morphology_has_positive_residual_and_original_coordinates(self):
        z=synthetic('shake',768);f=condition_features(z)
        out,p,ops=paper_filter(z[0],'paper2023',f)
        self.assertEqual(out.shape,z[0].shape)
        self.assertGreater(np.max(out[625:675,8:17]),0)
        self.assertTrue(np.all(out>=0))
        self.assertEqual(p['line_SE'],120)
        self.assertEqual(p['top_hat_sign'],'closed_image - opened_background (Fig.12)')

    def test_plan_validation_and_inspection_precondition(self):
        z=synthetic('quiet',768);raw=np.transpose(z,(1,0,2))
        conditions={'condition_segments':[dict(condition_features(z),segment_id='S01',sample_range=[0,767])]}
        with self.assertRaises(ValueError):apply_adaptive(raw,conditions,routes=[dict(segment_id='missing',algorithm='paper2023')])
        with self.assertRaises(ValueError):Agent().call('adaptive_detect','s1a0-1')
        self.assertEqual(spans(2100),[[0,2099]])

    def test_inference_never_reads_annotations(self):
        from adaptive_signal import inspect_conditions
        import pandas as pd
        original=pd.read_csv
        def guarded(path,*args,**kwargs):
            if 'annotations' in str(path):raise AssertionError('Inference attempted annotation access')
            return original(path,*args,**kwargs)
        with patch('pandas.read_csv',side_effect=guarded):
            result=inspect_conditions('s1a0-2')
        self.assertFalse(result['annotations_used'])

    def test_preservation_failure_falls_back_and_is_recorded(self):
        z=synthetic('quiet',768);raw=np.transpose(z,(1,0,2))
        conditions={'condition_segments':[dict(condition_features(z),segment_id='S01',sample_range=[0,767])]}
        with patch('adaptive_signal.preservation_guard',return_value=dict(passes=False,minimum_retention=0,landmark_count=1)):
            out,logs,*_=apply_adaptive(raw,conditions,strategy='paper2023')
        self.assertEqual(logs[0]['used'],'baseline')
        self.assertTrue(logs[0]['fallback_reason'])
        self.assertTrue(np.isfinite(out).all())

    def test_model_can_repair_invalid_route_parameter(self):
        replies=iter([
            dict(role='assistant',tool_calls=[dict(id='a',function=dict(name='adaptive_detect',arguments={'strategy':'routes'}))]),
            dict(role='assistant',tool_calls=[dict(id='b',function=dict(name='adaptive_detect',arguments={'strategy':'auto'}))]),
            dict(role='assistant',content='按修正参数完成。')])
        agent=CloudAgent(dict(model='mock',base_url='https://example.com',api_key='test'),lambda messages,tools:next(replies))
        with patch.object(agent,'call',side_effect=[ValueError('未知自适应策略'),dict(candidates=[])]):
            response=agent.ask('s1a0-1','自适应检测','adaptive')
        self.assertIn('candidates',response['result'])
        self.assertEqual(response['trace'][0]['status'],'rejected')

if __name__=='__main__':unittest.main()
