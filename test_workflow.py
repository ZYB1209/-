import unittest
from workflow import DetectionWorkflow

class WorkflowTests(unittest.TestCase):
    def test_fast_branch_and_mapping(self):
        result=DetectionWorkflow().run('s1a0-1')
        workflow=result['workflow'];nodes={n['id']:n for n in workflow['nodes']}
        self.assertEqual(nodes['temporal_energy']['status'],'skipped')
        self.assertEqual(nodes['coordinate_mapping']['status'],'completed')
        self.assertFalse(workflow['decisions']['annotations_used'])
        self.assertFalse(workflow['decisions']['length_mapping'])
        self.assertTrue(all('position_m' not in c for c in result['candidates']))
    def test_supplementary_branch(self):
        result=DetectionWorkflow().run('s1a0-1','balanced')
        self.assertTrue(result['workflow']['decisions']['extra_detector'])
        self.assertEqual(set(result['tools']),{'temporal_channel','temporal_energy'})
    def test_invalid_scope_stops(self):
        with self.assertRaises(ValueError):DetectionWorkflow().run('s1a0-1',region=[0,999999])

if __name__=='__main__':unittest.main()
