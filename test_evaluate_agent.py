import unittest
from evaluate_agent import event_metrics,point_metrics,ratios,annotations_blocked,aggregate
from unittest.mock import patch

class EvaluationTests(unittest.TestCase):
    def test_inference_cannot_open_gt_csv(self):
        import pandas as pd
        with annotations_blocked():
            with self.assertRaises(AssertionError):pd.read_csv('dataset/annotations/example.csv')
    def test_no_successful_records_not_perfect_accuracy(self):
        result=aggregate([])
        self.assertEqual(result['records'],0);self.assertIsNone(result['points']['accuracy'])
        self.assertIsNone(result['events']['0.5']['f1'])
    def test_one_gt_cannot_count_multiple_times(self):
        result=event_metrics([(10,19),(10,19)],[(10,19)])
        self.assertEqual((result['tp'],result['fp'],result['fn']),(1,1,0))
    def test_maximum_matching_avoids_greedy_loss(self):
        result=event_metrics([(0,9),(0,4)],[(0,4),(5,9)],.3)
        self.assertEqual(result['tp'],2)
    def test_disjoint_and_empty_predictions(self):
        result=event_metrics([],[(10,19)])
        self.assertEqual(result['fn'],1);self.assertEqual(result['recall'],0);self.assertIsNone(result['precision'])
        self.assertEqual(event_metrics([(0,4)],[(10,19)])['tp'],0)
    def test_inclusive_sample_confusion(self):
        result=point_metrics([(2,5)],[(4,7)],10)
        self.assertEqual((result['tp'],result['fp'],result['fn'],result['tn']),(2,2,2,4))
        self.assertAlmostEqual(result['accuracy'],.6)
    def test_union_not_duplicate_point_counts(self):
        result=point_metrics([(2,5),(4,7)],[(2,7)],10)
        self.assertEqual(result['tp'],6);self.assertEqual(result['tn'],4)
    def test_empty_denominators_are_undefined(self):
        self.assertIsNone(ratios(0,0,0)['f1'])
    def test_bad_label_boundaries_rejected(self):
        with self.assertRaises(ValueError):point_metrics([],[(0,10)],10)

if __name__=='__main__':unittest.main()
