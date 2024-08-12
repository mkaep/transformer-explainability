

import unittest

import numpy as np

from processtransformer.xai.attn_exploration_explainer import AttentionExplorationExplainer


class TestAttentionExplorationExplainer(unittest.TestCase):
    def test_transfer_local_to_global_score_dict(self):
        local_df = {
            'base_check':
                {'D': {'A': [0.0, 0.0, 1.0], 'C': [-1.0, 2.0, 1.0]},
                 'E': {'A': [-2.0, -2.0, 5.0], 'C': [-1.0, 80.0, 19.0]},
                 'F': {'A': [0, 0, 0, 0, 1], 'C': [-1, 1, 0, 0, 0, 0, 0, 10]},
                 }
        }
        score_dict = {
            'base_check':
                {'D': {'A': 0.0, 'C': 0.0},
                 'E': {'A': 0.0, 'C': 0.0},
                 'F': {'A': 0.0, 'C': 0.0},
                 }
        }
        AttentionExplorationExplainer.transfer_local_to_global_score_dict(local_df, score_dict, normal_to_abs_thr=0.8)
        self.assertEqual(np.isclose(score_dict['base_check']['D']['A'], 0.3333, rtol=1e-3), True)
        self.assertEqual(np.isclose(score_dict['base_check']['D']['C'], 0.0000, rtol=1e-3), True)
        self.assertEqual(np.isclose(score_dict['base_check']['E']['A'], 0.0000, rtol=1e-3), True)
        self.assertEqual(np.isclose(score_dict['base_check']['E']['C'], 0.3266, rtol=1e-3), True)
        self.assertEqual(np.isclose(score_dict['base_check']['F']['A'], 0.2000, rtol=1e-3), True)
        self.assertEqual(np.isclose(score_dict['base_check']['F']['C'], 0.1042, rtol=1e-3), True)


if __name__ == '__main__':
    unittest.main()
