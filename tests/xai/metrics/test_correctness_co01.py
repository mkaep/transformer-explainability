

import unittest
from unittest.mock import patch

import numpy as np

from processtransformer.xai.metrics.correctness_co01 import eval_correctness


class TestCorrectness(unittest.TestCase):
    @patch('processtransformer.xai.metrics.correctness_co01.get_xai_feature_importance', autospec=True)
    @patch('processtransformer.xai.metrics.correctness_co01.get_blackbox_feature_importance', autospec=True)
    def test_eval_correctness1(self, mock_bb_fi, mock_xai_fi):
        mock_bb_fi.return_value = {'A': 1.0, 'B': 0.0, 'C': 0.0}, None
        mock_xai_fi.return_value = {'A': 1.0, 'B': 0.0, 'C': 0.0}, None
        metrics = eval_correctness(None, None, None, None, None, None)
        assert np.isclose(metrics.spearman_corr, 1.0, atol=0.001)
        assert np.isclose(metrics.kendalltau_corr, 1.0, atol=0.001)
        assert np.isclose(metrics.pearson_corr, 1.0, atol=0.001)
        assert np.isclose(list(metrics.df.iloc[0])[:3], [1.0, 1.0, 1.0], atol=0.001).all()

        mock_bb_fi.return_value = {'B': 0.5, 'A': 1.0, 'C': 0.0}, None
        mock_xai_fi.return_value = {'A': 1.0, 'B': 0.5, 'C': 0.0}, None
        metrics = eval_correctness(None, None, None, None, None, None)
        assert np.isclose(metrics.spearman_corr, 1.0, atol=0.001)
        assert np.isclose(metrics.kendalltau_corr, 1.0, atol=0.001)
        assert np.isclose(metrics.pearson_corr, 1.0, atol=0.001)

        mock_bb_fi.return_value = {'A': -1.0, 'B': -0.5, 'C': 0.0}, None
        mock_xai_fi.return_value = {'A': 1.0, 'B': 0.5, 'C': 0.0}, None
        metrics = eval_correctness(None, None, None, None, None, None)
        assert np.isclose(metrics.spearman_corr, -1.0, atol=0.001)
        assert np.isclose(metrics.kendalltau_corr, -1.0, atol=0.001)
        assert np.isclose(metrics.pearson_corr, -1.0, atol=0.001)

        mock_bb_fi.return_value = {'B': 0.5, 'A': 1.0, 'C': 0.0, 'D': 0.01}, None
        mock_xai_fi.return_value = {'A': 1.0, 'B': 0.5, 'D': 0.03, 'C': 0.0}, None
        metrics = eval_correctness(None, None, None, None, None, None)
        assert np.isclose(metrics.spearman_corr, 1.0, atol=0.001)
        assert np.isclose(metrics.kendalltau_corr, 1.0, atol=0.001)
        assert np.isclose(metrics.pearson_corr, 1.0, atol=0.001)

        mock_bb_fi.return_value = {'B': 0.5, 'A': 1.0, 'C': 0.0, 'D': 0.01}, None
        mock_xai_fi.return_value = {'A': 1.0, 'B': 0.5, 'C': 0.0, 'D': 0.00}, None
        metrics = eval_correctness(None, None, None, None, None, None)
        assert np.isclose(metrics.spearman_corr, 0.9486, atol=0.001)
        assert np.isclose(metrics.kendalltau_corr, 0.9129, atol=0.001)
        assert np.isclose(metrics.pearson_corr, 1.0, atol=0.001)

        mock_bb_fi.return_value = {'A': 0.5, 'B': 1.0, 'C': 0.0, 'D': 0.0}, None
        mock_xai_fi.return_value = {'A': 0.0, 'B': 0.0, 'C': 0.5, 'D': 1.0}, None
        metrics = eval_correctness(None, None, None, None, None, None)
        assert np.isclose(metrics.spearman_corr, -0.8889, atol=0.001)
        assert np.isclose(metrics.kendalltau_corr, -0.8, atol=0.001)
        assert np.isclose(metrics.pearson_corr, -0.8182, atol=0.001)

        mock_bb_fi.return_value = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}, None
        mock_xai_fi.return_value = {'A': 7, 'B': 1, 'C': 2, 'D': 6, 'E': 4, 'F': 5, 'G': 3}, None
        metrics = eval_correctness(None, None, None, None, None, None)
        assert np.isclose(metrics.spearman_corr, -0.0714, atol=0.001)
        assert np.isclose(metrics.kendalltau_corr, -0.0476, atol=0.001)
        assert np.isclose(metrics.pearson_corr, -0.0714, atol=0.001)
        pass


if __name__ == '__main__':
    unittest.main()
