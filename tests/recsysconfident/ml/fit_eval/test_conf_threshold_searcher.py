import unittest
from unittest.mock import Mock
import pandas as pd
import numpy as np

from recsysconfident.ml.eval.conf_threshold_searcher import find_best_conf_threshold


class TestFindBestConfThreshold(unittest.TestCase):


    def ndcg_calculator(self, df: pd.DataFrame, k: int, c_t: float):

        if 0.8 <= c_t <= 0.82:
            return np.array([0.97, 0.3, 0.3, 0.3]), np.array([0.001, 0.1, 0.1, 0.1])
        else:
            return np.array([0.91, 0.2, 0.2, 0.2]), np.array([0.002, 0.01, 0.01, 0.01])

    def test_find_best_conf_threshold_with_mock(self):
        # Ensure 0.76 is within the quantile(0.65, 0.99) range
        conf_scores = np.linspace(0.0, 1.0, 101)  # 101 values between 0 and 1
        df_true_pred = pd.DataFrame({'conf_pred': conf_scores})
        k = 5

        mock_ndcd_calculator = Mock()
        mock_ndcd_calculator.side_effect = self.ndcg_calculator

        (
            best_quantile,
            best_conf_mean,
            best_conf_std,
            best_conf_threshold,
        ) = find_best_conf_threshold(df_true_pred, mock_ndcd_calculator, k)

        self.assertAlmostEqual(round(best_conf_mean[0], 2), 0.97)
        self.assertAlmostEqual(round(best_conf_std[0], 3), 0.001)
