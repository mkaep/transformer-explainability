import math

import numpy as np
import pandas as pd


# https://stats.stackexchange.com/questions/217466/for-model-selection-comparison-what-kind-of-test-should-i-use


class CorrectedRepeatedKFoldCVTest:

    # r = repetitions
    # k = n_folds

    def calculate_mean(self, matrix: np.ndarray, k: int, r: int) -> float:
        sum = 0.0
        for i in range(k):
            for j in range(r):
                sum = sum + matrix[i][j]
        return sum / (k * r)

    def calculate_variance(self, matrix: np.ndarray, m: float, k: int, r: int) -> float:
        sum = 0.0
        for i in range(k):
            for j in range(r):
                sum = sum + (matrix[i][j] - m) * (matrix[i][j] - m)
        return sum / (k * (r-1))

    def _get_value(self, i: int, j: int, approach: str, observations_df: pd.DataFrame) -> float:
        filtered_data_df = observations_df[observations_df.apply(lambda x: x['approach'] == approach and
                                           x['repetition'] == j and x['fold'] == i, axis=1)]
        assert len(filtered_data_df) == 1
        return filtered_data_df.iloc[0, -1]

    def calculate_t_value(self, r, k, observations_df) -> float:
        print(observations_df)
        assert len(observations_df['approach'].unique()) == 2

        degrees_of_freedom = r * k - 1
        approaches = observations_df['approach'].unique()

        matrix = np.zeros((k, r))

        for i in range(k):
            for j in range(r):
                matrix[i][j] = self._get_value(i, j, approaches[1], observations_df) - self._get_value(i, j, approaches[0], observations_df)

        m = self.calculate_mean(matrix, k, r)
        variance = self.calculate_variance(matrix, m, k, r)

        # Test zu Trainingsamples in einer Fold oder über alle Folds und Repetitions hinweg???
        correction_factor = 1 / (k * r) + ((k-1)/k)
        return m / math.sqrt(correction_factor * variance)








    #matrix = np.zeros((n_repetitions, n_folds))
    #for i in range(0, n_folds):
    #    for j in range(0, n_repetitions):
    #        matrix[i][j] = get_value(dataset, metric, approach_1, i, j, results_1_df) - \
    #                       get_value(dataset, metric, approach_2, i, j, results_2_df)

    #print(matrix)


    #sum = 0
    #for i in range(0, n_folds):
    #    for j in range(0, n_repetitions):
    #        sum = sum + matrix[i][j]

    #mean = 1 / (n_folds * n_repetitions) * sum

    #temp = 0
    #for i in range(0, n_folds):
    #    for j in range(0, n_repetitions):
    #        temp = temp + (matrix[i][j] - mean) * (matrix[i][j] - mean)

    #variance = 1 / (n_folds * (n_repetitions-1)) * temp