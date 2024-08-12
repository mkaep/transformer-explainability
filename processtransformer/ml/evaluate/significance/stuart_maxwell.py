import typing
import pandas as pd
import numpy as np
from processtransformer.ml.core import model
from scipy import stats
import sympy

# Implemented according to http://www.john-uebersax.com/stat/mcnemar.htm#stuart
# and https://search.r-project.org/CRAN/refmans/DescTools/html/StuartMaxwellTest.html


def combine_results(results_first_classifier: str, results_second_classifier: str) -> pd.DataFrame:
    # Load result function from fold_evaluation
    results_first_classifier_df = pd.read_csv(results_first_classifier, sep='\t')
    results_second_classifier_df = pd.read_csv(results_second_classifier, sep='\t')

    assert 'id' in results_first_classifier_df.columns \
           and model.PredictionTask.NEXT_ACTIVITY.value in results_first_classifier_df.columns
    results_first_classifier_df = results_first_classifier_df[['id', model.PredictionTask.NEXT_ACTIVITY.value]]
    results_first_classifier_df.columns = ['id', 'base']

    assert 'id' in results_second_classifier_df.columns \
           and model.PredictionTask.NEXT_ACTIVITY.value in results_second_classifier_df.columns
    results_second_classifier_df = results_second_classifier_df[['id', model.PredictionTask.NEXT_ACTIVITY.value]]
    results_second_classifier_df.columns = ['id', 'aug']

    return results_first_classifier_df.merge(results_second_classifier_df, how='inner', on='id')


def get_i_j_value(i: int, j: int, index_activity: typing.Dict[int, str], observations_df: pd.DataFrame) -> int:
    assert len(observations_df) > 0
    assert {'base', 'aug'}.issubset(observations_df.columns), f'Observation expect columns base and aug'
    clazz_i = index_activity[i]
    clazz_j = index_activity[j]

    first_filtered_df = observations_df[observations_df['base'] == clazz_i]
    second_filtered_df = first_filtered_df[first_filtered_df['aug'] == clazz_j]

    return len(second_filtered_df)


def build_contingency_table(observations_df: pd.DataFrame, verify_with_lib: bool = False) -> typing.Tuple[int, np.ndarray]:
    assert {'base', 'aug'}.issubset(observations_df.columns), f'Observation expect columns base and aug'

    classes = list(observations_df['base'].unique())
    classes.extend(list(observations_df['aug'].unique()))
    classes = set(classes)

    n_classes = len(classes)

    degree_of_freedoms = n_classes - 1

    matrix = np.zeros((n_classes, n_classes))
    index_activity = {i: clazz for i, clazz in enumerate(classes, start=0)}
    for i in index_activity.keys():
        for j in index_activity.keys():
            matrix[i][j] = get_i_j_value(i, j, index_activity, observations_df)

    if verify_with_lib is True:
        print(pd.crosstab(observations_df['base'], observations_df['aug']))

    return degree_of_freedoms, matrix


def remove_columns_with_perfect_agreement(matrix: np.ndarray) -> np.ndarray:
    while len(detect_perfect_agreement(matrix)) > 0:
        l = detect_perfect_agreement(matrix)
        matrix = np.delete(matrix, l[0], 0)  # delete row (axis = 0)
        matrix = np.delete(matrix, l[0], 1)  # delete column (axis = 1)
    return matrix


def perform_significance_test(results_first_classifier: str, results_second_classifier: str) -> typing.Tuple[float, int, float]:
    observations_df = combine_results(results_first_classifier, results_second_classifier)
    degrees_of_freedom, contingency_table = build_contingency_table(observations_df, verify_with_lib=False)

    # If there is perfect agreement for any category k, that category must be omitted in order to invert matrix S.
    contingency_table = remove_columns_with_perfect_agreement(contingency_table)

    # Calculate p-value
    chi_statistics = calculate_chi_square(contingency_table)
    p_value = stats.chi2.sf(chi_statistics, degrees_of_freedom)
    # 1 - stats.chi2.cdf(chi_statistics, degrees_of_freedom)

    return chi_statistics, degrees_of_freedom, p_value


def detect_perfect_agreement(matrix: np.ndarray):
    n_rows, n_cols = matrix.shape
    assert n_rows == n_cols
    to_remove = []

    for i in range(0, n_rows):
        if check(i, matrix) is True:
            to_remove.append(i)
    return to_remove


def check(i: int, matrix: np.ndarray):
    n_rows, n_cols = matrix.shape
    assert n_rows == n_cols
    perfect = True
    for j in range(n_cols):
        if i != j:
            if matrix[i][j] != 0 or matrix[j][i] != 0:
                perfect = False
                break
    return perfect


def calculate_chi_square(matrix: np.ndarray):
    dim = matrix.shape[0] - 1
    column_vectors = np.sum(matrix, axis=0)
    row_vectors = np.sum(matrix, axis=1)
    d = row_vectors - column_vectors
    # Select any K-1 values of d
    d = d[:-1]
    S = np.zeros((dim, dim))
    for i in range(0, dim):
        for j in range(0, dim):
            if i == j:
                S[i][i] = row_vectors[i] + column_vectors[i] - 2 * matrix[i][i]
            else:
                S[i][j] = -(matrix[i][j] + matrix[j][i])

    # Remove linear dependent columns
    _, columns = sympy.Matrix(S).rref()

    S = np.take(S, columns, axis=0) # keep rows
    S = np.take(S, columns, axis=1) # keep columns
    d = np.take(d, columns, axis=0) # keep rows

    #print(np.array_str(S, precision=2, suppress_small=True))

    # Calculate the Stuart-Max-Statistic
    return (np.transpose(d).dot(np.linalg.inv(S))).dot(d)


def test_method():
    matrix = np.array([
        [20,10, 5],
        [3, 30, 15],
        [0, 5, 40]
    ])




