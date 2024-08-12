from processtransformer.ml.evaluate import ev_model
import typing


def determine_repetitions_and_folds(evaluation_model: ev_model.EvaluationModel) -> typing.Tuple[int, int]:
    assert len(evaluation_model.runs) > 0, f'Evaluation model contains no runs for evaluation'
    current_n_repetitions = len(evaluation_model.runs[0].repetitions)

    assert len(evaluation_model.runs[0].repetitions) > 0, f'Error: Run without repetition'
    current_n_folds = len(evaluation_model.runs[0].repetitions[0].folds)

    for run in evaluation_model.runs:
        if len(run.repetitions) != current_n_repetitions:
            raise ValueError(f'Error: Inconsistent experiment directory. Different number of repetitions were '
                             f'found: {len(run.repetitions)} and {current_n_repetitions}')
        else:
            current_n_repetitions = len(run.repetitions)
        for fold in run.repetitions:
            if len(fold.folds) != current_n_folds:
                raise ValueError(f'Error: Inconsistent experiment directory. Different number of folds for repetitions '
                                 f'were found: {len(fold.folds)} and {current_n_folds})')
            else:
                current_n_folds = len(fold.folds)

    return current_n_repetitions, current_n_folds


def guess_significance_test_from_experiment_dir(evaluation_model: ev_model.EvaluationModel) -> typing.List[str]:
    num_repetitions, num_folds = determine_repetitions_and_folds(evaluation_model)

    if num_repetitions == 1 and num_folds == 1:
        return ['StuartMaxwell, McNemarBowker']
    elif num_repetitions > 1 and num_folds == 1:
        return []
    elif num_repetitions > 1 and num_folds > 1:
        return ['Modified-t-Test with Bengio correctur']
    else:
        raise NotImplementedError('Could not find an appropriate significance test')


def interpret_p_value_variant_1(p_value: float) -> str:
    if p_value > 0.10:
        return 'Weak or no evidence'
    elif 0.05 < p_value <= 0.10:
        return 'Moderate evidence'
    elif 0.01 < p_value <= 0.05:
        return 'Strong evidence'
    elif p_value <= 0.01:
        return 'Very strong evidence'
    else:
        raise ValueError('Should not happen')


# https://www.geo.fu-berlin.de/en/v/soga/Basics-of-statistics/Hypothesis-Tests/Introduction-to-Hypothesis-Testing/Critical-Value-and-the-p-Value-Approach/index.html
#interpret_p_value(p_value)
def interpret_p_value_variant_2(p_value: float) -> str:
    if p_value > 0.05:
        return 'Null hypothesis can not be rejected. There is not significant difference!'
    return 'Null hypothesis rejected. There is significant difference.'
