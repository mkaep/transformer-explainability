import tqdm
import click
import typing
import pandas as pd
from processtransformer.ml.evaluate.significance import stuart_maxwell
from processtransformer.ml.analysis import event_log_analysis
from processtransformer.ml.core import model, loader
from processtransformer.ml.pipeline import augmentation_pipeline, job_executor, classic_pipeline
from processtransformer.ml.persistence import json, tex
from processtransformer.ml.util import console
from processtransformer.ml.evaluate import evaluation


@click.group()
@click.option('--profile', '-p', is_flag=True, help='If set the command will be profiled and stored in ./profiles')
def cli(profile: bool = False):
    click.echo(profile)


@cli.command('complete', short_help='complete a job by executing waiting jobs and failed jobs again')
@click.argument('job-directory')
@click.option('-retry-failed', '--retry-failed',
              help='If set the program tries to execute failed jobs again',
              default=False, is_flag=True)
@click.option('-v', '--verbose',
              help='If set this program will be more verbose, i.e. commenting steps that are processed.',
              default=False, is_flag=True)
def complete_jobs(job_directory: str, retry_failed: bool = False, verbose: bool = False):
    """
    Load jobs, jobs_done, jobs_failed, jobs_queued and completes the experiment
    :param verbose: prints the progress to console
    :param retry_failed: indicates whether failed jobs should be run again
    :param job_directory: directory with the job files
    :return: Nothing
    """
    executor = job_executor.JobExecutor(job_directory, verbose=verbose)
    executor.run()

    if retry_failed is True:
        executor.retry_failed_executions()

def augmentation_dependency(experiment_dir: str, strategies: typing.List[str], datasets: typing.List[str],
                        approaches: typing.List[str], metric: str, aggregate_on: str = 'run', num_precision: int = 3,
                        target_file: str = None):

    #result = evaluation.evaluate_prefix_total(experiment_dir, strategies, datasets,
    #                                            approaches, metric, aggregate_on=aggregate_on,
    #                                            n_precision=num_precision,
    #                                            target_file=target_file)

    result = evaluation.augmentation_dependency(experiment_dir, strategies, datasets,
                                                                       approaches, metric, aggregate_on=aggregate_on,
                                                                       n_precision=num_precision,
                                                                       target_file=target_file)
    print(result)



def create_table(experiment_dir: str, strategies: typing.List[str], datasets: typing.List[str],
                        approaches: typing.List[str], metric: str, aggregate_on: str = 'run', num_precision: int = 3,
                        target_file: str = None):
    result = evaluation.get_table(experiment_dir, strategies, datasets,
                                                                       approaches, metric, aggregate_on=aggregate_on,
                                                                       n_precision=num_precision,
                                                                       target_file=target_file)
    click.echo(result)


def training_procedure_analysis(experiment_dir: str, strategies: typing.List[str], datasets: typing.List[str],
                        approaches: typing.List[str], metric: str, aggregate_on: str = 'run', num_precision: int = 3,
                        target_file: str = None):
    result = evaluation.training_procedure(experiment_dir, strategies, datasets,
                                                                       approaches, metric, aggregate_on=aggregate_on,
                                                                       n_precision=num_precision,
                                                                       target_file=target_file)
    click.echo(result)


def evaluate_experiment(experiment_dir: str, strategies: typing.List[str], datasets: typing.List[str],
                        approaches: typing.List[str], metric: str, aggregate_on: str = 'run', num_precision: int = 3,
                        target_file: str = None):

    result = evaluation.evaluate_strategies_on_datasets_and_approaches(experiment_dir, strategies, datasets,
                                                                       approaches, metric, aggregate_on=aggregate_on,
                                                                       n_precision=num_precision,
                                                                       target_file=target_file)

    with open('D:\\resultttt.txt', 'w', encoding='utf8') as f:
        f.write(result.to_latex())
    click.echo(result)


def evaluate_strategies_on_dataset_on_approach_detailed(experiment_dir: str, dataset: str, approach: str,
                                                        strategies: typing.List[str], metric: str):
    print(strategies)
    evaluation.evaluate_strategies_on_dataset_and_approach(experiment_dir, dataset, approach, strategies, metric)


@cli.command('evaluate-architecture', help='Extracts statistics about the training procedure and the used machine '
                                           'learning models')
@click.argument('experiment_dir')
@click.argument('strategies', nargs=-1)
@click.option('-aggregate_on', default='run', type=click.Choice(['run', 'fold', 'repetition']),
              help='Defines the aggregate level of the analysis')
@click.option('-num_precision', '--num_precision', type=int, default=2, required=False, help='Number of digits')
@click.option('-target_file', '--target_file', required=False)
def evaluate_architecture(experiment_dir: str, strategies: typing.List[str], aggregate_on: str = 'run',
                          num_precision: int = 2,
                          target_file: str = None):
    result = evaluation.evaluate_architecture(experiment_dir, strategies, aggregate_on=aggregate_on,
                                              n_precision=num_precision, target_file=target_file)
    click.echo(result)


@cli.command('evaluate-gain', help='Determines the gain of augmentation for different strategies')
@click.argument('experiment_dir')
@click.argument('strategies', nargs=-1)
@click.option('-d', '--datasets', multiple=True)
@click.option('-a', '--approaches', multiple=True)
@click.option('-m', '--metric_name', type=str, default='Accuracy', required=True)
@click.option('-aggregate_on', default='run', type=click.Choice(['run', 'fold', 'repetition']),
              help='Defines the aggregate level of the analysis')
@click.option('-num_precision', '--num_precision', type=int, default=2, required=False, help='Number of digits')
@click.option('-target_file', '--target_file', required=False)
def evaluate_gain(experiment_dir: str, strategies: typing.List[str], datasets: typing.List[str],
                  approaches: typing.List[str], metric_name: str = 'Accuracy', aggregate_on: str = 'run',
                  num_precision: int = 2,
                  target_file: str = None):
    result = evaluation.evaluate_gain_of_strategies_on_datasets_and_approaches(experiment_dir, list(strategies),
                                                                               list(datasets),
                                                                               list(approaches), metric_name,
                                                                               aggregate_on=aggregate_on,
                                                                               n_precision=num_precision,
                                                                               target_file=target_file)
    click.echo(result)


# todo in case of repetition > 1 we also need the aggregations
def check_significance(experiment_dir: str, strategies: typing.List[str], datasets: typing.List[str],
                       approaches: typing.List[str], num_precision: int = 2, target_file: str = None):
    result = evaluation.calculcate_significance_of_strategies_on_datasets_and_approaches(experiment_dir, strategies,
                                                                                         datasets, approaches,
                                                                                         num_precision, target_file)
    click.echo(result)


@cli.command('run-experiment', short_help='runs a defined experiment')
@click.argument('experiment_file', type=str)
@click.option('-v', '--verbose',
              help='If set this program will be more verbose, i.e. commenting steps that are processed.',
              default=False, is_flag=True)
def run_experiment(experiment_file: str, verbose: bool = False):
    experiment = json.JsonExperimentImporter(experiment_file).load(verbose)

    if isinstance(experiment, model.AugmentationExperiment):
        augmentation_pipeline.run_pipeline(experiment, verbose)
    else:
        classic_pipeline.run_pipeline(experiment, verbose)



def get_event_log_characteristics(event_logs: typing.Dict[str, str], stats_names: typing.List[str],
                                  clear_names: bool = True, orientation: str = 'columns', num_precision: int = 2,
                                  verbose: bool = False, target_file: str = None):
    assert orientation in {'row', 'columns'}
    assert len(event_logs.keys()) > 0
    assert num_precision > 0

    print(event_logs)

    stats: typing.Dict[str, typing.Dict] = dict()
    event_log_names = event_logs.keys()
    if verbose:
        event_log_names = tqdm.tqdm(event_logs.keys())
    for name in event_log_names:
        event_log = loader.Loader().load_event_log(event_logs[name])
        stats[name] = event_log_analysis.EventLogDescriptor(stats_names).run_analysis(event_log)

    if target_file is not None:
        tex.export_event_log_statistics_to_tex_file(stats, target_file, orientation, clear_names, num_precision)
    else:
        click.echo(stats)


def evaluate_correlations(experiment_dir: str, strategies: typing.List[str], datasets: typing.List[str],
                          approaches: typing.List[str], x_property: str = '', y_property: str = '',
                          aggregate_on: str = 'run',
                          target_file: str = None):
    supported_properties = ['']
    assert x_property in supported_properties, f'Currently only {supported_properties} are supported x_properties'
    assert y_property in supported_properties, f'Currently only {supported_properties} are supported y_properties'

    result = evaluation.evaluate_correlations(experiment_dir, strategies, datasets, approaches, x_property, y_property,
                                              aggregate_on, target_file=target_file, x_label='# Trainable parameters',
                                              y_label='Accuracy Gain in %', x_scale='log')

    click.echo(result)


if __name__ == '__main__':
    #datasets = {
    #    'Helpdesk': r'../data/Helpdesk.xes',
    #    'Sepsis': r'../data/Sepsis.xes',
    #    'BPIC13_closed': r'../data/BPI_Challenge_2013_closed_problems.xes',
    #    'BPIC13_incidents': r'../data/BPI_Challenge_2013_incidents.xes',
    #    'BPIC15_1': r'../data/BPIC15_1.xes',
    #    'BPIC12': r'../data/BPI_Challenge_2012.xes',
    #    'NASA': r'../data/NASA.xes',
    #}
    #get_event_log_characteristics(datasets, ['number_cases',
    #                                         'number_events',
    #                                         'number_activities',
    #                                         'avg_case_length',
    #                                         'max_case_length',
    #                                         'min_case_length',
    #                                         'avg_event_duration',
    #                                         'max_event_duration',
    #                                         'min_event_duration',
    #                                         'avg_case_duration',
    #                                         'max_case_duration',
    #                                         'min_case_duration',
    #                                         'number_variants'], target_file='D:\\CHAR.txt', orientation='row')
    # executor = job_executor.JobExecutor(r'/home/ai4-admin/runs/aug_study/.jobs', verbose=True)
    # executor.run()
    # executor.retry_failed_executions()
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    #datasets = ['Helpdesk']
    #datasets = ['Helpdesk']
    datasets = ['Sepsis', 'BPIC13_closed', 'BPIC13_incidents', 'Helpdesk', 'BPIC12', 'NASA', 'BPIC15_1']
    #datasets = ['Sepsis', 'Helpdesk']
    #datasets = ['BPIC12']
    #approaches = ['Mauro']
    approaches = ['Buksh', 'Camargo', 'Theis', 'Mauro', 'Tax', 'Pasquadibisceglie', 'Khan']
    #approaches = ['Buksh']
    strategies = ['base', '1_mixed_True_1.2', '2_mixed_True_1.4', '3_mixed_True_1.6', '4_mixed_True_2',
                  '5_mixed_True_2.4', '6_mixed_True_2.6', '7_mixed_True_3']
    experiment_dir = r'D:\Research_Results\aug_study_3'
    metric = 'Accuracy'
    # check_significance(experiment_dir, strategies, datasets, approaches, 2, None)

    # importer = json.JsonExperimentImporter(r'/home/ai4-admin/runs/exp_small.json')
    # loaded_experiment = importer.load()
    # augmentation_pipeline.run_pipeline(loaded_experiment, True)
    # print('ready')

    evaluate_correlations(experiment_dir, strategies, datasets, approaches, '', '', 'fold')
    #print(evaluation.evaluate_gain_of_strategies_on_datasets_and_approaches(experiment_dir, strategies, datasets, approaches, metric))
    #strategies_base = ['base']
    #augmentation_dependency(experiment_dir, strategies, datasets, approaches, metric)

    #create_table(experiment_dir, strategies, datasets, approaches, metric)
    #training_procedure_analysis(experiment_dir, strategies, datasets, approaches, metric)
    #evaluate_experiment(experiment_dir, strategies, datasets, approaches, metric)
    #check_significance(experiment_dir, strategies, datasets, approaches)

    #stuart_maxwell.perform_significance_test(r'D:\runs_8\compStudy\Helpdesk\Camargo\rep_0\fold_0\base\result.csv', r'D:\runs_8\compStudy\Helpdesk\Camargo\rep_0\fold_0\1_mixed_True_1.2\result.csv')
    # cli()
    #evaluate_strategies_on_dataset_on_approach_detailed(experiment_dir, 'Helpdesk', 'Camargo', strategies, metric)
    # evaluate_architecture(r'D:\runs_8\compStudy')
    # architecture_evaluation.evaluate_training_time(r'D:\runs_8\compStudy')

    # compare_event_log_statistics(r'D:\runs_8\compStudy', 'Helpdesk', ['base', '1_mixed_True_1.2', '2_mixed_True_1.4', '3_mixed_True_1.6', '3_mixed_True_2'],
    #                            0, 0, event_log_analysis.EventLogDescriptor.get_available_steps(), 2, target_dir='D:\\')
    # df = evaluate_experiment(r'D:\runs_8\compStudy', ['base', '1_mixed_True_1.2', '2_mixed_True_1.4', '3_mixed_True_1.6', '3_mixed_True_2'], ['Helpdesk', 'Sepsis', 'BPIC12', 'BPIC13', 'BPIC15_1'], ['Buksh', 'Camargo', 'Pasquadibisceglie', 'Theis'])
    # print(df)
