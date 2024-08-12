import os
import typing
from processtransformer.ml.persistence import json
from processtransformer.ml.core import model


def is_finished(job_dir: str, ignore_failed: bool = True) -> bool:
    """
    :param ignore_failed:
    :param job_dir:
    :return: True if there are no waiting jobs
    """
    importer_waiting_jobs = json.JsonJobImporter(os.path.join(job_dir, 'jobs_waiting.jsonl'))
    waiting_jobs = importer_waiting_jobs.load()

    failed_jobs = []
    if ignore_failed is False:
        importer_failed_jobs = json.JsonJobImporter(os.path.join(job_dir, 'jobs_failed.jsonl'))
        failed_jobs = importer_failed_jobs.load()

    return len(waiting_jobs) == 0 and len(failed_jobs) == 0


def approach_failed_on_dataset(approach: str, dataset: str, jobs: typing.List[model.Job]) -> bool:
    for job in jobs:
        if job.dataset == dataset and job.approach.name == approach:
            return True
    return False


def load_failed_jobs(experiment_dir: str) -> typing.List[model.Job]:
    importer_failed_jobs = json.JsonJobImporter(os.path.join(experiment_dir, '.jobs', 'jobs_failed.jsonl'))
    return importer_failed_jobs.load()
