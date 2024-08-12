import typing
import os
import sys
import subprocess
import stat

from processtransformer.ml.core import model
from processtransformer.ml.persistence import json


class JobExecutor:
    JOBS_FILE = 'jobs.jsonl'
    JOBS_DONE_FILE = 'jobs_done.jsonl'
    JOBS_FAILED_FILE = 'jobs_failed.jsonl'
    JOBS_WAITING_FILE = 'jobs_waiting.jsonl'

    def __init__(self, jobs_directory: str, jobs: typing.List[model.Job] = None,  verbose: bool = False):
        self._jobs_directory = jobs_directory
        self._verbose = verbose
        self._jobs_done = []
        self._jobs_failed = []

        if jobs is not None:
            self._jobs_waiting = jobs
        else:
            print(f'Try to reload an execution state from {jobs_directory}')
            self.reload_execution_state()

    def run(self):
        jobs_waiting_old = self._jobs_waiting.copy()
        for job in jobs_waiting_old:
            success = self.run_job(job)

            if success is True:
                self._jobs_done.append(job)
            else:
                self._jobs_failed.append(job)
            self._jobs_waiting.remove(job)
            self.save_execution_state()

            if self._verbose:
                print(f'{len(self._jobs_done)} / {len(jobs_waiting_old)}, TO DO: {len(self._jobs_waiting)}')

    def retry_failed_executions(self):
        jobs_failed_old = self._jobs_failed.copy()
        for job in jobs_failed_old:
            success = self.run_job(job)
            if success is True:
                self._jobs_failed.remove(job)
                self._jobs_done.append(job)
            self.save_execution_state()

    def reload_execution_state(self):
        job_files = os.listdir(self._jobs_directory)

        assert self.JOBS_FILE in job_files, f'Missing {self.JOBS_FILE} file'
        assert self.JOBS_DONE_FILE in job_files, f'Missing {self.JOBS_DONE_FILE} file'
        assert self.JOBS_FAILED_FILE in job_files, f'Missing {self.JOBS_FAILED_FILE} file'
        assert self.JOBS_WAITING_FILE in job_files, f'Missing {self.JOBS_WAITING_FILE} file'

        self._jobs_done = json.JsonJobImporter(os.path.join(self._jobs_directory, self.JOBS_DONE_FILE)).load()
        self._jobs_failed = json.JsonJobImporter(os.path.join(self._jobs_directory, self.JOBS_FAILED_FILE)).load()
        self._jobs_waiting = json.JsonJobImporter(os.path.join(self._jobs_directory, self.JOBS_WAITING_FILE)).load()

    def save_execution_state(self):
        json.JsonJobExporter(os.path.join(self._jobs_directory,
                                          self.JOBS_DONE_FILE)).save(self._jobs_done, self._verbose)
        json.JsonJobExporter(os.path.join(self._jobs_directory,
                                          self.JOBS_FAILED_FILE)).save(self._jobs_failed, self._verbose)
        json.JsonJobExporter(os.path.join(self._jobs_directory,
                                          self.JOBS_WAITING_FILE)).save(self._jobs_waiting, self._verbose)

    def run_job(self, job: model.Job) -> bool:
        if self._verbose:
            print(f'Run Job...')

        command = job.get_conda_command()
        # Create .bat file for calling the approach, necessary due to a bug in conda:
        # https://github.com/ContinuumIO/anaconda-issues/issues/12876
        # Hence, we change during the execution to the base environment and process than the conda run command
        temp_dir = os.path.join(self._jobs_directory, '.tmp')
        os.makedirs(temp_dir, exist_ok=True)
        with open(os.path.join(temp_dir, 'run_job.bat'), 'w', encoding='utf8') as f:
            f.write('call activate base\n')
            f.write(command)

        st = os.stat(os.path.join(temp_dir, 'run_job.bat'))
        os.chmod(os.path.join(temp_dir, 'run_job.bat'), st.st_mode | stat.S_IEXEC)

        try:
            subprocess.run(os.path.join(temp_dir, 'run_job.bat'), shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f'Error: Process could not be executed. Please check the command: {command}!')
            print(f'Job failed... {e.output} bla, {sys.executable}')
            return False
        print(sys.executable)

        if self._verbose:
            print(f'Finished Job successfully...')
        return True
