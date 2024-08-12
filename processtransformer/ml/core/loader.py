

import pm4py
from zipfile import ZipFile
import gzip
import os
from pm4py.objects.log.obj import EventLog


class Loader:
    SUPPORTED_FILE_EXTENSIONS = {'.xes', '.zip', '.gz'}

    @classmethod
    def can_import_from(cls, file_path: str) -> bool:
        if len(file_path) == 0 or os.path.exists(file_path) is not True:
            return False
        extension = os.path.splitext(file_path)[-1].lower()
        if extension not in Loader.SUPPORTED_FILE_EXTENSIONS:
            return False
        return True

    @staticmethod
    def _load_from_zip_archive(file_path: str) -> EventLog:
        with ZipFile(file_path, 'r') as zipObj:
            content = zipObj.namelist()
            if len(content) == 1:
                if os.path.splitext(content[0])[-1].lower() == '.xes':
                    zipObj.extractall(os.path.dirname(file_path))
                    return pm4py.read_xes(os.path.join(os.path.dirname(file_path), zipObj.namelist()[0]))
                raise ValueError(f'Archive {file_path} does not contain a .xes file')
            else:
                raise ValueError(f'The archive {file_path} contains no or more than one event log')

    @staticmethod
    def _load_from_gz_archive(file_path: str) -> EventLog:
        splitted = file_path.split('.gz')

        if len(splitted[0].split('.xes')) == 2:
            with gzip.open(file_path, 'rb') as file_in:
                with open(splitted[0], 'w') as file_out:
                    for line in file_in:
                        file_out.write(line.decode('utf-8'))
            return pm4py.read_xes(splitted[0])
        else:
            raise ValueError(f'Archive {file_path} does not contain a .xes file')

    @staticmethod
    def load_event_log(file_path: str, verbose=False) -> EventLog:
        if verbose:
            print(f' Load event log from {file_path}')
        if Loader.can_import_from(file_path):
            extension = os.path.splitext(file_path)[-1].lower()
            if extension == '.xes':
                return pm4py.read_xes(file_path, return_legacy_log_object=True)
            elif extension == '.zip':
                return Loader._load_from_zip_archive(file_path)
            else:
                return Loader._load_from_gz_archive(file_path)

        raise NotImplementedError(f'Could not load event log from given file {file_path}')
