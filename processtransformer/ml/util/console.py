import click
import typing
from pm4py.objects.log.obj import Trace


def trace_to_string(trace: Trace) -> str:
    str_repr = '<'
    for i, event in enumerate(trace):
        if i == len(trace) - 1:
            str_repr = str_repr + f'{event["concept:name"]}, {event["time:timestamp"]}'
        else:
            str_repr = str_repr + f'{event["concept:name"]}, {event["time:timestamp"]} | '
    return str_repr + '>'


class EventLogsDefinitionsParam(click.ParamType):
    name = 'splits'
    SEPERATOR_TOKEN = '#'

    def convert(self, value: typing.Any, param: typing.Optional[click.Parameter],
                ctx: typing.Optional[click.Context]) -> typing.Any:
        if type(value) is str:
            splits: typing.Dict[str, str] = {}
            raw_splits = value.split(';')
            for raw_split in raw_splits:
                if ':' not in raw_split:
                    self.fail(f'Missing : in event log definition "{raw_split}". Expected format "<name>:<path>"')
                values = raw_split.split(self.SEPERATOR_TOKEN)
                if len(values) != 2:
                    self.fail(f'Raw event log definition "{raw_split}" has an unsupported format, '
                              f'expected "<name>:<path>"')
                name, path = values
                try:
                    splits[name] = str(path)
                except ValueError:
                    self.fail(f'Expected path in split definition "{raw_split}" '
                              f'to be a str, but could not parse it.')
            return splits

        if type(value) is dict:
            # already parsed
            return value

        self.fail(f'Unsupported type "{type(value)}".')
