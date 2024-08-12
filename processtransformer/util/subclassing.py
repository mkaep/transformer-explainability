

import abc
import typing

from processtransformer.xai.event_eval import Evaluator
from processtransformer.xai.explainer import Explainer


class BaseSubclasses(abc.ABC):
    base_class = None

    @classmethod
    @abc.abstractmethod
    def get_all_subclasses(cls):
        pass

    @staticmethod
    def get_subclasses_from_list(to_check: typing.List):
        all_subclasses = []
        while len(to_check) > 0:
            new_to_check = []

            # Go over all children found up to now
            for c in to_check:
                all_subclasses.append(c)
                # Add all of their children again
                for s in c.__subclasses__():
                    new_to_check.append(s)
            to_check = new_to_check

        return all_subclasses

    def get_type(self, name: str):
        for subclass in self.get_all_subclasses():
            if subclass.get_name() == name:
                return subclass

        raise ValueError(f"Could not find {self.base_class.__name__} with name {name}. "
                         f"Possible values: {[sub.get_name() for sub in self.get_all_subclasses()]}")


class EvaluatorSubclasses(BaseSubclasses):
    base_class = Evaluator

    @classmethod
    def get_all_subclasses(cls):
        # noinspection PyUnresolvedReferences
        import processtransformer.xai.event_eval
        to_check = cls.base_class.__subclasses__()

        return BaseSubclasses.get_subclasses_from_list(to_check)


class ExplainerSubclasses(BaseSubclasses):
    base_class = Explainer

    @classmethod
    def get_all_subclasses(cls):
        # Do NOT remove import! Required to read all subclasses.
        # Only subclasses from imports are considered.
        # If you get across this: Make sure to import your subclass in the __init__.py
        # noinspection PyUnresolvedReferences
        import processtransformer.xai
        to_check = cls.base_class.__subclasses__()

        return BaseSubclasses.get_subclasses_from_list(to_check)
