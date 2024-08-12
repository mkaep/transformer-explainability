from abc import ABC


class JsonSerializable(ABC):
    def to_dict(self):
        """
        Returns object as dictionary.
        """
        raise NotImplementedError()

    @staticmethod
    def from_dict(data):
        """
        Creates object from dictionary.
        """
        raise NotImplementedError()
