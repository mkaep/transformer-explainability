

import typing
import typing as tp

import numpy as np

from processtransformer.models.helper import Predictor
from processtransformer.models.transformer import Transformer
from processtransformer.util.softmax_funcs import filter_softmax_vector, softmax_vector_to_predictions
from processtransformer.util.types import RelationsDict
from processtransformer.util.types import XWordDict, YWordDict, Trace, XEvent, YEvent

InternalRelationsDict = tp.Dict[tp.Tuple[XEvent], tp.Tuple[tp.Tuple[YEvent, float]]]


class TransformerBaseCheck:

    def __init__(self,
                 model: Transformer,
                 x_word_dict: XWordDict,
                 y_word_dict: YWordDict,
                 prediction_threshold: float = 0.1
                 ):
        super().__init__()
        self.predictor = Predictor(model, x_word_dict, y_word_dict)
        self.prediction_threshold = prediction_threshold

    def compare_with_other(self, trace: Trace, other_dict: RelationsDict) -> float:
        own_dict = self.get_relations_for_trace(trace)

        num_rows = len(own_dict.items())
        num_cols = len(other_dict.items())
        if num_rows == 0 or num_cols == 0:
            return 0.0

        # own, i.e. "ground truth"(GT)-relationships as rows, XAI-relationships as columns
        mat = np.ndarray(shape=(num_rows, num_cols))

        for i, (own_x, own_y) in enumerate(own_dict.items()):
            own_x = set(own_x)
            own_y = set(own_y)
            for j, (other_x, other_y) in enumerate(other_dict.items()):
                other_x = set(other_x)
                other_y = set(other_y)

                match_score = self._compute_match(own_x, own_y, other_x, other_y)
                mat[i, j] = match_score

        # Check if every row (GT) has been chosen at least once
        row_occupancy = np.zeros(shape=num_rows)
        for col in mat.T:
            row_occupancy[np.argmax(col)] = 1.0
        row_denominator = np.sum(row_occupancy) / num_rows

        col_denominator = 1.0 / mat.shape[1]
        return np.sum(np.max(mat, axis=0)) * col_denominator * row_denominator

    @classmethod
    def _compute_match(cls, own_x, own_y, other_x, other_y) -> float:
        return (cls._eval_intersection(own_x, other_x) + cls._eval_intersection(own_y, other_y)) / 2.0

    @classmethod
    def _eval_intersection(cls, other, own) -> float:
        # E.g. {A, B, C}, {B, C, D} --> {B, C}, makes 2/3
        # Penalize, if either of those has way more events
        len_own = len(own)
        len_other = len(other)
        if len_own == 0 or len_other == 0:
            return 0.0
        num_intersection = len(own.intersection(other))

        return min(num_intersection / len_own, num_intersection / len_other)

    def get_relations_for_trace(self, trace: Trace) -> RelationsDict:
        _, _, _, _, softmax, _ = self.predictor.make_prediction(trace, return_softmax=True)
        base_softmax = filter_softmax_vector(softmax, self.prediction_threshold)

        relations_dict: InternalRelationsDict = dict()
        relations_dict[tuple(trace)] = tuple(base_softmax)

        while True:
            new_relations_dict: InternalRelationsDict = dict()

            for x_events, y_events in relations_dict.items():
                if len(x_events) == 1:
                    # Cannot split this trace anymore
                    new_relations_dict[x_events] = y_events
                    continue

                dct = self._mask_single_events(x_events, y_events)
                new_relations_dict.update(dct)

            if set(relations_dict.keys()) == set(new_relations_dict.keys()):
                break
            relations_dict = new_relations_dict

        # Remove masked events, e.g. (A, B, M-C, D, M-E) becomes (A, B, D)
        relations_dict = {tuple([ev for ev in key if not ev.startswith('M-')]): value
                          for key, value in relations_dict.items()}
        final_dict: RelationsDict = dict()

        for key, value in relations_dict.items():
            tpl = tuple(softmax_vector_to_predictions(list(value)))
            # Remove empty predictions
            if len(tpl) == 0:
                continue
            final_dict[key] = tpl

        # Remove subsets of the left hand side.
        # E.g. E -> R, S and E, J -> R, S will remove the second rule and keep just the first one
        reverse_keys = set(final_dict.values())
        # Dict of {(str, ): [(str, )]} (key: tuple of str, value: list of tuples of str)
        reverse_dict: typing.Dict[typing.Tuple[str], typing.List[typing.Tuple[str]]] = {key: [] for key in reverse_keys}
        for key, value in final_dict.items():
            reverse_dict[value].append(key)
        for key, value in reverse_dict.items():
            lst = list(set(value))
            lst.sort(key=lambda x: len(x))
            to_remove = set()
            for i in range(0, len(lst)):
                i_set = set(lst[i])
                for j in range(i + 1, len(lst)):
                    j_set = set(lst[j])
                    if i_set.issubset(j_set):
                        to_remove.add(j)
            lst = [item for index, item in enumerate(lst) if index not in to_remove]
            reverse_dict[key] = lst

        # Invert dict and merge left hand sides with the same right hand side
        # E.g. A, B -> X, Y and B, C -> X, Y become A, B, C -> X, Y
        final_dict = dict()
        for key, value_list in reverse_dict.items():
            new_key = tuple(set(event for sublist in value_list for event in sublist))
            new_value = tuple(set(key))
            final_dict[new_key] = new_value

        return final_dict

    def _mask_single_events(self,
                            base_trace: tp.Tuple[XEvent],
                            base_pred: tp.Tuple[tp.Tuple[YEvent, float]],
                            ) -> InternalRelationsDict:
        base_set = set(softmax_vector_to_predictions(list(base_pred)))

        internal_dict: InternalRelationsDict = dict()
        reduced = False

        for i in range(len(base_trace)):
            if base_trace[i].startswith('M-'):
                # Ignore if already masked
                continue

            # Mask out one event after the other. I.e. A, B, C -> M-A, B, C; A, M-B, C; A, B, M-C
            trace = [event if i != j else f'M-{event}' for j, event in enumerate(base_trace)]
            _, _, _, _, softmax, _ = self.predictor.make_prediction(trace, return_softmax=True)
            softmax = filter_softmax_vector(softmax, self.prediction_threshold)
            softmax_set = set(softmax_vector_to_predictions(softmax))

            # Remove masked event
            shortened_trace = tuple(trace)
            if softmax_set == base_set:
                internal_dict[shortened_trace] = tuple(softmax)
                reduced = True

        if not reduced:
            internal_dict[base_trace] = base_pred

        return internal_dict
