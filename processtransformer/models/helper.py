

import threading
import typing

import numpy as np
import scipy.special
import tensorflow as tf

from processtransformer.models.transformer import Transformer
from processtransformer.util.types import Trace


class Predictor:
    def __init__(self, model: Transformer,
                 x_word_dict: typing.Dict[str, int],
                 y_word_dict: typing.Dict[str, int],
                 show_pad: bool = False):
        self.model = model
        self.x_word_dict = x_word_dict
        self.y_word_dict = y_word_dict
        self.inverse_x_word_dict: typing.Dict[int, str] = {value: key for key, value in x_word_dict.items()}
        self.inverse_y_word_dict: typing.Dict[int, str] = {value: key for key, value in y_word_dict.items()}
        self.show_pad = show_pad

        self.lock = threading.Lock()
        self.cache = dict()

    def make_multi_predictions(self,
                               event_traces: typing.List[Trace],
                               return_softmax: bool = False,
                               attn_indices_to_mask_list: typing.List = None,
                               ):
        if len(event_traces) == 0:
            return []

        if attn_indices_to_mask_list is None:
            attn_indices_to_mask_list = [[]] * len(event_traces)

        assert len(event_traces) == len(attn_indices_to_mask_list)

        info = []
        for event_trace, attn_indices_to_mask in zip(event_traces, attn_indices_to_mask_list):
            mask, pad_count, padded_trace, token_trace = self._prepare_trace_and_mask(event_trace, attn_indices_to_mask)
            info.append((mask, pad_count, padded_trace, token_trace, event_trace, attn_indices_to_mask))

        token_traces = list(list(zip(*info))[3])
        token_traces = tf.concat(token_traces, axis=0)
        masks = list(list(zip(*info))[0])
        if len(masks) > 0 and masks[0] is not None:
            masks = tf.concat(masks, axis=0)
        else:
            masks = None

        y_pred_softmaxes = []
        attn_scores = []
        steps = 100
        # Split up so we do not require more GPU memory than available (hopefully)
        for i in range(0, len(token_traces), steps):
            sub_traces = token_traces[i:i+steps]
            sub_masks = masks[i:i+steps]
            sms, attns = self.model(sub_traces, mask=sub_masks, return_attention_scores=True)
            y_pred_softmaxes.append(sms.numpy())
            attn_scores.append(attns.numpy())

        y_pred_softmaxes = np.concatenate(y_pred_softmaxes, axis=0)
        attn_scores = np.concatenate(attn_scores, axis=0)

        batch_size = attn_scores.shape[0]  # number of traces
        y_pred_softmaxes = np.split(y_pred_softmaxes, batch_size, axis=0)
        attn_scores = np.split(attn_scores, batch_size, axis=0)

        output_list = []
        for (mask, pad_count, padded_trace, token_trace, event_trace, attn_indices_to_mask), \
                y_pred_softmax, attn in zip(info, y_pred_softmaxes, attn_scores):

            key = self._get_input_as_key(event_trace, return_softmax, attn_indices_to_mask)
            output = self._transform_into_output(event_trace, key, pad_count, padded_trace,
                                                 return_softmax, y_pred_softmax, attn)
            output_list.append(output)

        return output_list

    def make_prediction(self, event_trace: Trace,
                        return_softmax: bool = False,
                        attn_indices_to_mask=None,
                        ):
        """
        Returns: predicted-event, predicted-token, attention-scores, result-trace
        Or (if return_softmax=True): (*result, output-softmax as tuples of <event-name, value>, output-softmax)
        """
        # String to int
        if attn_indices_to_mask is None:
            attn_indices_to_mask = []

        key = self._get_input_as_key(event_trace, return_softmax, attn_indices_to_mask)
        if key in self.cache.keys():
            return self.cache[key]

        mask, pad_count, padded_trace, token_trace = self._prepare_trace_and_mask(event_trace, attn_indices_to_mask)

        y_pred_softmax, attn_scores = self.model(token_trace, mask=mask, return_attention_scores=True)

        return self._transform_into_output(event_trace, key, pad_count, padded_trace, return_softmax,
                                           y_pred_softmax, attn_scores)

    @staticmethod
    def _get_input_as_key(event_trace: Trace, return_softmax: bool, attn_indices_to_mask):
        return tuple(event_trace), return_softmax, tuple(attn_indices_to_mask)

    def _transform_into_output(self, event_trace, key, pad_count, padded_trace, return_softmax,
                               y_pred_softmax, attn_scores):
        # Apply 'real' softmax function and create tuples<Name, Float-Value>
        last_softmax: np.ndarray = scipy.special.softmax(y_pred_softmax).reshape(-1)
        last_softmax_tuples = [(self.inverse_y_word_dict[index], y_value) for index, y_value in enumerate(last_softmax)]
        y_pred_token: int = np.argmax(y_pred_softmax, axis=1).item(0)

        # Int to string
        y_pred_event = self.inverse_y_word_dict[y_pred_token]
        result_trace = padded_trace

        if not self.show_pad:
            # Delete padding from attention score matrix (last two axes)
            # First two axes (layer and heads respectively) stay untouched
            attn_scores = np.delete(attn_scores, np.arange(pad_count), 2)
            attn_scores = np.delete(attn_scores, np.arange(pad_count), 3)

            # Also, return original trace and not the padded trace
            result_trace = event_trace

        result = y_pred_event, y_pred_token, attn_scores, result_trace
        if return_softmax:
            result = (*result, last_softmax_tuples, last_softmax)

        if key is not None:
            with self.lock:
                self.cache[key] = result
        return result

    def _prepare_trace_and_mask(self, event_trace, attn_indices_to_mask, truncating='pre'):
        max_case_len = self.model.max_case_length.numpy()
        if len(event_trace) > max_case_len:
            if truncating == 'pre':
                event_trace = event_trace[-max_case_len:]
            else:
                event_trace = event_trace[:max_case_len]

        default_val = self.x_word_dict['[PAD]']
        token_trace = [self.x_word_dict.get(event, default_val) for event in event_trace]
        unknown_events_indices = set(i for i, event in enumerate(event_trace) if event not in self.x_word_dict.keys())
        attn_indices_to_mask = unknown_events_indices.union(set(attn_indices_to_mask))

        # Pad
        pad_count = len(token_trace)
        token_trace = tf.keras.preprocessing.sequence.pad_sequences(
            [token_trace], maxlen=max_case_len, truncating=truncating)
        pad_count = len(token_trace.reshape(-1)) - pad_count
        # To nd-array
        token_trace = np.array(token_trace, dtype=np.float32)
        padded_trace = [self.inverse_x_word_dict[token] for token in token_trace.reshape(-1)]

        if attn_indices_to_mask is None:
            attn_indices_to_mask = set()
        else:
            attn_indices_to_mask = set(attn_indices_to_mask)

        # 0 if masked, 1 if not
        nd = np.asarray([1 if i not in attn_indices_to_mask else 0
                         for i in range(len(event_trace))])

        if pad_count > 0:
            # Pad with zeros at start
            nd = np.concatenate([np.zeros(shape=pad_count), nd])

        # Transform to right dimension for Transformer
        nd = nd[np.newaxis, np.newaxis, np.newaxis, :]
        mask = tf.convert_to_tensor(nd, dtype=tf.int32)
        mask = tf.einsum('abcd,abce->abed', mask, mask)

        return mask, pad_count, padded_trace, token_trace
