

import random
import typing

import numpy as np


def trace_masking_modification(trace: typing.List[str], masking_positions=None):
    """
    Note: masking-positions is not a list of indices to mask.
    Rather, each int in the list represents the combination of masked and non-masked events.
    The number 6, 110 in binary, keeps the first two events and masks the last one.
    The number 10, 1010 in binary, keeps the first and third event and masks the second and fourth one.
    """
    num_masking_positions = len(trace)

    if masking_positions is None:
        # start with 1 -> leave out fully-masked trace
        masking_positions = range(1, 2 ** num_masking_positions)

    for i in masking_positions:
        # cut of '0b' at the start
        binary = bin(i)[2:]
        # make binary string long enough for indexing
        binary = '0' * (num_masking_positions - len(binary)) + binary
        # 1 == use original event, 0 == mask/pad. E.g. 00001 masks all but the last event.
        masked_trace = [event if binary[index] == '1' else '[PAD]' for index, event in enumerate(trace)]
        yield masked_trace


def trace_partial_masking_modification(trace: typing.List[str], count: int):
    if len(trace) == 0:
        return []

    # Can go over all possible combinations
    possible_combinations = 2 ** len(trace)
    if count > possible_combinations:
        return trace_masking_modification(trace)

    # Choose some combinations only
    dt = np.dtype(np.uint64)
    max_value_dt = 2 ** (dt.itemsize * 8)

    high = min(possible_combinations, max_value_dt)
    rng = np.random.Generator(np.random.PCG64(1776))
    indices = rng.integers(low=0, high=high, size=count, dtype=dt)

    return trace_masking_modification(trace, indices)


def random_trace_sampling(times: int, trace: typing.List[str]):
    rng = np.random.Generator(np.random.PCG64(4711))
    for i in range(times):
        # Randomly change order of events in trace
        yield rng.permutation(trace)
