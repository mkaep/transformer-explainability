

import pickle
import lzma

file_ending = "xz"


def _append_file_ending(filename):
    return ("%s." + file_ending) % filename


def compress(obj, filename: str):
    if not filename.endswith('.xz'):
        filename = _append_file_ending(filename)

    with lzma.open(filename, "wb") as file:
        pickle.dump(obj, file)


def decompress(filename: str):
    if not filename.endswith('.xz'):
        filename = _append_file_ending(filename)

    with lzma.open(filename, "rb") as file:
        return pickle.load(file)
