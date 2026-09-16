"""Shared test helpers for materialising data to temporary files.

These scoped context managers write a frame/array to a temp file and yield its
path, cleaning the file up on exit. They live at the ``tests`` level so
every test package (optimizer.logical, physical, application, ...) draws the
same helper instead of re-deriving the ``tempfile`` boilerplate.
"""
import os
import tempfile
from contextlib import contextmanager

import numpy as np


@contextmanager
def csv_file(df, **to_csv_kwargs):
    """Write `df` to a temp .csv file and yield its path; cleaned up on exit."""
    # newline="" prevents the text-mode handle from adding an extra "\r" to the
    # "\r\n" that pandas' csv writer already emits (which would otherwise leave a
    # stray "\r" on the last column of every line on Windows).
    tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w", newline="")
    df.to_csv(tmp, index=False, **to_csv_kwargs)
    tmp.close()
    try:
        yield tmp.name
    finally:
        os.remove(tmp.name)


@contextmanager
def npy_file(arr):
    """Write `arr` to a temp .npy file and yield its path; cleaned up on exit."""
    tmp = tempfile.NamedTemporaryFile(suffix=".npy", delete=False, mode="wb")
    np.save(tmp, arr)
    tmp.close()
    try:
        yield tmp.name
    finally:
        os.remove(tmp.name)


@contextmanager
def parquet_file(df):
    """Write `df` to a temp .parquet file and yield its path; cleaned up on exit."""
    tmp = tempfile.NamedTemporaryFile(suffix=".parquet", delete=False, mode="wb")
    df.to_parquet(tmp.name)
    tmp.close()
    try:
        yield tmp.name
    finally:
        os.remove(tmp.name)
