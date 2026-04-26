"""Functions to help rapidly featurize text."""

import concurrent.futures
import logging
import os
from typing import Any, Callable, Optional, Union

import pandas as pd
from .dag.backends import Backend, LocalBackend


def apply_io_bound_function(
    df: pd.DataFrame,
    user_function: Callable[[str], Any],
    text_column: str,
    num_threads: int = 4,
    cache_folder: Optional[str] = None,
    backend: Optional[Backend] = None,
) -> pd.Series:
    """
    Apply an I/O bound user-provided function to a specific column in a Pandas DataFrame with threading and caching.

    Parameters:
        df (pd.DataFrame): Pandas DataFrame containing the data.
        user_function (function): User-provided function that takes text as input and returns a result.
        text_column (str): Name of the column containing the text to process.
        num_threads (int): Number of threads for concurrent processing.
        cache_folder (str): Folder to store the cached outputs. Legacy parameter, uses LocalBackend internally.
        backend (Backend): A Wordcel Backend instance to handle caching.

    Returns:
        pd.Series: A series of the results, index-aligned with the original df.
    """
    if backend is None and cache_folder:
        backend = LocalBackend({"cache_dir": cache_folder})

    user_fn_name = user_function.__name__

    def process_text_with_caching(text: str, identifier: Any) -> Any:
        cache_node_id = f"featurize_{user_fn_name}_{identifier}"
        
        if backend and backend.exists(cache_node_id, text):
            return backend.load(cache_node_id, text)

        result = user_function(text)

        if backend:
            backend.save(cache_node_id, text, result)

        return result

    def process_row(args):
        index, text = args
        # We use (text, index) as identifier for backward compatibility if needed, 
        # but Backend handles hashing of 'text' as input_data.
        return process_text_with_caching(text, index)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Use enumerate to keep track of rows for the identifier
        results = list(executor.map(process_row, enumerate(df[text_column])))

    return pd.Series(results, index=df.index)
