"""Functions to help rapidly featurize text."""

import concurrent.futures
import json
import logging
import os
from typing import Any, Callable, Optional

import pandas as pd


def apply_io_bound_function(
    df: pd.DataFrame,
    user_function: Callable[[str], Any],
    text_column: str,
    num_threads: int = 4,
    cache_folder: Optional[str] = None,
) -> pd.Series:
    """
    Apply an I/O bound user-provided function to a specific column in a Pandas DataFrame with threading and caching.

    Parameters:
        df (pd.DataFrame): Pandas DataFrame containing the data.
        user_function (function): User-provided function that takes text as input and returns a JSON.
        text_column (str): Name of the column containing the text to process.
        num_threads (int): Number of threads for concurrent processing.
        cache_folder (str): Folder to store the cached JSON outputs.

    Returns:
        pd.Series: A series of the results, index-aligned with the original df.
    """
    if cache_folder and not os.path.exists(cache_folder):
        os.makedirs(cache_folder)

    user_fn_name = user_function.__name__

    def process_text_with_caching(text: str, identifier: Any) -> Any:
        if cache_folder:
            # Check if result is already cached
            identifier = str(identifier).replace("/", "_")
            cache_file = os.path.join(
                cache_folder,
                f"{identifier}_{user_fn_name}.json",
            )
            if os.path.exists(cache_file):
                with open(cache_file, "r") as f:
                    logging.info(f"Found cached result: {cache_file}")
                    return json.load(f)

        result = user_function(text)

        if cache_folder:
            # Cache the result
            with open(cache_file, "w") as f:
                json.dump(result, f)

        return result

    def process_row(args):
        index, text = args
        return process_text_with_caching(text, index)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(executor.map(process_row, enumerate(df[text_column])))

    return pd.Series(results, index=df.index)
