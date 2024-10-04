"""Functions to help rapidly featurize text."""

import concurrent.futures
import json
import os
import pandas as pd


def apply_io_bound_function(
    df,
    user_function,
    text_column=None,
    id_column=None,
    result_field="result",
    num_threads=4,
    cache_folder=None,
):
    """
    Apply an I/O bound user-provided function to a specific column in a Pandas DataFrame with threading and caching.

    Parameters:
        df (pd.DataFrame): Pandas DataFrame containing the data.
        user_function (function): User-provided function that takes text as input and returns a JSON.
        text_column (str): Name of the column containing the text to process. If None, the first string column will be used.
        id_column (str): Name of the column to be used as the identifier. If None, the DataFrame index will be used.
        result_field (str): Name of the new colummn to be created. Will also be used in the cache's naming convention.
        num_threads (int): Number of threads for concurrent processing.
        cache_folder (str): Folder to store the cached JSON outputs.

    Returns:
        pd.DataFrame: The original DataFrame with an additional 'result' column containing the JSON outputs.
    """
    assert text_column is not None, "The 'text_column' argument must be specified."

    if cache_folder and not os.path.exists(cache_folder):
        os.makedirs(cache_folder)

    if id_column is None:
        id_column = df.index.name if df.index.name else "index"

    user_fn_name = user_function.__name__

    def process_text_with_caching(text, identifier):
        if cache_folder:
            # Check if result is already cached
            identifier = str(identifier).replace("/", "_")
            result_field_normalized = result_field.replace("/", "_")  # For naming...
            cache_file = os.path.join(
                cache_folder,
                f"{identifier}_{user_fn_name}_{result_field_normalized}.json",
            )
            if os.path.exists(cache_file):
                with open(cache_file, "r") as f:
                    print(f"Found cached result: {cache_file}.")
                    return json.load(f)

        result = user_function(text)

        if cache_folder:
            # Cache the result
            with open(cache_file, "w") as f:
                json.dump(result, f)

        return result

    def process_row(row):
        idx, row = row
        text = row[text_column]
        identifier = row[id_column]
        inference = process_text_with_caching(text, identifier)
        return {id_column: identifier, result_field: inference}

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(executor.map(process_row, df.iterrows()))

    return pd.DataFrame(results)
