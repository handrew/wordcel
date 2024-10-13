"""Default helper functions for the DAG / Nodes."""
import pandas as pd
from sqlalchemy import create_engine
from ..featurize import apply_io_bound_function
from ..llms import llm_call


def read_sql(query: str, connection_string: str) -> pd.DataFrame:
    """Helper function to execute a read-only SQL query."""
    engine = create_engine(connection_string)
    results = pd.read_sql(query, connection_string)
    engine.dispose()
    return results


def llm_filter(
    df: pd.DataFrame, column: str, prompt: str, model="gpt-4o-mini", num_threads: int = 1
) -> pd.DataFrame:
    """Helper function to filter a DataFrame using an LLM yes/no question."""
    if num_threads == 1:
        results = df[column].apply(
            lambda value: llm_call(
                prompt + "\n\n----\n\n" + value, model=model
            )
        )
    else:
        results = apply_io_bound_function(
            df,
            lambda value: llm_call(
                prompt + "\n\n----\n\n" + value, model=model
            ),
            text_column=column,
            num_threads=num_threads,
        )

    results = results.str.lower().str.strip()
    return df[results.str.lower().str.startswith("yes")]