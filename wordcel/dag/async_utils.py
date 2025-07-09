"""Utilities for running async code within the synchronous DAG executor."""

import asyncio
import threading
from typing import Coroutine, Any

# Thread-local storage to hold a unique event loop for each thread
_thread_local = threading.local()


def run_async_in_thread(coro: Coroutine) -> Any:
    """
    Runs a coroutine in a thread-safe manner by managing a persistent
    event loop for each thread.

    This should be used as a replacement for asyncio.run() when calling
    async code from a synchronous function that may be executed in
    different threads by a thread pool (e.g., in a PythonFunctionNode
    run by the ParallelDAGExecutor).

    Example:
        # your_async_functions.py
        async def my_async_task():
            # ... async logic ...
            return "done"

        # your_sync_wrapper.py
        from wordcel.dag.async_utils import run_async_in_thread
        from .your_async_functions import my_async_task

        def sync_task_for_dag():
            return run_async_in_thread(my_async_task())

    Your DAG's PythonFunctionNode would then point to `your_sync_wrapper.sync_task_for_dag`.
    """
    # Check if this thread already has an event loop
    try:
        loop = _thread_local.loop
    except AttributeError:
        # If not, create a new one and store it in thread-local storage
        loop = asyncio.new_event_loop()
        _thread_local.loop = loop
        asyncio.set_event_loop(loop)

    # Run the coroutine and return the result.
    # This will use the thread-specific event loop.
    return loop.run_until_complete(coro)
