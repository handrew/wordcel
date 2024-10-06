"""Backends to store data."""
import os
import json
import pandas as pd
from io import StringIO
from abc import ABC, abstractmethod
from typing import Any, Dict


class Backend(ABC):
    """Abstract class for a backend to store data."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the backend."""
        self.config = config or {}

    @abstractmethod
    def save(self, key: str, data: Any) -> None:
        """
        Save data to the backend.
        """
        pass

    @abstractmethod
    def load(self, key: str) -> Any:
        """
        Load data from the backend.
        """
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """
        Check if data exists in the backend.
        """
        pass


class LocalBackend(Backend):
    """Local backend to store data in a cache directory."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the local backend."""
        # Check for the cache directory.
        cache_dir = config.get("cache_dir")
        if not cache_dir:
            raise ValueError("Local backend requires a `cache_dir`.")
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_path(self, key: str) -> str:
        return os.path.join(self.cache_dir, f"{key}.json")

    def save(self, key: str, data: Any) -> None:
        """Save data or DataFrame to a JSON file."""
        if isinstance(data, pd.DataFrame):
            data = data.to_json()
            # Add a flag to indicate that the data is a DataFrame.
            data = {"__dataframe__": True, "data": data}

        with open(self._get_path(key), "w") as f:
            json.dump(data, f, indent=4)

    def load(self, key: str) -> Any:
        """Load data or DataFrame from a JSON file."""
        with open(self._get_path(key), "r") as f:
            data = json.load(f)
            # Check for the flag indicating a DataFrame.
            if data is not None and "__dataframe__" in data:
                data = pd.read_json(StringIO(data["data"]))
            return data

    def exists(self, key: str) -> bool:
        return os.path.exists(self._get_path(key))
    

BACKEND_TYPES: Dict[str, Backend] = {
    "local": LocalBackend,
}


class BackendRegistry:
    """Registry for backends."""

    _registry: Dict[str, Backend] = {}

    @classmethod
    def register(cls, name: str, backend: Backend) -> None:
        cls._registry[name] = backend

    @classmethod
    def get(cls, name: str) -> Backend:
        return cls._registry.get(name)

    @classmethod
    def register_default_backends(cls) -> None:
        for name, backend in BACKEND_TYPES.items():
            cls.register(name, backend)
