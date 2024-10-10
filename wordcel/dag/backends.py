"""Backends to store data."""
import os
import hashlib
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
    def save(self, node_id: str, input_data: Any, data: Any) -> None:
        """
        Save data to the backend.
        """
        pass

    @abstractmethod
    def load(self, node_id: str, input_data: Any) -> Any:
        """
        Load data from the backend.
        """
        pass

    @abstractmethod
    def exists(self, node_id: str, input_data: Any) -> bool:
        """
        Check if data exists in the backend.
        """
        pass


class LocalBackend(Backend):
    """Local backend to store data in a cache directory."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the local backend."""
        # Check for the cache directory.
        cache_dir = os.path.expanduser(config.get("cache_dir"))
        if not cache_dir:
            raise ValueError("Local backend requires a `cache_dir`.")
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_path(self, key: str) -> str:
        return os.path.join(self.cache_dir, f"{key}.json")
    
    def _generate_input_hash(self, input_data: Any) -> str:
        """Generate a hash for the input data."""
        if input_data is None:
            return "none"
        if isinstance(input_data, pd.DataFrame):
            data_str = input_data.to_json(orient="records")
        elif isinstance(input_data, (list, dict)):
            data_str = json.dumps(input_data, sort_keys=True)
        else:
            data_str = str(input_data)
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def generate_cache_key(self, node_id: str, input_data: Any) -> str:
        """Generate a cache key for the node and input data."""
        input_hash = self._generate_input_hash(input_data)
        return f"{node_id}_{input_hash}"

    def save(self, node_id: str, input_data: Any, data: Any) -> None:
        """Save data or DataFrame to a JSON file."""
        if isinstance(data, pd.DataFrame):
            data = data.to_json(orient="records")
            # Add a flag to indicate that the data is a DataFrame.
            data = {"__dataframe__": True, "data": data}

        cache_key = self.generate_cache_key(node_id, input_data)
        with open(self._get_path(cache_key), "w") as f:
            json.dump(data, f, indent=2)

    def load(self, node_id: str, input_data: Any) -> Any:
        """Load data or DataFrame from a JSON file."""
        cache_key = self.generate_cache_key(node_id, input_data)
        with open(self._get_path(cache_key), "r") as f:
            data = json.load(f)
            # Check for the flag indicating a DataFrame.
            if data is not None and "__dataframe__" in data:
                data = pd.read_json(StringIO(data["data"]))
            return data

    def exists(self, node_id: str, input_data: Any) -> bool:
        cache_key = self.generate_cache_key(node_id, input_data)
        return os.path.exists(self._get_path(cache_key))
    

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
