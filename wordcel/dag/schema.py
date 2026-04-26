from typing import List, Optional, Union, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class BackendConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: str
    cache_dir: Optional[str] = None


class ExecutorConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: str = "parallel"
    max_workers: int = 4


class DAGConfigSection(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str
    backend: Optional[BackendConfig] = None
    executor: Optional[ExecutorConfig] = Field(default_factory=ExecutorConfig)
    # Legacy fields
    max_workers: Optional[int] = None
    enable_parallel: Optional[bool] = None


class NodeConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str
    type: str
    input: Optional[Union[str, List[str]]] = None


class WordcelConfig(BaseModel):
    dag: DAGConfigSection
    nodes: List[Dict[str, Any]]
