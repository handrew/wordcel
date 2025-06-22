"""DAG execution strategies."""
import time
import threading
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict, deque
from typing import Dict, Any, TYPE_CHECKING
from rich.console import Console

if TYPE_CHECKING:
    from .dag import WordcelDAG
    from .nodes import Node

# Default console instance
_default_console = Console()


class DAGExecutor(ABC):
    """Abstract base class for DAG execution strategies."""
    
    def __init__(self, verbose: bool = False, console: Console = None):
        self.verbose = verbose
        self.console = console or _default_console
    
    @abstractmethod
    def execute(self, dag: 'WordcelDAG', input_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute the DAG and return results."""
        pass
    
    def _prepare_incoming_input(self, dag: 'WordcelDAG', input_data: Dict[str, Any], 
                               results: Dict[str, Any], node_id: str) -> Any:
        """Prepare the incoming input for a node."""
        incoming_edges = dag.graph.nodes[node_id].get("input")
        incoming_input = None
        if input_data is not None and isinstance(input_data, dict) and node_id in input_data:
            # First check if the input data is given at runtime.
            # If so, we don't need to look at the incoming edges.
            assert (
                incoming_edges is None
            ), "Node cannot have both `input` and input data given at runtime."
            incoming_input = input_data[node_id]
        elif isinstance(incoming_edges, list):
            incoming_input = [results[input_id] for input_id in incoming_edges]
        else:
            incoming_input = results.get(incoming_edges)

        return incoming_input

    def _check_result_is_json_serializable(self, dag: 'WordcelDAG', results: Dict[str, Any], node_id: str):
        """Check if result is JSON serializable."""
        from .dag import _is_json_serializable
        if not _is_json_serializable(results[node_id]):
            result_type = type(results[node_id]).__name__
            raise ValueError(
                f"Node `{node_id}` returned type `{result_type}`, which either is not serializable or contains something, like a DataFrame, that is not serializable."
            )


class SequentialDAGExecutor(DAGExecutor):
    """Sequential execution strategy - executes nodes one by one in topological order."""
    
    def execute(self, dag: 'WordcelDAG', input_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute the DAG sequentially."""
        import networkx as nx
        from .nodes import NodeRegistry
        
        results = {}
        nodes_list = list(nx.topological_sort(dag.graph))
        total_nodes = len(nodes_list)

        self.console.print(f"\n🚀 [bold blue]Executing DAG (Sequential):[/bold blue] [bold]{dag.name}[/bold]")
        self.console.print(f"📊 [dim]Total nodes: {total_nodes}[/dim]\n")

        for i, node_id in enumerate(nodes_list, 1):
            node = dag.nodes[node_id]
            node_type = node.__class__.__name__
            start_time = time.time()
            
            # Rich formatted progress
            self.console.print(f"[bold cyan]\\[{i}/{total_nodes}][/bold cyan] [bold]{node_id}[/bold] [dim]({node_type})[/dim]", end=" ")

            try:
                # Get the incoming edges and their inputs.
                incoming_input = self._prepare_incoming_input(dag, input_data, results, node_id)
                
                # Check the cache, if we have a backend.
                if dag.backend and dag.backend.exists(node_id, incoming_input):
                    self.console.print("[yellow]📦 cache[/yellow]", end=" ")
                    results[node_id] = dag.backend.load(node_id, incoming_input)
                else:
                    self.console.print("[blue]⚡ exec[/blue]", end=" ")
                    results[node_id] = node.execute(incoming_input)

                    # If the node is not a DAG node, check if the result is JSON serializable.
                    is_dag_node = isinstance(node, NodeRegistry.get("dag"))
                    if not is_dag_node:
                        self._check_result_is_json_serializable(dag, results, node_id)

                    # Don't save DAG nodes to cache, since they may have their own cache.
                    if dag.backend and not is_dag_node:
                        dag.backend.save(node_id, incoming_input, results[node_id])

                elapsed = time.time() - start_time
                self.console.print(f"[bold green]✓[/bold green] [green]{elapsed:.2f}s[/green]")

                if self.verbose:
                    self.console.print(f"[dim]Result for {node_id}:[/dim]")
                    print(results[node_id])
                    self.console.print()

            except Exception as e:
                elapsed = time.time() - start_time
                self.console.print(f"[bold red]✗[/bold red] [red]{elapsed:.2f}s[/red]")
                
                error_context = {
                    'node_id': node_id,
                    'node_type': node_type,
                    'input_type': type(incoming_input).__name__ if incoming_input is not None else 'None',
                    'config_keys': list(node.config.keys())
                }
                self.console.print(f"[red]Error:[/red] {e}")
                self.console.print(f"[dim]Context: {error_context}[/dim]")
                raise RuntimeError(f"Node {node_id} ({node_type}) failed: {e}") from e

        self.console.print(f"\n[bold green]🎉 DAG completed successfully![/bold green] [dim]({total_nodes} nodes)[/dim]")
        return results


class ParallelDAGExecutor(DAGExecutor):
    """Parallel execution strategy - executes independent nodes concurrently."""
    
    def __init__(self, max_workers: int = 4, verbose: bool = False, console: Console = None):
        super().__init__(verbose, console)
        self.max_workers = max_workers
    
    def _execute_node(self, dag: 'WordcelDAG', node_id: str, node: 'Node', 
                     input_data: Dict[str, Any], results: Dict[str, Any], 
                     results_lock: threading.Lock) -> Dict[str, Any]:
        """Execute a single node and return its result."""
        from .nodes import NodeRegistry
        
        node_type = node.__class__.__name__
        start_time = time.time()
        
        try:
            # Get the incoming input (thread-safe read from results)
            with results_lock:
                incoming_input = self._prepare_incoming_input(dag, input_data, results, node_id)
            
            # Check cache if we have a backend
            result = None
            cache_hit = False
            if dag.backend and dag.backend.exists(node_id, incoming_input):
                result = dag.backend.load(node_id, incoming_input)
                cache_hit = True
            else:
                result = node.execute(incoming_input)
                
                # Validate result is JSON serializable
                is_dag_node = isinstance(node, NodeRegistry.get("dag"))
                if not is_dag_node:
                    temp_results = {node_id: result}
                    self._check_result_is_json_serializable(dag, temp_results, node_id)
                
                # Save to cache if available
                if dag.backend and not is_dag_node:
                    dag.backend.save(node_id, incoming_input, result)
            
            elapsed = time.time() - start_time
            return {
                'node_id': node_id,
                'result': result,
                'elapsed': elapsed,
                'cache_hit': cache_hit,
                'node_type': node_type,
                'success': True
            }
            
        except Exception as e:
            elapsed = time.time() - start_time
            return {
                'node_id': node_id,
                'result': None,
                'elapsed': elapsed,
                'cache_hit': False,
                'node_type': node_type,
                'success': False,
                'error': e
            }

    def execute(self, dag: 'WordcelDAG', input_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute the DAG with parallel execution of independent nodes."""
        results = {}
        results_lock = threading.Lock()
        
        # Build dependency tracking
        in_degree = {}
        dependencies = defaultdict(set)
        dependents = defaultdict(set)
        
        for node_id in dag.graph.nodes():
            predecessors = list(dag.graph.predecessors(node_id))
            in_degree[node_id] = len(predecessors)
            dependencies[node_id] = set(predecessors)
            for pred in predecessors:
                dependents[pred].add(node_id)
        
        # Find nodes ready to execute (no dependencies)
        ready_queue = deque([node_id for node_id, degree in in_degree.items() if degree == 0])
        executing = set()
        completed = set()
        total_nodes = len(dag.graph.nodes())
        
        self.console.print(f"\n🚀 [bold blue]Executing DAG (Parallel):[/bold blue] [bold]{dag.name}[/bold]")
        self.console.print(f"📊 [dim]Total nodes: {total_nodes}, Max workers: {self.max_workers}[/dim]\n")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_node = {}
            
            while ready_queue or executing:
                # Submit ready nodes for execution
                while ready_queue and len(executing) < self.max_workers:
                    node_id = ready_queue.popleft()
                    if node_id not in executing and node_id not in completed:
                        node = dag.nodes[node_id]
                        executing.add(node_id)
                        
                        future = executor.submit(
                            self._execute_node, 
                            dag, node_id, node, input_data, results, results_lock
                        )
                        future_to_node[future] = node_id
                        
                        self.console.print(f"[bold cyan]🟦[/bold cyan] [bold]{node_id}[/bold] [dim]({node.__class__.__name__})[/dim] started")
                
                # Wait for at least one node to complete
                if future_to_node:
                    # Wait for the first future to complete
                    done_futures = as_completed(future_to_node.keys())
                    
                    # Process the first completed future
                    future = next(done_futures)
                    node_id = future_to_node[future]
                    executing.remove(node_id)
                    completed.add(node_id)
                    del future_to_node[future]
                    
                    # Get the result
                    execution_result = future.result()
                    
                    if execution_result['success']:
                        # Store result thread-safely
                        with results_lock:
                            results[node_id] = execution_result['result']
                        
                        # Print success
                        cache_indicator = "[yellow]📦 cache[/yellow]" if execution_result['cache_hit'] else "[blue]⚡ exec[/blue]"
                        self.console.print(f"[bold green]✓[/bold green] [bold]{node_id}[/bold] {cache_indicator} [green]{execution_result['elapsed']:.2f}s[/green]")
                        
                        if self.verbose:
                            self.console.print(f"[dim]Result for {node_id}:[/dim]")
                            print(execution_result['result'])
                            self.console.print()
                        
                        # Update dependencies and add newly ready nodes
                        for dependent in dependents[node_id]:
                            dependencies[dependent].discard(node_id)
                            if not dependencies[dependent] and dependent not in completed and dependent not in executing:
                                ready_queue.append(dependent)
                    
                    else:
                        # Handle error
                        error = execution_result['error']
                        self.console.print(f"[bold red]✗[/bold red] [bold]{node_id}[/bold] [red]{execution_result['elapsed']:.2f}s[/red]")
                        
                        error_context = {
                            'node_id': node_id,
                            'node_type': execution_result['node_type'],
                            'config_keys': list(dag.nodes[node_id].config.keys())
                        }
                        self.console.print(f"[red]Error:[/red] {error}")
                        self.console.print(f"[dim]Context: {error_context}[/dim]")
                        raise RuntimeError(f"Node {node_id} ({execution_result['node_type']}) failed: {error}") from error
        
        self.console.print(f"\n[bold green]🎉 DAG completed successfully![/bold green] [dim]({total_nodes} nodes)[/dim]")
        return results


class ExecutorRegistry:
    """Registry for DAG executor types."""
    
    _registry: Dict[str, type] = {}
    
    @classmethod
    def register(cls, name: str, executor_class: type) -> None:
        """Register an executor class."""
        cls._registry[name] = executor_class
    
    @classmethod
    def get(cls, name: str) -> type:
        """Get an executor class by name."""
        return cls._registry.get(name)
    
    @classmethod
    def create(cls, executor_type: str = "parallel", **kwargs) -> DAGExecutor:
        """Create an executor instance."""
        executor_class = cls.get(executor_type)
        if executor_class is None:
            raise ValueError(f"Unknown executor type: {executor_type}")
        return executor_class(**kwargs)
    
    @classmethod
    def register_default_executors(cls) -> None:
        """Register the default executor types."""
        cls.register("sequential", SequentialDAGExecutor)
        cls.register("parallel", ParallelDAGExecutor)


# Register default executors
ExecutorRegistry.register_default_executors()