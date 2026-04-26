import os
import yaml
import pandas as pd
import pytest
from wordcel.dag import WordcelDAG
from unittest.mock import patch, MagicMock

class TestRefactorBaseline:
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        self.test_csv_path = "baseline_data.csv"
        self.dag_yaml_path = "baseline_dag.yaml"
        self.cache_dir = "baseline_cache"
        self.output_txt = "baseline_output.txt"

        # Create test data
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "text": ["apple", "banana", "cherry"],
            "value": [10, 20, 30]
        })
        df.to_csv(self.test_csv_path, index=False)

        yield

        # Cleanup
        for f in [self.test_csv_path, self.dag_yaml_path, self.output_txt]:
            if os.path.exists(f):
                os.remove(f)
        if os.path.exists(self.cache_dir):
            import shutil
            shutil.rmtree(self.cache_dir)

    def create_dag_config(self, parallel=False, cache=False):
        config = {
            "dag": {
                "name": "baseline_dag",
                "executor_type": "parallel" if parallel else "sequential",
            },
            "nodes": [
                {
                    "id": "load_csv",
                    "type": "csv",
                    "path": self.test_csv_path
                },
                {
                    "id": "filter_data",
                    "type": "dataframe_operation",
                    "input": "load_csv",
                    "operation": "query",
                    "args": ["value > 15"]
                },
                {
                    "id": "format_text",
                    "type": "string_template",
                    "input": "filter_data",
                    "template": "Process this: ${text}",
                    "mode": "multiple"
                },
                {
                    "id": "llm_process",
                    "type": "llm",
                    "input": "format_text",
                    "template": "Translate to French: {input}",
                    "num_threads": 2
                },
                {
                    "id": "save_results",
                    "type": "file_writer",
                    "input": "llm_process",
                    "path": self.output_txt
                }
            ]
        }
        if cache:
            config["dag"]["backend"] = {
                "type": "local",
                "cache_dir": self.cache_dir
            }
        return config

    def test_baseline_execution_sequential(self, mock_llm_call):
        mock_llm_call.side_effect = lambda x, **kwargs: f"French_{x}"
        
        config = self.create_dag_config(parallel=False, cache=False)
        with open(self.dag_yaml_path, "w") as f:
            yaml.dump(config, f)
            
        dag = WordcelDAG(self.dag_yaml_path)
        results = dag.execute()
        
        assert "save_results" in results
        assert os.path.exists(self.output_txt)
        assert len(results["filter_data"]) == 2
        # The mock returns "French_" + the full prompt
        assert results["llm_process"] == [
            "French_Translate to French: Process this: banana", 
            "French_Translate to French: Process this: cherry"
        ]

    def test_baseline_execution_parallel(self, mock_llm_call):
        mock_llm_call.side_effect = lambda x, **kwargs: f"French_{x}"
        
        config = self.create_dag_config(parallel=True, cache=False)
        with open(self.dag_yaml_path, "w") as f:
            yaml.dump(config, f)
            
        dag = WordcelDAG(self.dag_yaml_path)
        results = dag.execute()
        
        assert "save_results" in results
        assert os.path.exists(self.output_txt)
        assert results["llm_process"] == [
            "French_Translate to French: Process this: banana", 
            "French_Translate to French: Process this: cherry"
        ]

    def test_baseline_caching(self, mock_llm_call):
        mock_llm_call.side_effect = lambda x, **kwargs: f"French_{x}"
        
        config = self.create_dag_config(parallel=False, cache=True)
        with open(self.dag_yaml_path, "w") as f:
            yaml.dump(config, f)
            
        # First run
        dag = WordcelDAG(self.dag_yaml_path)
        dag.execute()
        assert mock_llm_call.call_count == 2
        
        # Second run - should use cache
        mock_llm_call.reset_mock()
        dag.execute()
        assert mock_llm_call.call_count == 0

    def test_baseline_input_validation(self):
        # Create a DAG with invalid input types
        config = {
            "dag": {"name": "invalid_input"},
            "nodes": [
                {"id": "node1", "type": "yaml", "path": self.dag_yaml_path},
                {"id": "node2", "type": "llm_filter", "input": "node1", "column": "col", "prompt": "p"}
            ]
        }
        with open(self.dag_yaml_path, "w") as f:
            yaml.dump(config, f)
            
        dag = WordcelDAG(self.dag_yaml_path)
        # Wordcel wraps errors in RuntimeError
        with pytest.raises(RuntimeError) as excinfo:
            dag.execute()
        assert "invalid input type" in str(excinfo.value)
        assert "pandas.DataFrame" in str(excinfo.value)
        assert "got dict" in str(excinfo.value)

    def test_baseline_missing_config(self):
        config = {
            "dag": {"name": "missing_config"},
            "nodes": [
                {"id": "node1", "type": "csv"} # missing path
            ]
        }
        with open(self.dag_yaml_path, "w") as f:
            yaml.dump(config, f)
            
        with pytest.raises(ValueError) as excinfo:
            WordcelDAG(self.dag_yaml_path)
        assert "requires a 'path' configuration" in str(excinfo.value)
