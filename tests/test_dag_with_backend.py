import os
import yaml
import pandas as pd
import pytest
from wordcel.dag import WordcelDAG


class TestDAGWithBackend:
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Set up and clean up test files."""
        # Test files
        self.test_csv_path = "test_backend.csv"
        self.dag_yaml_path = "test_backend_dag.yaml"
        self.cache_dir = "test_backend_cache"

        # Clean up any existing files
        test_files = [self.test_csv_path, self.dag_yaml_path]
        for file_path in test_files:
            if os.path.exists(file_path):
                os.remove(file_path)

        if os.path.exists(self.cache_dir):
            import shutil

            shutil.rmtree(self.cache_dir)

        yield

        # Clean up after test
        for file_path in test_files:
            if os.path.exists(file_path):
                os.remove(file_path)

        if os.path.exists(self.cache_dir):
            import shutil

            shutil.rmtree(self.cache_dir)

    def create_test_dag_config(self):
        """Create a DAG configuration for testing."""
        return {
            "dag": {
                "name": "test_backend_dag",
                "backend": {"type": "local", "cache_dir": self.cache_dir},
            },
            "nodes": [
                {"id": "csv_node", "type": "csv", "path": self.test_csv_path},
                {
                    "id": "operation_node",
                    "type": "dataframe_operation",
                    "input": "csv_node",
                    "operation": "head",
                    "kwargs": {"n": 2},
                },
            ],
        }

    def test_dag_with_backend_basic_execution(self):
        """Test basic DAG execution with backend caching."""
        # Create test CSV file
        with open(self.test_csv_path, "w") as f:
            f.write("col1,col2\n1,a\n2,b\n3,c")

        # Create DAG configuration
        dag_config = self.create_test_dag_config()
        with open(self.dag_yaml_path, "w") as f:
            yaml.dump(dag_config, f)

        # Execute DAG
        dag = WordcelDAG(self.dag_yaml_path)
        results = dag.execute()

        # Assertions
        assert "csv_node" in results
        assert "operation_node" in results
        assert isinstance(results["csv_node"], pd.DataFrame)
        assert isinstance(results["operation_node"], pd.DataFrame)
        assert len(results["csv_node"]) == 3  # Original data has 3 rows
        assert len(results["operation_node"]) == 2  # Head operation returns 2 rows
        assert list(results["csv_node"].columns) == ["col1", "col2"]

    def test_dag_backend_caching(self):
        """Test that backend caching works correctly."""
        # Create test CSV file
        with open(self.test_csv_path, "w") as f:
            f.write("col1,col2\n1,a\n2,b\n3,c")

        # Create DAG configuration
        dag_config = self.create_test_dag_config()
        with open(self.dag_yaml_path, "w") as f:
            yaml.dump(dag_config, f)

        # First execution
        dag = WordcelDAG(self.dag_yaml_path)
        first_results = dag.execute()

        # Cache directory should be created
        assert os.path.exists(self.cache_dir)

        # Second execution (should use cache)
        second_results = dag.execute()

        # Results should be identical
        pd.testing.assert_frame_equal(
            first_results["csv_node"], second_results["csv_node"]
        )
        pd.testing.assert_frame_equal(
            first_results["operation_node"], second_results["operation_node"]
        )

    def test_dag_backend_cache_behavior_with_file_changes(self):
        """Test current cache behavior when input files change (cache persists)."""
        # Create initial test CSV file
        with open(self.test_csv_path, "w") as f:
            f.write("col1,col2\n1,a\n2,b\n3,c")

        # Create DAG configuration
        dag_config = self.create_test_dag_config()
        with open(self.dag_yaml_path, "w") as f:
            yaml.dump(dag_config, f)

        # First execution
        dag = WordcelDAG(self.dag_yaml_path)
        first_results = dag.execute()

        # Verify first results
        assert len(first_results["csv_node"]) == 3
        assert first_results["csv_node"].iloc[0]["col1"] == 1

        # Modify the CSV file
        with open(self.test_csv_path, "w") as f:
            f.write("col1,col2\n4,d\n5,e\n6,f")

        # Second execution with modified file
        new_results = dag.execute()

        # Current behavior: cache persists despite file changes
        # This documents the current limitation of the caching system
        assert len(new_results["csv_node"]) == 3
        assert new_results["csv_node"].iloc[0]["col1"] == 1  # Still returns cached data
        assert first_results["csv_node"].equals(new_results["csv_node"])
