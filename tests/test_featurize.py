import os
import pandas as pd
import pytest
from unittest.mock import patch
from wordcel.featurize import apply_io_bound_function


class TestFeaturize:
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Clean up cache folder before and after tests."""
        cache_folder = "cache"

        # Clean up before test
        if os.path.exists(cache_folder):
            import shutil

            shutil.rmtree(cache_folder)

        yield

        # Clean up after test
        if os.path.exists(cache_folder):
            import shutil

            shutil.rmtree(cache_folder)

    def test_apply_io_bound_function_basic(self):
        """Test basic functionality of apply_io_bound_function (mocked)."""
        # Create a simple mock function
        def mock_sentiment(text: str) -> str:
            if "love" in text.lower() or "amazing" in text.lower() or "perfect" in text.lower():
                return "POS"
            elif "terrible" in text.lower() or "disappointed" in text.lower():
                return "NEG"
            return "NEG"

        # Sample data
        data = {
            "id": [1, 2, 3],
            "text": [
                "I love this product! It's amazing.",
                "The service was terrible. I'm very disappointed.",
                "The weather today is just perfect.",
            ],
        }
        df = pd.DataFrame(data)

        # Apply the function
        results = apply_io_bound_function(
            df,
            mock_sentiment,
            text_column="text",
            num_threads=2,
            cache_folder="cache",
        )

        # Assertions
        assert isinstance(results, pd.Series)
        assert len(results) == 3
        assert all(isinstance(result, str) for result in results)
        assert results.iloc[0] == "POS"
        assert results.iloc[1] == "NEG"
        assert results.iloc[2] == "POS"

    def test_apply_io_bound_function_with_dataframe_join(self):
        """Test joining results back to original DataFrame (mocked)."""
        # Create a simple mock function
        def mock_sentiment(text: str) -> str:
            if "fantastic" in text.lower() or "recommend" in text.lower():
                return "POS"
            elif "bad" in text.lower():
                return "NEG"
            return "NEG"

        # Sample data
        data = {
            "id": [1, 2],
            "text": [
                "This movie is fantastic. I highly recommend it.",
                "I had a bad experience with this company's customer support.",
            ],
        }
        df = pd.DataFrame(data)

        # Apply the function
        results = apply_io_bound_function(
            df,
            mock_sentiment,
            text_column="text",
            num_threads=1,
            cache_folder="cache",
        )

        # Join results with original DataFrame
        df["results"] = results

        # Assertions
        assert "results" in df.columns
        assert len(df) == 2
        assert df["results"].notna().all()
        assert df["results"].iloc[0] == "POS"
        assert df["results"].iloc[1] == "NEG"
