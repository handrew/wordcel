import os
import pandas as pd
import pytest
from wordcel.featurize import apply_io_bound_function
from wordcel.llms import llm_call
from wordcel.config import DEFAULT_MODEL


def sentiment_classify(text: str) -> str:
    """Helper function for sentiment classification."""
    prompt = f"Classify the sentiment of the following text into one of two categories, POS or NEG. Respond in one word only.\n\n{text}"
    return llm_call(prompt, model=DEFAULT_MODEL, max_tokens=32)


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
        """Test basic functionality of apply_io_bound_function."""
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
            sentiment_classify,
            text_column="text",
            num_threads=2,
            cache_folder="cache",
        )

        # Assertions
        assert isinstance(results, pd.Series)
        assert len(results) == 3
        assert all(isinstance(result, str) for result in results)
        assert all(
            result.strip() in ["POS", "NEG", "Positive", "Negative"]
            or "pos" in result.lower()
            or "neg" in result.lower()
            for result in results
        )

    def test_apply_io_bound_function_with_dataframe_join(self):
        """Test joining results back to original DataFrame."""
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
            sentiment_classify,
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
        assert isinstance(df["results"].iloc[0], str)
        assert isinstance(df["results"].iloc[1], str)
