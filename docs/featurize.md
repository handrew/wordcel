# Fast LLM pandas apply

This module provides `apply_io_bound_function`, a function to help rapidly featurize text using concurrent processing and caching. It is conceptually similar to Pandas' `.apply`. Because LLMs are primarily consumed via API, users are subject to network latency, errors, and inference costs. Accordingly, the function differs from Pandas' `.apply` in that it handles (a) locally caching outputs from LLM providers and (b) threading, which increases speed.


## apply_io_bound_function

```python
def apply_io_bound_function(
    df,
    user_function,
    text_column,
    num_threads=4,
    cache_folder=None
)
```

Apply an I/O bound user-provided function to a specific column in a Pandas DataFrame with threading and caching.

**Parameters:**
- `df` (pd.DataFrame): Pandas DataFrame containing the data.
- `user_function` (function): User-provided function that takes text as input and returns a JSON.
- `text_column` (str): Name of the column containing the text to process.
- `num_threads` (int, optional): Number of threads for concurrent processing. Default: 4
- `cache_folder` (str, optional): Folder to store the cached JSON outputs. Default: None

**Returns:**
- pd.Series: A series of the results, index-aligned with the original df.


**Example:**

First import the relevant functions.

```python
import pandas as pd
from wordcel.featurize import apply_io_bound_function
from wordcel.llms import openai_call
```

Then, load your data.

```python
data = {
    "id": [1, 2, 3, 4, 5],
    "text": [
        "I love this product! It's amazing.",
        "The service was terrible. I'm very disappointed.",
        "The weather today is just perfect.",
        "This movie is fantastic. I highly recommend it.",
        "I had a bad experience with this company's customer support.",
    ],
}
df = pd.DataFrame(data)
```

Define your LLM function for extracting features from the text.

```python
def sentiment_classify(text):
    prompt = f"Classify the sentiment of the following text into one of two categories, POS or NEG. Respond in one word only.\n\n{text}"
    return openai_call(prompt, model="gpt-3.5-turbo", max_tokens=32)
```

Finally, give `apply_io_bound_function` your df, function, column to process, and optionally the number of threads you'd like to use and a cache folder (if none is provided then one will be created for you). The function uses the DataFrame's index to create unique cache files.

```python
results = apply_io_bound_function(
    df,
    sentiment_classify,
    text_column="text",
    num_threads=4,
    cache_folder="cache",
)
print(results)
df["results"] = results 
print("\nJoined Results:")
print(df)
```

This will output:

```
0    POS
1    NEG
2    POS
3    POS
4    NEG
Name: text, dtype: object

Joined Results:
   id                                               text results
0   1                 I love this product! It's amazing.     POS
1   2   The service was terrible. I'm very disappointed.     NEG
2   3                 The weather today is just perfect.     POS
3   4    This movie is fantastic. I highly recommend it.     POS
4   5  I had a bad experience with this company's cus...     NEG
```
