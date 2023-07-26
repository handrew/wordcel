# 😶 Wordcel

`wordcel` is a library of functions that help users create text-based features from large language models.

Today, it is comprised of a single function `apply_io_bound_function`, which is conceptually similar to Pandas' `.apply`. Because LLMs are primarily consumed via API, users are subject to network latency, errors, and inference costs. Accordingly, the aforementioned `wordcel` function differs from Pandas' `.apply` in that it handles (a) caching outputs from LLM providers and (b) threading, which increases speed.

However, it is up to the user to handle a variety of odds and ends such as API retrying (though `wordcel.llm_providers.openai_call` is provided as a convenience, and popular libraries like Langchain and Llama Index also provide support for retrying) and chunking text. 

## Installation

You can simply `pip install wordcel`.

## Example Usage

First import the relevant functions.

```python
import pandas as pd
from wordcel.featurize import apply_io_bound_function
from wordcel.llm_providers import openai_call
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

Finally, give `apply_io_bound_function` your df, function, column to process, a unique identifying column, and optionally the number of threads you'd like to use and a cache folder (if none is provided then one will be created for you). 

```python
results = apply_io_bound_function(
    df,
    sentiment_classify,
    text_column="text",
    id_column="id",
    num_threads=4,
    cache_folder="cache",
)
print(results)
joined_results = df.join(results.set_index("id"), on="id")
print()
print(joined_results)
```

This will output:

```
   id result
0   1    POS
1   2    NEG
2   3    POS
3   4    POS
4   5    NEG

   id                                               text result
0   1                 I love this product! It's amazing.    POS
1   2   The service was terrible. I'm very disappointed.    NEG
2   3                 The weather today is just perfect.    POS
3   4    This movie is fantastic. I highly recommend it.    POS
4   5  I had a bad experience with this company's cus...    NEG
```