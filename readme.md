# 😶 Wordcel

`wordcel` is a library of functions that provides a set of common tools for working with large language models.

Candidly, it is mostly a set of functions that I myself use on a regular basis -- my own personal swiss army knife. 

## Installation

You can simply `pip install wordcel`.

## Documentation

### LLM providers

For convenience, I provide a number of functions to call popular APIs. 

```python
from wordcel.llm_providers import openai_call, anthropic_call, gemini_call
from wordcel.llm_providers import openai_embed

openai_call("hey")
```

They are extremely bare, so it is up to the user to handle a variety of odds and ends such as API retrying.

### Contextual Retrieval

I provide a minimal implementation of a version of Anthropic's [Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval/), without the need for several paid / private APIs. It is an extremely stripped down version with no reranking, no BM-25 -- just TF-IDF, vector embeddings, and rank fusion.

It's all in memory. You can persist the retriever to disk and load it back, but it's not optimized for speed, memory usage, or disk space. 

It is loosely inspired by [Anthropic's provided workbook](https://github.com/anthropics/anthropic-cookbook/blob/main/skills/contextual-embeddings/guide.ipynb).

```python
from wordcel.rag.contextual_retrieval import ContextualRetrieval
retriever = ContextualRetrieval(docs)
retriever.index_documents()
print(retriever.retrieve("<your search_query>"))
retriever.save("retriever.pkl")
```

### Fast LLM pandas apply

The primary function here is `apply_io_bound_function`, which is conceptually similar to Pandas' `.apply`. Because LLMs are primarily consumed via API, users are subject to network latency, errors, and inference costs. Accordingly, the function differs from Pandas' `.apply` in that it handles (a) locally caching outputs from LLM providers and (b) threading, which increases speed.

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
