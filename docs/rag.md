# RAG Utilities and Contextual Retrieval

This module provides utilities for Retrieval-Augmented Generation (RAG). 

There is also a minimal implementation of Anthropic's [Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval/), without the need for several paid / private APIs. It is an extremely stripped down version with no reranking, no BM-25 -- just TF-IDF, vector embeddings, and rank fusion. It's all in memory. You can persist the retriever to disk and load it back, but it's not optimized for speed, memory usage, or disk space. It is loosely inspired by [Anthropic's provided workbook](https://github.com/anthropics/anthropic-cookbook/blob/main/skills/contextual-embeddings/guide.ipynb).

## rag/utils.py

This file contains helper functions for RAG operations.

### generate_similar_queries

```python
def generate_similar_queries(query: str, llm_fn=openai_call) -> List[str]
```

Generates queries conceptually and semantically similar to the input query.

**Parameters:**
- `query` (str): The original query.
- `llm_fn` (function, optional): The language model function to use. Default: `openai_call`

**Returns:**
- List[str]: A list of similar queries.

### chunk_text

```python
def chunk_text(text: str, chunk_size: int = 200, chunk_overlap: int = 10) -> List[str]
```

Breaks down a text into chunks of a specified size.

**Parameters:**
- `text` (str): The input text to chunk.
- `chunk_size` (int, optional): The size of each chunk. Default: 1024
- `chunk_overlap` (int, optional): The overlap between chunks. Default: 10

**Returns:**
- List[str]: A list of text chunks.

### embed_chunks

```python
def embed_chunks(chunks: List[str], embed_fn=openai_embed) -> List[str]
```

Embeds a list of text chunks.

**Parameters:**
- `chunks` (List[str]): The list of text chunks to embed.
- `embed_fn` (function, optional): The embedding function to use. Default: `openai_embed`

**Returns:**
- List[str]: A list of embeddings.

### create_tfidf_index

```python
def create_tfidf_index(chunks: List[str])
```

Creates a TF-IDF index for a list of chunks.

**Parameters:**
- `chunks` (List[str]): The list of text chunks to index.

**Returns:**
- Tuple[TfidfVectorizer, scipy.sparse.csr_matrix]: The TF-IDF vectorizer and the TF-IDF matrix.

### cosine_similarity_vectorized

```python
def cosine_similarity_vectorized(vector: np.array, matrix: np.array) -> np.array
```

Computes cosine similarity between a vector and a matrix (vectorized).

**Parameters:**
- `vector` (np.array): The input vector.
- `matrix` (np.array): The input matrix.

**Returns:**
- np.array: An array of cosine similarities.

## rag/contextual_retrieval.py

This file contains the `ContextualRetrieval` class, which implements a minimal version of Contextual Retrieval.

```python
class ContextualRetrieval:
    def __init__(self, docs: List[str], chunk_size=1024, chunk_overlap=10, llm_fn=anthropic_call)
```

**Parameters:**
- `docs` (List[str]): List of documents to index.
- `chunk_size` (int, optional): Size of each chunk. Default: 1024
- `chunk_overlap` (int, optional): Overlap between chunks. Default: 10
- `llm_fn` (function, optional): Language model function to use. Default: `anthropic_call`

### Methods

#### from_documents

```python
@classmethod
def from_documents(cls, docs: List[str], llm_fn=anthropic_call) -> ContextualRetrieval
```

Creates a ContextualRetrieval instance from a list of documents.

#### from_saved

```python
@classmethod
def from_saved(cls, path: str) -> ContextualRetrieval
```

Loads a saved ContextualRetrieval instance from a file.

#### index_documents

```python
def index_documents(self, situate=False)
```

Indexes the documents by chunking, embedding, and creating a TF-IDF index. "Situating" each chunk is a computationally expensive set of inferences, so it is set to False by default.

#### save

```python
def save(self, path: str)
```

Saves the current state of the retriever to a file.

#### load

```python
def load(self, path: str)
```

Loads a saved retriever from a file.

#### retrieve

```python
def retrieve(self, query: str, top_k: int = 5, semantic_weight=0.5, tfidf_weight=0.5) -> List[Dict]
```

Retrieves the top-k chunks for a given query.

**Parameters:**
- `query` (str): The search query.
- `top_k` (int, optional): Number of top results to return. Default: 5
- `semantic_weight` (float, optional): Weight for semantic similarity. Default: 0.5
- `tfidf_weight` (float, optional): Weight for TF-IDF similarity. Default: 0.5

**Returns:**
- List[Dict]: List of dictionaries containing retrieved chunks and their metadata.

#### generate

```python
def generate(self, search_query: str, generation_query: str = None, top_k: int = 5, semantic_weight=0.5, tfidf_weight=0.5, llm_fn=anthropic_call) -> str
```

Retrieves top-k chunks using `search_query` and generates a response using `generation_query`.

**Parameters:**
- `search_query` (str): The query for retrieving chunks.
- `generation_query` (str, optional): The query for generating the response. If None, uses `search_query`.
- `top_k` (int, optional): Number of top results to use. Default: 5
- `semantic_weight` (float, optional): Weight for semantic similarity. Default: 0.5
- `tfidf_weight` (float, optional): Weight for TF-IDF similarity. Default: 0.5
- `llm_fn` (function, optional): Language model function to use. Default: `anthropic_call`

**Returns:**
- str: The generated response.
