"""WIP: Functions to help with RAG."""

import json
import numpy as np
from typing import List
from langchain.text_splitter import CharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer
from ..llm_providers import openai_call, openai_embed


"""RAG helper functions."""


def generate_similar_queries(query, llm_fn=openai_call):
    """Generate similar queries to the input query."""
    prompt = f"""I would like you to generate queries conceptually and semantically similar
    to the following query, for use in a vector / embedding retrieval system:

    Query: {query}

    Please respond with a Python array which I can read with `json.loads()`, with no
    additional commentary or text. No need to provide backticks, I will read your response
    directly.
    """
    response = llm_fn(prompt)
    similar_queries = json.loads(response)
    return similar_queries


"""Embedding helper functions."""


def chunk_text(text: str, chunk_size: int = 200, chunk_overlap: int = 10) -> List[str]:
    """Break down a text into chunks of a specified size."""
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_text(text)
    return chunks


def embed_chunks(chunks: List[str], embed_fn=openai_embed) -> List[str]:
    """Embed a list of chunks."""
    assert isinstance(chunks, list), "Chunks must be a list."
    embeddings = []
    for chunk in chunks:
        embedding = embed_fn(chunk)
        embeddings.append(embedding)
    return np.array(embeddings)


def create_tfidf_index(chunks: List[str]):
    """Create a TF-IDF index for a list of chunks."""
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(chunks)
    return vectorizer, tfidf_matrix


def cosine_similarity_vectorized(vector, matrix) -> np.array:
    """Compute cosine similarity between a vector and a matrix. Vectorized."""
    norm_vector = np.linalg.norm(vector)
    norm_matrix = np.linalg.norm(matrix, axis=1)
    dot_product = np.dot(matrix, vector)
    cosine_similarities = dot_product / (norm_vector * norm_matrix)
    return cosine_similarities
