"""
Minimal implementation of a version of Contextual Retrieval from Anthropic
without the need for several paid / private APIs.

No reranking, no BM-25. Just TF-IDF, vector embeddings, rank fusion.

It's all in memory. You can persist the retriever to disk and load it back,
but it's not optimized for speed, memory usage, or disk space.

https://www.anthropic.com/news/contextual-retrieval/

Reference workbook:
https://github.com/anthropics/anthropic-cookbook/blob/main/skills/contextual-embeddings/guide.ipynb
"""

import pickle
import logging
import pandas as pd
from typing import List, Dict
import anthropic
from .utils import (
    chunk_text,
    embed_chunks,
    create_tfidf_index,
    cosine_similarity_vectorized,
)
from ..llm_providers import anthropic_call

log: logging.Logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DOCUMENT_CONTEXT_PROMPT = """
<document>
{doc_content}
</document>
"""

CHUNK_CONTEXT_PROMPT = """
Here is the chunk we want to situate within the whole document
<chunk>
{chunk_content}
</chunk>
Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk.
Answer only with the succinct context and nothing else.
"""


def situate_context(doc: str, chunk: str) -> str:
    """Situate a chunk within a document."""
    client = anthropic.Anthropic()
    response = client.beta.prompt_caching.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=1024,
        temperature=0.0,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": DOCUMENT_CONTEXT_PROMPT.format(doc_content=doc),
                        "cache_control": {"type": "ephemeral"},
                    },
                    {
                        "type": "text",
                        "text": CHUNK_CONTEXT_PROMPT.format(chunk_content=chunk),
                    },
                ],
            }
        ],
        extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"},
    )
    return response.content[0].text


class ContextualRetrieval:
    def __init__(
        self, docs: List[str], chunk_size=200, chunk_overlap=10, llm_fn=anthropic_call
    ):
        self.docs = docs
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.llm_fn = llm_fn

        # To be populated by index_documents() or load().
        self.chunks = None  # Dataframe of (doc_idx, chunk_idx, chunk) tuples.
        self.embeddings = None
        self.tfidf_vectorizer = None
        self.tfidf_index = None

    @classmethod
    def from_documents(cls, docs: List[str], llm_fn=anthropic_call):
        instance = cls(docs, llm_fn)
        instance.index_documents()
        return instance

    @classmethod
    def from_saved(cls, path: str):
        instance = cls([], None)  # Create an empty instance
        instance.load(path)
        return instance

    def index_documents(self):
        """Index the documents by chunking, embedding, and creating a TF-IDF index."""
        log.info("Indexing documents...")
        if self.chunks is None:
            log.info("Chunking documents...")
            chunks = [
                (
                    doc_idx,
                    chunk_text(
                        doc,
                        chunk_size=self.chunk_size,
                        chunk_overlap=self.chunk_overlap,
                    ),
                )
                for doc_idx, doc in enumerate(self.docs)
            ]

            log.info("Situating chunks within documents...")
            situated_chunks = []
            chunk_idx = 0
            for doc_idx, doc_chunks in chunks:
                for doc_chunk in doc_chunks:
                    situated_chunk = (
                        situate_context(self.docs[doc_idx], doc_chunk)
                        + "\n\n"
                        + doc_chunk
                    )
                    situated_chunks.append((doc_idx, chunk_idx, situated_chunk))
                    chunk_idx += 1
            self.chunks = pd.DataFrame(
                situated_chunks, columns=["doc_idx", "chunk_idx", "chunk"]
            )

        if self.embeddings is None:
            log.info("Embedding chunks...")
            self.embeddings = embed_chunks(self.chunks["chunk"].tolist())

        if self.tfidf_index is None:
            log.info("Creating TF-IDF index...")
            self.tfidf_vectorizer, self.tfidf_index = create_tfidf_index(
                self.chunks["chunk"].tolist()
            )

    def save(self, path: str):
        """Save the current state of the retriever to a file."""
        # Save a pickled version of the chunks, embeddings, and TF-IDF index.
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "chunks": self.chunks.to_dict(orient="records"),
                    "embeddings": self.embeddings,
                    "tfidf_index": self.tfidf_index,
                    "tfidf_vectorizer": self.tfidf_vectorizer,
                },
                f,
            )

    def load(self, path: str):
        """Load a saved retriever from a file."""
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.chunks = pd.DataFrame(data["chunks"])
            self.embeddings = data["embeddings"]
            self.tfidf_index = data["tfidf_index"]
            self.tfidf_vectorizer = data["tfidf_vectorizer"]

    def retrieve(
        self, query: str, top_k: int = 5, semantic_weight=0.5, tfidf_weight=0.5
    ) -> List[Dict]:
        """Retrieve the top-k chunks for a given query."""
        if self.chunks is None or self.embeddings is None or self.tfidf_index is None:
            raise ValueError(
                "Documents have not been processed. Call index_documents() first."
            )

        query_embedding = embed_chunks([query])[0]
        query_tfidf = self.tfidf_vectorizer.transform([query]).toarray().squeeze(0)
        semantic_similarities = cosine_similarity_vectorized(
            query_embedding, self.embeddings
        )
        tfidf_similarities = cosine_similarity_vectorized(
            query_tfidf, self.tfidf_index.toarray()
        )

        semantic_ranks = semantic_similarities.argsort()[::-1][:top_k].tolist()
        tfidf_ranks = tfidf_similarities.argsort()[::-1][:top_k].tolist()

        # Combine and score the chunks. Rank fusion.
        chunk_ids = list(set(semantic_ranks + tfidf_ranks))
        chunk_id_to_score = {}
        for chunk_id in chunk_ids:
            score = 0
            if chunk_id in semantic_ranks:
                idx = semantic_ranks.index(chunk_id)
                score += semantic_weight * (
                    1 / (idx + 1)
                )  # Weighted 1/n scoring for semantic.

            if chunk_id in tfidf_ranks:
                idx = tfidf_ranks.index(chunk_id)
                score += tfidf_weight * (1 / (idx + 1))

            chunk_id_to_score[chunk_id] = score

        # Sort chunk IDs by their scores in descending order.
        sorted_chunk_ids = sorted(
            chunk_id_to_score.keys(), key=lambda x: chunk_id_to_score[x], reverse=True
        )

        # Assign new scores based on the sorted order.
        for index, chunk_id in enumerate(sorted_chunk_ids):
            chunk_id_to_score[chunk_id] = 1 / (index + 1)

        # Prepare final results.
        final_results = []
        for chunk_id in sorted_chunk_ids[:top_k]:
            doc_idx, chunk_idx = self.chunks.iloc[chunk_id][["doc_idx", "chunk_idx"]]
            final_results.append(
                {
                    "doc_idx": doc_idx,
                    "chunk_idx": chunk_idx,
                    "chunk": self.chunks.iloc[chunk_id]["chunk"],
                    "semantic_similarity": semantic_similarities[chunk_id],
                    "tfidf_similarity": tfidf_similarities[chunk_id],
                    "score": chunk_id_to_score[chunk_id],
                }
            )

        return final_results

    def generate(
        self,
        search_query: str,
        generation_query: str = None,
        top_k: int = 5,
        semantic_weight=0.5,
        tfidf_weight=0.5,
        llm_fn=anthropic_call,
    ) -> str:
        """Retrieves top-k chunks using `search_query` and generates the
        response given those chunks using `generation_query` if given or
        `search_query` if not.

        This function is only given as an example. You should likely customize
        your own generation function based on your use case."""
        retrieved_chunks = self.retrieve(
            search_query,
            top_k=top_k,
            semantic_weight=semantic_weight,
            tfidf_weight=tfidf_weight,
        )
        if generation_query is None:
            generation_query = search_query

        prompt = ""
        for chunk in retrieved_chunks:
            prompt += f"{chunk}\n\n"

        prompt += f"Given the above information, answer the following query:\n\n{generation_query}"

        response = llm_fn(prompt)
        return response
