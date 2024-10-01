"""WIP: Functions to help with RAG."""
import json
from .llm_providers import openai_call


def generate_similar_queries(query):
    """Generate similar queries to the input query."""
    prompt = f"""I would like you to generate queries conceptually and semantically similar
    to the following query, for use in a vector / embedding retrieval system:

    Query: {query}

    Please respond with a Python array which I can read with `json.loads()`, with no
    additional commentary or text. No need to provide backticks, I will read your response
    directly.
    """
    response = openai_call(prompt)
    similar_queries = json.loads(response)
    return similar_queries
