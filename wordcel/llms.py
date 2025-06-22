"""LLM API wrapper for LiteLLM."""

import os
from typing import Any, Dict, List, Optional

import openai
import litellm

SUPPORTED_PROVIDERS = ["openai", "anthropic", "gemini"]


def llm_call(prompt: str, model: Optional[str] = None, **kwargs: Any) -> str:
    """Wrapper for OAI, Anthropic, and Google calls using litellm.

    Args:
        prompt: The input prompt for the LLM
        model: Model name in format '<provider>/<model>' (e.g., 'openai/gpt-4o')
        **kwargs: Additional parameters like temperature, max_tokens, system_prompt

    Returns:
        The LLM's response content as a string

    Raises:
        AssertionError: If model is None or not in correct format
    """

    assert model is not None, "Model name must be specified."
    assert "/" in model, "Model name must be in the form `<provider>/<model>`."
    provider, model_name = model.split("/")
    error_msg = f"Provider `{provider}` not supported. Supported providers: {SUPPORTED_PROVIDERS}. "
    error_msg += (
        "Give your model in the form `<provider>/<model>`, like `openai/gpt-4o`."
    )
    assert provider in SUPPORTED_PROVIDERS, error_msg

    # Create messages list
    messages = [{"role": "user", "content": prompt}]
    if "system_prompt" in kwargs and kwargs["system_prompt"]:
        messages.insert(0, {"role": "system", "content": kwargs.pop("system_prompt")})

    # Set API keys if not already in environment
    if provider == "anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_key:
            os.environ["ANTHROPIC_API_KEY"] = anthropic_key
    elif provider == "google" and not os.getenv("GOOGLE_API_KEY"):
        google_key = os.getenv("GEMINI_API_KEY")
        if google_key:
            os.environ["GOOGLE_API_KEY"] = google_key

    # Prepare litellm parameters
    litellm_model = f"{provider}/{model_name}"
    litellm_params = {
        "model": litellm_model,
        "messages": messages,
    }

    # Map common parameters
    if "temperature" in kwargs:
        litellm_params["temperature"] = kwargs.pop("temperature")
    if "max_tokens" in kwargs:
        litellm_params["max_tokens"] = kwargs.pop("max_tokens")
    if "reasoning_effort" in kwargs and kwargs["reasoning_effort"] is not None:
        litellm_params["reasoning_effort"] = kwargs.pop("reasoning_effort")

    # Add any remaining kwargs
    litellm_params.update(kwargs)

    # Make the API call
    response = litellm.completion(**litellm_params)

    return response.choices[0].message.content


def openai_embed(text: str, model: str = "text-embedding-3-small") -> List[float]:
    """Wrapper over OpenAI's embedding API.

    Args:
        text: The text to embed
        model: OpenAI embedding model name

    Returns:
        List of float values representing the embedding vector
    """
    client = openai.OpenAI()
    response = (
        client.embeddings.create(
            input=[text],
            model=model,
        )
        .data[0]
        .embedding
    )
    return response
