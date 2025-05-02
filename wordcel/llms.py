"""LLM API wrappers for OpenAI, Anthropic, and Gemini."""

import os
import openai
from openai import AsyncOpenAI
from openai.types.shared import Reasoning
from agents import Agent, ModelSettings, Runner, RunConfig
from agents import ModelProvider, Model, OpenAIChatCompletionsModel
from agents import (
    set_default_openai_api,
    set_tracing_disabled,
)

SUPPORTED_PROVIDERS = ["openai", "anthropic", "google"]
GEMINI_BASE_API = "https://generativelanguage.googleapis.com/v1beta/openai/"
ANTHROPIC_BASE_API = "https://api.anthropic.com/v1/"
set_default_openai_api("chat_completions")
set_tracing_disabled(disabled=True)


def llm_call(prompt, model=None, **kwargs):
    """Router for openai_call, gemini_call, and anthropic_call."""
    assert model is not None, "Model name must be specified."
    provider, model = model.split("/")
    error_msg = f"Provider `{provider}` not supported. Supported providers: {SUPPORTED_PROVIDERS}. "
    error_msg += "Give your model in the form `<provider>/<model>`, like `openai/gpt-4o`."
    assert provider in SUPPORTED_PROVIDERS, error_msg

    if provider == "openai":
        return openai_call(prompt, model=model, **kwargs)
    elif provider == "google":
        return openai_call(prompt, model=model, base_url=GEMINI_BASE_API, api_key=os.getenv("GEMINI_API_KEY"), **kwargs)
    elif provider == "anthropic":
        return openai_call(prompt, model=model, base_url=ANTHROPIC_BASE_API, api_key=os.getenv("ANTHROPIC_API_KEY"), **kwargs)
    else:
        raise ValueError(error_msg)


def openai_call(
    prompt,
    system_prompt=None,
    model=None,
    max_tokens=None,
    temperature=1,
    reasoning_effort=None,
    base_url=None,
    api_key=None,
    **kwargs,
):
    """Wrapper over OpenAI's completion API."""
    assert reasoning_effort is None or reasoning_effort in [
        "low",
        "medium",
        "high",
        "none",
    ]

    assert model is not None, "Model name must be specified."
    # Assert both base_url and api_key are set or neither is set.
    assert (base_url is None) == (
        api_key is None
    ), "Both api_base and api_key must be set or neither should be set."
    # Assert that the client isn't given if the base_url and api_key are set.
    if base_url is not None:
        client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        class CustomModelProvider(ModelProvider):
            def get_model(self, model_name: str | None) -> Model:
                return OpenAIChatCompletionsModel(model=model, openai_client=client)
    else:
        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    agent = Agent(
        name="Agent",
        instructions=system_prompt,
        model=model,
        model_settings=ModelSettings(
            reasoning=Reasoning(effort=reasoning_effort) if reasoning_effort else None,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=kwargs.get("top_p", 1),
            presence_penalty=kwargs.get("presence_penalty", 0),
        ),
    )

    messages = [{"role": "user", "content": prompt}]
    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})

    result = Runner.run_sync(
        agent,
        input=messages,
        run_config=RunConfig(model_provider=CustomModelProvider()) if base_url else None,
    )

    text = result.final_output

    return text


def openai_embed(text, model="text-embedding-3-small"):
    """Wrapper over OpenAI's embedding API."""
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
