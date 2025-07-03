# LLM API Wrappers

This package provides a unified function, `llm_call`, for interacting with various LLM APIs, including OpenAI, Anthropic, and Google Gemini, powered by the `litellm` library. It also provides a wrapper for OpenAI's embedding API.

## llm_call

```python
def llm_call(prompt: str, model: Optional[str] = None, **kwargs: Any) -> str:
```

Wrapper for OpenAI, Anthropic, and Google calls using `litellm`.

**Parameters:**
- `prompt` (str): The user's input prompt.
- `model` (str): The model to use, in the format `<provider>/<model>` (e.g., "openai/gpt-4o").
- `**kwargs`: Additional parameters passed to `litellm.completion`, such as:
  - `system_prompt` (str, optional): System message to set context.
  - `temperature` (float, optional): Controls randomness in output.
  - `max_tokens` (int, optional): Maximum number of tokens in the response.

**Supported Providers:**
- `openai`: For OpenAI models (e.g., "openai/gpt-4o").
- `anthropic`: For Anthropic models (e.g., "anthropic/claude-3-haiku-20240307").
- `gemini`: For Google Gemini models (e.g., "gemini/gemini-2.5-flash").

**Returns:**
- str: The generated text response from the language model.

**Example:**
```python
from wordcel.llms import llm_call
response = llm_call("What is the capital of France?", model="openai/gpt-4o")
print(response)
```

## openai_embed

```python
def openai_embed(text, model="text-embedding-3-small")
```

Wrapper function for OpenAI's embedding API.

**Parameters:**
- `text` (str): The input text to embed.
- `model` (str, optional): The embedding model to use. Default: "text-embedding-3-small"

**Returns:**
- list: The embedding vector for the input text.

**Example:**
```python
from wordcel.llms import openai_embed
embedding = openai_embed("Hello, world!")
print(len(embedding))  # Print the dimensionality of the embedding
```

## Environment Variables

The following environment variables are used by these functions:
- `OPENAI_API_KEY`: Required for OpenAI API calls.
- `ANTHROPIC_API_KEY`: Required for Anthropic API calls.
- `GEMINI_API_KEY`: Required for Google Gemini API calls.

## Usage with Different Providers

### OpenAI
```python
response = llm_call("Tell me a joke", model="openai/gpt-4o")
```

### Google Gemini
```python
response = llm_call("Explain machine learning", model="gemini/gemini-2.5-flash")
```

### Anthropic
```python
response = llm_call("Write a poem about nature", model="anthropic/claude-3-haiku-20240307")
```
