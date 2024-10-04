# LLM APIs

For convenience, I provide a few minimal wrapper functions for various LLM APIs.

## anthropic_call

```python
def anthropic_call(
    prompt,
    system_prompt="You are a helpful assistant.",
    model="claude-3-haiku-20240307",
    temperature=0,
    max_tokens=1024
)
```

Wrapper function for Anthropic's completion API.

**Parameters:**
- `prompt` (str): The user's input prompt.
- `system_prompt` (str, optional): System message to set context. Default: "You are a helpful assistant."
- `model` (str, optional): The model to use. Default: "claude-3-haiku-20240307"
- `temperature` (float, optional): Controls randomness in output. Default: 0
- `max_tokens` (int, optional): Maximum number of tokens in the response. Default: 1024

**Returns:**
- str: The generated text response.

**Example:**
```python
from wordcel.llm_providers import anthropic_call
response = anthropic_call("What is the capital of France?")
print(response)
```

## openai_call

```python
def openai_call(
    prompt,
    system_prompt=None,
    model="gpt-4o-mini",
    temperature=0,
    max_tokens=1024,
    stop=None,
    base_url=None,
    api_key=None
)
```

Wrapper function for OpenAI's completion API.

**Parameters:**
- `prompt` (str): The user's input prompt.
- `system_prompt` (str, optional): System message to set context. Default: None
- `model` (str, optional): The model to use. Default: "gpt-4o-mini"
- `temperature` (float, optional): Controls randomness in output. Default: 0
- `max_tokens` (int, optional): Maximum number of tokens in the response. Default: 1024
- `stop` (str or list, optional): Sequences where the API will stop generating further tokens. Default: None
- `base_url` (str, optional): Custom API base URL. Default: None
- `api_key` (str, optional): Custom API key. Default: None

**Returns:**
- str: The generated text response.

**Example:**
```python
from wordcel.llm_providers import openai_call
response = openai_call("Explain quantum computing in simple terms.")
print(response)
```

# gemini_call

```python
def gemini_call(
    prompt,
    system_prompt=None,
    model="gemini-1.5-flash",
    temperature=0,
    max_tokens=8192
)
```

Wrapper function for Google Gemini's text generation API.

**Parameters:**
- `prompt` (str): The user's input prompt.
- `system_prompt` (str, optional): System message to set context. Default: None
- `model` (str, optional): The model to use. Default: "gemini-1.5-flash"
- `temperature` (float, optional): Controls randomness in output. Default: 0
- `max_tokens` (int, optional): Maximum number of tokens in the response. Default: 8192

**Returns:**
- str: The generated text response.

**Note:** Requires `GEMINI_API_KEY` environment variable to be set.

**Example:**
```python
from wordcel.llm_providers import gemini_call
response = gemini_call("What are the main differences between Python and JavaScript?")
print(response)
```

# openai_embed

```python
from wordcel.llm_providers import openai_embed
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
embedding = openai_embed("Hello, world!")
print(len(embedding))  # Print the dimensionality of the embedding
```
