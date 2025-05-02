# LLM API Wrappers

This package provides minimal wrapper functions for interacting with various LLM APIs, including OpenAI, Anthropic, and Google Gemini.

## llm_call

```python
def llm_call(prompt, model=None, **kwargs)
```

Router function that directs calls to the appropriate API based on the provider specified in the model name.

**Parameters:**
- `prompt` (str): The user's input prompt.
- `model` (str): Must be in the format `<provider>/<model>` (e.g., "openai/gpt-4o").
- `**kwargs`: Additional parameters passed to the specific API call function.

**Supported Providers:**
- `openai`: OpenAI models (e.g., "openai/gpt-4o")
- `google`: Google Gemini models (e.g., "google/gemini-1.5-flash")
- `anthropic`: Anthropic models (e.g., "anthropic/claude-3-haiku-20240307")

**Returns:**
- str: The generated text response.

**Example:**
```python
response = llm_call("What is the capital of France?", model="openai/gpt-4o")
print(response)
```

## openai_call

```python
def openai_call(
    prompt,
    system_prompt=None,
    model=None,
    max_tokens=None,
    temperature=1,
    reasoning_effort=None,
    base_url=None,
    api_key=None,
    **kwargs
)
```

Wrapper function for OpenAI's completion API. This function is also used internally for Google and Anthropic API calls.

**Parameters:**
- `prompt` (str): The user's input prompt.
- `system_prompt` (str, optional): System message to set context. Default: None
- `model` (str): The model to use.
- `max_tokens` (int, optional): Maximum number of tokens in the response. Default: None
- `temperature` (float, optional): Controls randomness in output. Default: 1
- `reasoning_effort` (str, optional): Level of reasoning effort ("low", "medium", "high", or "none"). Default: None
- `base_url` (str, optional): Custom API base URL. Default: None
- `api_key` (str, optional): Custom API key. Default: None
- `**kwargs`: Additional parameters including:
  - `top_p` (float, optional): Controls diversity via nucleus sampling. Default: 1
  - `presence_penalty` (float, optional): Penalizes repeated tokens. Default: 0

**Returns:**
- str: The generated text response.

**Note:** When using `base_url` and `api_key`, both must be provided together.

**Example:**
```python
from wordcel.llms import openai_call
response = openai_call(
    "Explain quantum computing in simple terms.",
    model="gpt-4o",
    system_prompt="You are a science educator for beginners.",
    temperature=0.7
)
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
- `OPENAI_API_KEY`: Required for OpenAI API calls
- `ANTHROPIC_API_KEY`: Required for Anthropic API calls
- `GEMINI_API_KEY`: Required for Google Gemini API calls

## Usage with Different Providers

### OpenAI
```python
response = llm_call("Tell me a joke", model="openai/gpt-4o")
```

### Google Gemini
```python
response = llm_call("Explain machine learning", model="google/gemini-2.0-flash")
```

### Anthropic
```python
response = llm_call("Write a poem about nature", model="anthropic/claude-3-haiku-20240307")
```