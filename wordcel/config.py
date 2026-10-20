"""Configuration constants for Wordcel."""

# Default model configurations by provider
DEFAULT_GOOGLE_MODEL = "gemini/gemini-3.8-flash"
DEFAULT_OPENAI_MODEL = "openai/gpt-6.1-sol"
DEFAULT_ANTHROPIC_MODEL = "anthropic/claude-opus-5-5"

DEFAULT_MODELS = {
    "gemini": DEFAULT_GOOGLE_MODEL,
    "google": DEFAULT_GOOGLE_MODEL,
    "openai": DEFAULT_OPENAI_MODEL,
    "anthropic": DEFAULT_ANTHROPIC_MODEL,
}

# Default model configuration
DEFAULT_MODEL = DEFAULT_GOOGLE_MODEL
