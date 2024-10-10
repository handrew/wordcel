import os
import anthropic
import openai
import google.generativeai as genai


def anthropic_call(
    prompt,
    system_prompt="You are a helpful assistant.",
    model="claude-3-haiku-20240307",
    temperature=0,
    max_tokens=1024,
):
    """Wrapper over Anthropic's completion API."""
    client = anthropic.Anthropic()
    message = client.messages.create(
        model=model,
        system=system_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[
            {"role": "user", "content": prompt},
        ],
    )

    text_response = message.content[0].text
    return text_response


def openai_call(
    prompt,
    system_prompt=None,
    model="gpt-4o-mini",
    temperature=0,
    max_tokens=1024,
    stop=None,
    base_url=None,
    api_key=None,
):
    """Wrapper over OpenAI's completion API."""
    # Assert both base_url and api_key are set or neither is set.
    assert (base_url is None) == (
        api_key is None
    ), "Both api_base and api_key must be set or neither should be set."
    client = openai.OpenAI()
    if base_url is not None:
        client = openai.OpenAI(base_url=base_url, api_key=api_key)

    messages = [{"role": "user", "content": prompt}]
    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        temperature=temperature,
        stop=stop,
    )
    text = response.choices[0].message.content

    return text


def gemini_call(
    prompt,
    system_prompt=None,
    model="gemini-1.5-flash",
    temperature=0,
    max_tokens=8192,
):
    """Wrapper over Google Gemini's text generation API."""
    if "GEMINI_API_KEY" not in os.environ:
        raise ValueError("GEMINI_API_KEY environment variable not set.")

    genai.configure(api_key=os.environ["GEMINI_API_KEY"])

    generation_config = {
        "temperature": temperature,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": max_tokens,
        "response_mime_type": "text/plain",
    }

    if system_prompt is None:
        model = genai.GenerativeModel(
            model_name=model,
            generation_config=generation_config,
        )
    else:
        model = genai.GenerativeModel(
            model_name=model,
            generation_config=generation_config,
            system_instruction=system_prompt,
        )

    # Start the chat session
    chat_session = model.start_chat()

    # Send the message and get the response
    response = chat_session.send_message(prompt)
    text = response.text

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
