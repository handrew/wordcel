import os
import time
import anthropic
import openai
import google.generativeai as genai


def anthropic_call(prompt, system_prompt="You are a helpful assistant.", model="claude-3-haiku-20240307", temperature=0, max_tokens=1024, sleep=60):
    """Wrapper over Anthropic's completion API."""
    client = anthropic.Anthropic()
    message = client.messages.create(
        model=model,
        system=system_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[
            {
                "role": "user",
                "content": prompt
            },
        ]
    )

    text_response = message.content[0].text
    return text_response


def openai_call(
    prompt, system_prompt=None, model="gpt-3.5-turbo", temperature=0, max_tokens=1024, stop=None, sleep=60
):
    """Wrapper over OpenAI's completion API."""
    client = openai.OpenAI()
    try:
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

    # With the following code, you'd get
    # "TypeError: catching classes that do not inherit from BaseException is not allowed"
    # except (
    #     openai.RateLimitError,
    #     openai.APIError,
    #     openai.Timeout,
    #     openai.APIConnectionError,
    # ) as exc:
    # So we use this instead. But this does not seem quite right either?
    except Exception as exc:
        if "maximum context length" in str(exc):
            raise ValueError("Maximum context length exceeded.")
        print(exc)
        print("Error from OpenAI's API. Sleeping for a few seconds.")
        time.sleep(sleep)
        text = openai_call(
            prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
        )

    return text


def gemini_call(
    prompt, model_name="gemini-1.5-flash", temperature=0, max_tokens=8192, sleep=60
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

    model = genai.GenerativeModel(model_name=model_name, generation_config=generation_config)
    
    # Start the chat session 
    chat_session = model.start_chat()

    # Send the message and get the response
    response = chat_session.send_message(prompt)
    text = response.text 
    
    return text
