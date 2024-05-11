import time
import anthropic
import openai


def anthropic_call(prompt, model="claude-3-haiku-20240307", temperature=0, max_tokens=1024, sleep=60):
    """Wrapper over Anthropic's completion API."""
    client = anthropic.Anthropic()
    try:
        message = client.messages.create(
            model=model,
            max_tokens=1000,
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        }
                    ]
                }
            ]
        )
    except Exception as exc:
        print(exc)
        print("Error from Anthropic's API. Sleeping for a few seconds.")
        time.sleep(sleep)
        message = anthropic_call(
            prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    text_response = message.content[0].text
    return text_response


def openai_call(
    prompt, model="gpt-3.5-turbo", temperature=0, max_tokens=1024, stop=None, sleep=60
):
    """Wrapper over OpenAI's completion API."""
    client = openai.OpenAI()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
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
