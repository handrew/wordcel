import pandas as pd
from wordcel.featurize import apply_io_bound_function
from wordcel.llms import llm_call


def sentiment_classify(text: str) -> str:
    prompt = f"Classify the sentiment of the following text into one of two categories, POS or NEG. Respond in one word only.\n\n{text}"
    return llm_call(prompt, model="openai/gpt-3.5-turbo", max_tokens=32)


if __name__ == "__main__":
    # Sample data
    data = {
        "id": [1, 2, 3, 4, 5],
        "text": [
            "I love this product! It's amazing.",
            "The service was terrible. I'm very disappointed.",
            "The weather today is just perfect.",
            "This movie is fantastic. I highly recommend it.",
            "I had a bad experience with this company's customer support.",
        ],
    }
    df = pd.DataFrame(data)

    # Apply the function
    results = apply_io_bound_function(
        df,
        sentiment_classify,
        text_column="text",
        num_threads=2,
        cache_folder="cache",
    )
    print("Results:")
    print(results)

    # Join the results with the original DataFrame
    df["results"] = results 
    print("\nJoined Results:")
    print(df)
