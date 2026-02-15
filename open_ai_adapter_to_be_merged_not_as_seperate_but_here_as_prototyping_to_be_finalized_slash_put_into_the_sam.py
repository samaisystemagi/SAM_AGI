from openai import OpenAI

# The client automatically picks up the OPENAI_API_KEY environment variable
client = OpenAI()


def chat_with_gpt(prompt):
    """Sends a prompt to the ChatGPT API and returns the response."""
    # We use the chat completions endpoint and a chat model like gpt-3.5-turbo
    # Note use latest model availble here https://platform.openai.com/docs/models
    # Also Codex is prefered but we can fallback to the latest model pro or fallback to other ... etc
    response = client.chat.completions.create(
        model="gpt-5.3-pro",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content


# Example usage
user_prompt = input("How can I help you today? ")
api_response = chat_with_gpt(user_prompt)
print(f"\nChatGPT: {api_response}")
