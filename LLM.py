from ollama import Client
from helper import get_ollama_api_key, load_env 

# Model token limits 
model_limits = {
    "gpt-oss:120b-cloud": 120000
}
# Reserve tokens for system prompt, user prompt, and resonse, as well as approximate character count per token.
reserved_tokens = 4000
char_count = 4

# Provides an estimate of the token count for a given piece of text
def estimate_tokens(text):
    return len(text) // char_count

# Returns the available tokens for a given model
def get_token_budget(model):
    limit = model_limits.get(model)
    return limit - reserved_tokens


def _content_to_str(content):
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        texts = []
        for part in content:
            if isinstance(part, dict) and "text" in part:
                texts.append(str(part["text"]))
            else:
                texts.append(str(part))
        return "".join(texts)

    return str(content)


def normalize_for_ollama(messages):
    norm = []
    for m in messages:
        role = m.get("role", "user")
        content = _content_to_str(m.get("content", ""))
        norm.append({"role": role, "content": content})
    return norm


def API_helper(messages):
    client = Client(host="http://localhost:11434")
    messages = normalize_for_ollama(messages)

    resp = client.chat(
        model="gpt-oss:120b-cloud",
        messages=messages,
        options={
            "num_predict": 4096
        }
    )
    output = resp["message"]["content"]
    return output