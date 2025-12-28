from ollama import Client
from helper import get_ollama_api_key, load_env  

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
        # If history is tuples, you'll handle that earlier — this assumes dict messages here
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
    )
    output = resp["message"]["content"]
    return output