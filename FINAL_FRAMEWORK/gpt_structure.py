"""
gpt_structure.py — Multi-Provider Drop-in Replacement
══════════════════════════════════════════════════════
Replaces the original OpenAI-only gpt_structure.py from the Smallville
generative agents repo. All function signatures are preserved so that
run_gpt_prompt.py and the cognitive modules work without modification.

Routes through llm_helper.py which supports:
  Anthropic (Claude) | OpenAI (GPT) | Google Gemini | Ollama (local)

Configuration: set env vars as described in llm_helper.py
"""

import json
import time
import sys
import os

# Add project root to path so we can import llm_helper
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
# Also try two levels up (for when run from persona/prompt_template/)
_alt_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _alt_root not in sys.path:
    sys.path.insert(0, _alt_root)

from llm_helper import call_llm, PROVIDER, MODEL

# ─────────────────────────────────────────────────────────────
# Rate limiting
# ─────────────────────────────────────────────────────────────
def temp_sleep(seconds=0.1):
    time.sleep(seconds)


# ─────────────────────────────────────────────────────────────
# Core request functions — all route through call_llm()
# ─────────────────────────────────────────────────────────────
def ChatGPT_single_request(prompt):
    """Single-shot chat request. Used throughout run_gpt_prompt.py."""
    temp_sleep()
    return call_llm(
        system_prompt="",
        messages=[{"role": "user", "content": prompt}],
    )


def GPT4_request(prompt):
    """GPT-4 level request — now routes to whatever provider is configured."""
    temp_sleep()
    try:
        return call_llm(
            system_prompt="",
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as e:
        print(f"LLM ERROR: {e}")
        return "LLM ERROR"


def ChatGPT_request(prompt):
    """ChatGPT request — now routes to whatever provider is configured."""
    try:
        return call_llm(
            system_prompt="",
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as e:
        print(f"LLM ERROR: {e}")
        return "LLM ERROR"


# ─────────────────────────────────────────────────────────────
# Safe generation with validation/retry — signatures preserved
# ─────────────────────────────────────────────────────────────
def GPT4_safe_generate_response(prompt,
                                example_output,
                                special_instruction,
                                repeat=3,
                                fail_safe_response="error",
                                func_validate=None,
                                func_clean_up=None,
                                verbose=False):
    prompt = 'GPT-3 Prompt:\n"""\n' + prompt + '\n"""\n'
    prompt += f"Output the response to the prompt above in json. {special_instruction}\n"
    prompt += "Example output json:\n"
    prompt += '{"output": "' + str(example_output) + '"}'

    if verbose:
        print("LLM PROMPT")
        print(prompt)

    for i in range(repeat):
        try:
            curr_gpt_response = GPT4_request(prompt).strip()
            end_index = curr_gpt_response.rfind('}') + 1
            curr_gpt_response = curr_gpt_response[:end_index]
            curr_gpt_response = json.loads(curr_gpt_response)["output"]

            if func_validate(curr_gpt_response, prompt=prompt):
                return func_clean_up(curr_gpt_response, prompt=prompt)

            if verbose:
                print("---- repeat count: \n", i, curr_gpt_response)
                print(curr_gpt_response)
                print("~~~~")

        except Exception:
            pass

    return False


def ChatGPT_safe_generate_response(prompt,
                                   example_output,
                                   special_instruction,
                                   repeat=3,
                                   fail_safe_response="error",
                                   func_validate=None,
                                   func_clean_up=None,
                                   verbose=False):
    prompt = '"""\n' + prompt + '\n"""\n'
    prompt += f"Output the response to the prompt above in json. {special_instruction}\n"
    prompt += "Example output json:\n"
    prompt += '{"output": "' + str(example_output) + '"}'

    if verbose:
        print("LLM PROMPT")
        print(prompt)

    for i in range(repeat):
        try:
            curr_gpt_response = ChatGPT_request(prompt).strip()
            end_index = curr_gpt_response.rfind('}') + 1
            curr_gpt_response = curr_gpt_response[:end_index]
            curr_gpt_response = json.loads(curr_gpt_response)["output"]

            if func_validate(curr_gpt_response, prompt=prompt):
                return func_clean_up(curr_gpt_response, prompt=prompt)

            if verbose:
                print("---- repeat count: \n", i, curr_gpt_response)
                print(curr_gpt_response)
                print("~~~~")

        except Exception:
            pass

    return False


def ChatGPT_safe_generate_response_OLD(prompt,
                                       repeat=3,
                                       fail_safe_response="error",
                                       func_validate=None,
                                       func_clean_up=None,
                                       verbose=False):
    if verbose:
        print("LLM PROMPT")
        print(prompt)

    for i in range(repeat):
        try:
            curr_gpt_response = ChatGPT_request(prompt).strip()
            if func_validate(curr_gpt_response, prompt=prompt):
                return func_clean_up(curr_gpt_response, prompt=prompt)
            if verbose:
                print(f"---- repeat count: {i}")
                print(curr_gpt_response)
                print("~~~~")

        except Exception:
            pass
    print("FAIL SAFE TRIGGERED")
    return fail_safe_response


# ─────────────────────────────────────────────────────────────
# Legacy GPT-3 completion API (still used by some prompt paths)
# ─────────────────────────────────────────────────────────────
def GPT_request(prompt, gpt_parameter):
    """
    Legacy completion-style request. The original used openai.Completion.create()
    which is deprecated. We route through chat instead.
    """
    temp_sleep()
    try:
        return call_llm(
            system_prompt="",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=gpt_parameter.get("max_tokens", 1024),
        )
    except Exception as e:
        print(f"TOKEN LIMIT EXCEEDED: {e}")
        return "TOKEN LIMIT EXCEEDED"


# ─────────────────────────────────────────────────────────────
# Prompt template loading — preserved from original
# ─────────────────────────────────────────────────────────────
def generate_prompt(curr_input, prompt_lib_file):
    """
    Takes in the current input and the path to a prompt file. The prompt file
    contains the raw str prompt with !<INPUT>! placeholders that get replaced.
    """
    if type(curr_input) == type("string"):
        curr_input = [curr_input]
    curr_input = [str(i) for i in curr_input]

    f = open(prompt_lib_file, "r")
    prompt = f.read()
    f.close()
    for count, i in enumerate(curr_input):
        prompt = prompt.replace(f"!<INPUT {count}>!", i)
    if "<commentblockmarker>###</commentblockmarker>" in prompt:
        prompt = prompt.split("<commentblockmarker>###</commentblockmarker>")[1]
    return prompt.strip()


def safe_generate_response(prompt,
                           gpt_parameter,
                           repeat=5,
                           fail_safe_response="error",
                           func_validate=None,
                           func_clean_up=None,
                           verbose=False):
    if verbose:
        print(prompt)

    for i in range(repeat):
        curr_gpt_response = GPT_request(prompt, gpt_parameter)
        if func_validate(curr_gpt_response, prompt=prompt):
            return func_clean_up(curr_gpt_response, prompt=prompt)
        if verbose:
            print("---- repeat count: ", i, curr_gpt_response)
            print(curr_gpt_response)
            print("~~~~")
    return fail_safe_response


# ─────────────────────────────────────────────────────────────
# Embeddings — multi-provider
# ─────────────────────────────────────────────────────────────
def get_embedding(text, model="text-embedding-ada-002"):
    """
    Get text embedding. Tries OpenAI embeddings first (works for OpenAI and
    Ollama providers). Falls back to a simple bag-of-words hash embedding
    for providers that don't have an embeddings endpoint.
    """
    text = text.replace("\n", " ")
    if not text:
        text = "this is blank"

    # Try OpenAI-compatible embeddings (works for openai and ollama providers)
    if PROVIDER in ("openai", "ollama"):
        try:
            import openai as _openai

            if PROVIDER == "ollama":
                base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
                client = _openai.OpenAI(
                    base_url=f"{base_url.rstrip('/')}/v1",
                    api_key="ollama",
                )
                # Ollama uses its own model names for embeddings
                emb_model = os.environ.get("EMBEDDING_MODEL", "nomic-embed-text")
            else:
                client = _openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
                emb_model = model

            response = client.embeddings.create(input=[text], model=emb_model)
            return response.data[0].embedding
        except Exception as e:
            print(f"Embedding API error ({PROVIDER}): {e}. Falling back to hash embedding.")

    # Fallback: simple deterministic hash-based embedding
    # This is NOT semantically meaningful but allows the system to function
    # without an embeddings API. For production, use a real embedding model.
    return _hash_embedding(text)


def _hash_embedding(text, dim=1536):
    """
    Generate a deterministic pseudo-embedding from text using hashing.
    NOT semantically meaningful — only a fallback so the system doesn't crash.
    For real semantic retrieval, use OpenAI/Ollama embeddings.
    """
    import hashlib
    import struct

    words = text.lower().split()
    embedding = [0.0] * dim

    for word in words:
        h = hashlib.sha256(word.encode()).digest()
        for i in range(0, min(len(h), dim * 4), 4):
            idx = (i // 4) % dim
            val = struct.unpack('f', h[i:i+4])[0]
            if not (val != val):  # check for NaN
                embedding[idx] += val

    # Normalize
    magnitude = sum(x * x for x in embedding) ** 0.5
    if magnitude > 0:
        embedding = [x / magnitude for x in embedding]

    return embedding


# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print(f"Provider: {PROVIDER}")
    print(f"Model: {MODEL}")
    print("Testing ChatGPT_request...")
    result = ChatGPT_request("Say 'hello world' in exactly two words.")
    print(f"Response: {result}")
