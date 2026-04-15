"""
LLM Helper — Multi-Provider
════════════════════════════
Supports Anthropic (Claude), OpenAI (GPT), Google Gemini, and Ollama
through a single unified interface.

PROVIDER SELECTION (checked in order):
  1. LLM_PROVIDER env var          →  "anthropic" | "openai" | "gemini" | "ollama"
  2. Auto-detect from API keys     →  whichever key is set first
  3. Ollama fallback               →  if nothing else is configured

CONFIGURATION (env vars):
  ┌─────────────────────┬──────────────────────────────────────────┐
  │ Provider            │ Required env vars                        │
  ├─────────────────────┼──────────────────────────────────────────┤
  │ anthropic           │ ANTHROPIC_API_KEY                        │
  │ openai              │ OPENAI_API_KEY                           │
  │ gemini              │ GEMINI_API_KEY  (or GOOGLE_API_KEY)      │
  │ ollama              │ OLLAMA_BASE_URL (default: localhost:11434)│
  └─────────────────────┴──────────────────────────────────────────┘

  LLM_MODEL     →  Override the default model for any provider
  LLM_PROVIDER  →  Force a specific provider

EXAMPLES:
  # Anthropic (default)
  ANTHROPIC_API_KEY=sk-ant-... streamlit run app.py

  # OpenAI
  LLM_PROVIDER=openai OPENAI_API_KEY=sk-... streamlit run app.py

  # Gemini
  LLM_PROVIDER=gemini GEMINI_API_KEY=AIza... streamlit run app.py

  # Ollama (local, no key needed)
  LLM_PROVIDER=ollama LLM_MODEL=llama3 streamlit run app.py

  # Ollama on a remote machine
  LLM_PROVIDER=ollama OLLAMA_BASE_URL=http://192.168.1.10:11434 LLM_MODEL=mistral streamlit run app.py
"""

import os
import json
import logging
from typing import Optional

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# DEFAULT MODELS PER PROVIDER
# ─────────────────────────────────────────────────────────────
DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-4-20250514",
    "openai":    "gpt-4o-mini",
    "gemini":    "gemini-2.0-flash",
    "ollama":    "llama3",
}

# ─────────────────────────────────────────────────────────────
# PROVIDER DETECTION
# ─────────────────────────────────────────────────────────────
def _detect_provider() -> str:
    """Detect which LLM provider to use based on env vars."""
    explicit = os.environ.get("LLM_PROVIDER", "").lower().strip()
    if explicit in DEFAULT_MODELS:
        return explicit

    # Auto-detect from API keys
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        return "gemini"

    # Default to ollama (no key needed)
    return "ollama"


PROVIDER = _detect_provider()
MODEL = os.environ.get("LLM_MODEL", DEFAULT_MODELS.get(PROVIDER, ""))

log.info(f"LLM provider: {PROVIDER}, model: {MODEL}")


# ─────────────────────────────────────────────────────────────
# PROVIDER: ANTHROPIC
# ─────────────────────────────────────────────────────────────
_anthropic_client = None

def _call_anthropic(system_prompt: str, messages: list[dict], max_tokens: int = 1024) -> str:
    global _anthropic_client
    import anthropic

    if _anthropic_client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set.")
        _anthropic_client = anthropic.Anthropic(api_key=api_key)

    filtered = [m for m in messages if m["role"] in ("user", "assistant")]

    response = _anthropic_client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=filtered,
    )
    return response.content[0].text


# ─────────────────────────────────────────────────────────────
# PROVIDER: OPENAI
# ─────────────────────────────────────────────────────────────
_openai_client = None

def _call_openai(system_prompt: str, messages: list[dict], max_tokens: int = 1024) -> str:
    global _openai_client
    import openai

    if _openai_client is None:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set.")
        _openai_client = openai.OpenAI(api_key=api_key)

    # OpenAI uses system role in the messages array
    api_messages = [{"role": "system", "content": system_prompt}]
    for m in messages:
        if m["role"] in ("user", "assistant"):
            api_messages.append({"role": m["role"], "content": m["content"]})

    response = _openai_client.chat.completions.create(
        model=MODEL,
        max_tokens=max_tokens,
        messages=api_messages,
    )
    return response.choices[0].message.content


# ─────────────────────────────────────────────────────────────
# PROVIDER: GEMINI
# ─────────────────────────────────────────────────────────────
_gemini_client = None

def _call_gemini(system_prompt: str, messages: list[dict], max_tokens: int = 1024) -> str:
    global _gemini_client
    from google import genai
    from google.genai import types

    if _gemini_client is None:
        api_key = os.environ.get("GEMINI_API_KEY", os.environ.get("GOOGLE_API_KEY", ""))
        if not api_key:
            raise ValueError("GEMINI_API_KEY (or GOOGLE_API_KEY) not set.")
        _gemini_client = genai.Client(api_key=api_key)

    # Build contents from messages
    contents = []
    for m in messages:
        if m["role"] == "user":
            contents.append(types.Content(role="user", parts=[types.Part(text=m["content"])]))
        elif m["role"] == "assistant":
            contents.append(types.Content(role="model", parts=[types.Part(text=m["content"])]))

    response = _gemini_client.models.generate_content(
        model=MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            max_output_tokens=max_tokens,
        ),
    )
    return response.text


# ─────────────────────────────────────────────────────────────
# PROVIDER: OLLAMA (via OpenAI-compatible API)
# ─────────────────────────────────────────────────────────────
_ollama_client = None

def _call_ollama(system_prompt: str, messages: list[dict], max_tokens: int = 1024) -> str:
    global _ollama_client
    import openai

    if _ollama_client is None:
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        # Ollama exposes an OpenAI-compatible endpoint at /v1
        _ollama_client = openai.OpenAI(
            base_url=f"{base_url.rstrip('/')}/v1",
            api_key="ollama",  # Ollama doesn't need a real key
        )

    api_messages = [{"role": "system", "content": system_prompt}]
    for m in messages:
        if m["role"] in ("user", "assistant"):
            api_messages.append({"role": m["role"], "content": m["content"]})

    response = _ollama_client.chat.completions.create(
        model=MODEL,
        messages=api_messages,
    )
    return response.choices[0].message.content


# ─────────────────────────────────────────────────────────────
# DISPATCH TABLE
# ─────────────────────────────────────────────────────────────
_PROVIDERS = {
    "anthropic": _call_anthropic,
    "openai":    _call_openai,
    "gemini":    _call_gemini,
    "ollama":    _call_ollama,
}


# ─────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────
def call_llm(system_prompt: str, messages: list[dict], max_tokens: int = 1024) -> str:
    """
    Send a chat completion request and return the text response.

    Uses whichever provider was detected at startup. The interface is
    identical regardless of provider — just set the right env vars.

    Args:
        system_prompt: The system-level instructions.
        messages:      List of {"role": "user"|"assistant", "content": "..."}.
        max_tokens:    Maximum tokens in the response.

    Returns:
        The assistant's text response.
    """
    provider_fn = _PROVIDERS.get(PROVIDER)
    if provider_fn is None:
        return _fallback_response()

    try:
        return provider_fn(system_prompt, messages, max_tokens)
    except ImportError as e:
        return (
            f"*Missing package for {PROVIDER}. "
            f"Install it with: pip install {_install_hint()}* (Error: {e})"
        )
    except Exception as e:
        log.error(f"LLM error ({PROVIDER}): {e}")
        return f"*The winds of fate are silent...* (Error: {e})"


def generate_json(system_prompt: str, user_prompt: str) -> Optional[dict]:
    """
    Send a prompt expecting a JSON response and parse it.

    Args:
        system_prompt: System instructions (should mention JSON-only output).
        user_prompt:   The user's generation request.

    Returns:
        Parsed dict or list, or None on failure.
    """
    text = call_llm(system_prompt, [{"role": "user", "content": user_prompt}])

    if text.startswith("*") and "Error" in text:
        log.error(f"LLM returned error: {text}")
        return _fallback_json()

    try:
        return _parse_json_response(text)
    except json.JSONDecodeError as e:
        log.error(f"JSON parse error: {e}\nRaw: {text[:300]}")
        return None


def get_provider_info() -> dict:
    """Return current provider configuration for display in the UI."""
    return {
        "provider": PROVIDER,
        "model": MODEL,
        "status": "configured",
    }


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def _parse_json_response(text: str):
    """Strip markdown fences and parse JSON from an LLM response."""
    cleaned = text.strip()

    # Remove ```json ... ``` wrapping
    if cleaned.startswith("```"):
        # Find the end of the first line (might be ```json or just ```)
        first_nl = cleaned.find("\n")
        if first_nl != -1:
            cleaned = cleaned[first_nl + 1:]
        else:
            cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    # Try parsing as-is
    return json.loads(cleaned)


def _install_hint() -> str:
    """Return the pip install command for the current provider."""
    hints = {
        "anthropic": "anthropic",
        "openai":    "openai",
        "gemini":    "google-genai",
        "ollama":    "openai",  # Ollama uses the OpenAI client
    }
    return hints.get(PROVIDER, PROVIDER)


def _fallback_response() -> str:
    return (
        "*Greetings, stranger. No LLM provider is configured. "
        "Set one of: ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY, "
        "or LLM_PROVIDER=ollama.*"
    )


def _fallback_json() -> dict:
    return {
        "game_name": "The Forgotten Realms",
        "world_name": "Aethermoor",
        "description": "A mist-shrouded continent where ancient magic lingers in crumbling ruins.",
        "name": "Aethermoor",
    }
