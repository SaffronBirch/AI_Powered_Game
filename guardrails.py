from typing import Any, Dict
#from helper import RAGChatWidget, SimpleVectorDB
from helper import load_world, save_world, load_env, get_ollama_api_key
from LLM import API_helper, _content_to_str

# Guardrails imports
from guardrails import Guard, OnFailAction, settings
from guardrails.validator_base import (
    FailResult,
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
)