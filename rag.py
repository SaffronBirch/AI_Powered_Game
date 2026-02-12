import numpy as np
from ollama import Client
from LLM import estimate_tokens, get_token_budget

client = Client(host="http://localhost:11434")
embed_model = "nomic-embed-text"
model = "gpt-oss:120b-cloud"

#######################################################################