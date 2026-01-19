import os
import json
from dotenv import load_dotenv, find_dotenv
                                                                                                                                    
def load_env():
    _ = load_dotenv(find_dotenv())

def save_world(world, filename):
    with open(filename, 'w') as f:
        json.dump(world, f)

def load_world(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def get_ollama_api_key():
     load_env()
     ollama_api_key = os.getenv("Enter API Key Here")
     return ollama_api_key