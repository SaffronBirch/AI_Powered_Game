

import gradio as gr
from dataclasses import dataclass, field, replace
import statistics
import math
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
import json
import itertools
import os
#from generate_npc_admin_session import build_trait_phrase, build_biography_description, build_item_preambles, generate_admin_session
from helper import save_world, load_world
from pathlib import Path
import re
 

'''
Enter a character name. If the character already exists in personas/ load the corresponding JSON file. If the character doesn't exist,
load the template JSON file.
'''

base_path = Path(__file__).parent

def load_character():
    folder_path = base_path / "personas"

    charName = input("Enter your characters name: ").lower()

    match = next(
        (x for x in os.listdir(folder_path) if x.lower() == charName),
        None
    )

    if match:
        file_path = folder_path / match / "bootstrap_memory" / "scratch.json"
    else:
        file_path = folder_path / "template" / "scratch.json"

    return load_world(file_path)

character_data = load_character()

#######################################################################################################################################################################
#######################################################################################################################################################################
#######################################################################################################################################################################

# Linguistic qualifiers for each of the 9 levels
# Level 1 = extremely low, Level 5 = neutral, Level 9 = extremely high
level_qualifiers = {
    1: ("extremely {low}"),
    2: ("very {low}"),
    3: ("{low}"),
    4: ("a bit {low}"),
    5: ("neither {low} nor {high}"),
    6: ("a bit {high}"),
    7: ("{high}"),
    8: ("very {high}"),
    9: ("extremely {high}"),
}
# Item instruction variants (ev0-ev4)
item_instructions = {
    1: ("Considering the statement"),
    2: ("Thinking about the statement"),
    3: ("Reflecting on the statement"),
    4: ("Evaluating the statement"),
    5: ("Regarding the statement"),
}



def build_trait_phrase(persona: dict) -> str:
    trait_phrase = []

    traits = persona.get("innate")
    trait_phrase.append(f"I am {traits}.")
    
    return (". ".join(trait_phrase) + ".")



def build_biography_description(persona: dict) -> str:
    parts = []

    bio = persona.get("first_person_bio")
    parts.append(bio)

    return parts



def build_item_preambles(persona:dict, traits:str, biography:str) -> dict:
    # Preamble ID is the key, rest is the value
    item_preambles = {}
    
    ids = []


    preamble_id = persona.get("trait_score")

    beginning = "For the following task, respond in a way that matches this description: "

    for id, instruction in item_instructions.items():
        full_preamble_id = f"{preamble_id}-ev{id}"
        preamble = f"{beginning}{traits} {biography} {instruction}, "
        item_preambles[full_preamble_id] = preamble

    return item_preambles


def generate_admin_session(persona, item_preambles):
    template_file = "smart_npc_shaping_session_template.json"
    base_path = Path(__file__).parent
    template_path = base_path/ "testing_files" / template_file

    output_file = f"{persona['name']}_shaping_admin_session.json"
    output_path = base_path / "personas" / f"{persona['name']}" / output_file

    with open(template_path, 'r') as f:
        session = json.load(f)
    
    preamble_ids = list(item_preambles.keys())

    for scale_id in session["measures"]["IPIP300"]["scales"]:
        session["measures"]["IPIP300"]["scales"][scale_id]["item_preamble_ids"] = preamble_ids
    
    session["item_preambles"] = item_preambles

    with open(output_path, "w") as f:
        json.dump(session, f)


def generate_session(persona):
            
    trait_phrase = build_trait_phrase(persona)
    bio = build_biography_description(persona)
    item_preambles = build_item_preambles(persona, trait_phrase, bio)
    generate_admin_session(persona, item_preambles)
    return "✅ Admin session saved successfully."


generate_session(character_data)






