import json
import itertools
import os
from pathlib import Path
 
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

def build_trait_phrase(adj_rating_pairs) -> str:
    trait_phrase = []

    for trait, (low, high), rating in adj_rating_pairs:
        if 1<= rating <= 9:
            trait_phrase.append("I am " + level_qualifiers[rating].format(low=low, high=high))
        else:
            "Not a valid rating. Please choose from 1-9."
    
    return (". ".join(trait_phrase) + ".")


def build_biography_description(profile) -> str:
    parts = []

    if profile.name and profile.pronouns:
        parts.append(f"My name is {profile.name} ({profile.pronouns}).")
    elif profile.name:
        parts.append(f"My name is {profile.name}.")
    if profile.age:
        parts.append(f"I am {profile.age} years old.")
    if profile.role and profile.race:
        parts.append(f"I am a {profile.race} {profile.role}.")
    elif profile.role:
        parts.append(f"I am a {profile.role}.")
    elif profile.race:
        parts.append(f"I am a {profile.race}.")
    if profile.appearance:
        parts.append(profile.appearance)
    if profile.backstory:
        parts.append(profile.backstory)
    if profile.relationships:
        parts.append(profile.relationships)
    if profile.skills:
        parts.append(f"{profile.skills}.")
    if profile.opinions:
        parts.append(profile.opinions)
    if profile.loves:
        parts.append(f"{profile.loves}.")
    if profile.hates:
        parts.append(f"{profile.hates}.")
    if profile.hobbies:
        parts.append(profile.hobbies)
    if profile.goals:
        parts.append(profile.goals)
    if profile.flaws:
        parts.append(profile.flaws)

    return " ".join(parts)


def build_item_preambles(trait_phrase, biography, median_ratings) -> dict:
    # Preamble ID is the key, rest is the value
    item_preambles = {}
    
    ids = []

    for trait, rating in median_ratings.items():
        ids.append(f"{trait}{rating}")

    preamble_id = "-".join(ids)

    beginning = "For the following task, respond in a way that matches this description: "
    for id, instruction in item_instructions.items():
        full_preamble_id = f"{preamble_id}-ev{id}"
        preamble = f"{beginning}{trait_phrase} {biography.strip()} {instruction}, "
        item_preambles[full_preamble_id] = preamble
    
    return item_preambles


def generate_admin_session(item_preambles):
    filename = "smart_npc_shaping_session_template.json"
    base_path = Path(__file__).parent
    file_path = base_path/ "testing_files" / filename

    with open(file_path, 'r') as f:
        session = json.load(f)
    
    preamble_ids = list(item_preambles.keys())

    for scale_id in session["measures"]["IPIP300"]["scales"]:
        session["measures"]["IPIP300"]["scales"][scale_id]["item_preamble_ids"] = preamble_ids
    
    session["item_preambles"] = item_preambles

    with open("my_npc_shaping_admin_session.json", "w") as f:
        json.dump(session, f)



# ratings = [["kind", "unkind", 1],
#            ["intelligent", "unintelligent", 5], 
#            ["happy", "unhappy", 9],
# ]

# phrase = build_trait_phrase(ratings)

# bio = """
# My name is Aldric (he/him) I am 52 years old. I am a human village guardian. \
# I have a weathered face and a long grey beard. I walk with a slight limp from an old battle wound. \
# I have defended my village for thirty year. I lost my family to a great war. I have lived alone in the mountains ever since. \
# I have one trusted friend, the village elder. I distrust strangers easily. I am an expert swordsman, I know the mountain passes better than anyone. \
# I believe honour is more important than victory. I think most wars are started by cowards. 
# """

# medians = [
#     ["ext", 1],
#     ["agr", 1],
#     ["con", 1],
#     ["neu", 1],
#     ["ope", 1],
# ]

# print(build_item_preambles(phrase, bio, medians))




