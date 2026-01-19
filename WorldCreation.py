############################################## Description ##############################################

# #Hierarchial Content Generation

#         level1
#         //  \\         
#        //    \\
#       //      \\
#     level2    level2
#    // \\     //   \\      
#   //   \\   //     \\ 
#  //     \\ //       \\      
# level3  level3   level3

#Include previously generated content to make AI model more holistic.

############################################## Prompts #####################################

#Create the System Prompt
system_prompt = f"""
Your job is to help create an immersive fantasy world based on the video game, "The Witcher 3: Wild Hunt". \

Instructions:
- Return ONLY valid JSON (no markdown, no extra text).
- Only generate plain JSON.
- Do not include explanations or commentary.
- Use simple clear language.
- You must stay below 3-5 sentences for each description.
"""
#Create the World Prompt
world_prompt = """
Generate a description of the world that is explored in "The Witcher 3: Wild Hunt". 

Return ONLY valid JSON.

Schema:
{
  "name": "The Continent", 
  "description": "..."
}

"""

#Create the Region Prompt
region_prompt = """
Generate a description for each of the major regions in The Witcher 3: Wild Hunt. \
For each regions's description, detail all of the different aspects of the region and all \
of the people who reside there.

Return ONLY valid JSON.

Schema:
{
  "regions": [
    {"name": "White Orchard", "description": "..."},
    {"name": "Royal Palace in Vizima", "description": "..."},
    {"name": "Velen", "description": "..."},
    {"name": "Novigrad", "description": "..."},
    {"name": "The Skellige Isles", "description": "..."},
    {"name": "Kaer Morhen", "description": "..."}
  ]
}
    """

#Create Character of Geralt
# Geralt starts in White Orchard by default. Depending on which region Geralt starts the chat in, 
# that determines how much information he should know at a given time.
def get_geralt(region):
  return f"""
  Imagine you are portraying the character Geralt of Rivia. Create a description of Geralt \
  based on the region he is in. Describe Geralt's appearance, profession, and purpose for being in \
  that particular region. 

  Return ONLY valid JSON.

  Schema:
  {{
    "characters": [
      {{
        "name":"Geralt of Rivia", 
        "description": "...", "region": 
        "{region['name']}"}}
    ]
  }}
  """

############################################## Content Generation ##############################################

from LLM import API_helper
import json, re

# World Generation
world_messages =[ 
  {"role": "system", "content": system_prompt},
  {"role": "user", "content": world_prompt}
]
world_output = API_helper(world_messages)

world_data = json.loads(world_output)

world_name = world_data["name"].strip()
world_description = world_data["description"].strip()

world = {
  "name": world_name,
  "description": world_description
}

print(world)

#Region Generation
region_messages = [
  {"role": "system", "content": system_prompt},
  {"role": "user", "content": region_prompt}
]
regions_output = API_helper(region_messages)

region_data = json.loads(regions_output)
regions = {}

for r in region_data["regions"]:
  region_name = r["name"].strip()
  region_description = r["description"].strip()

  region = {
      "name": region_name,
      "description": region_description,
      "world": world['name']
  }
  regions[region_name] = region

  print(f'Created region "{region_name}" in {world["name"]}')

world['regions'] = regions

# Main Character Generation
def create_main_character(region):
  character_messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": get_geralt(region)}
  ]
  character_output = API_helper(character_messages)

  character_data = json.loads(character_output)
  characters = {}

  for c in character_data["characters"]:
    character_name = c["name"].strip()
    character_description = c["description"].strip()

  character = {
    "name": character_name,
    "description": character_description,
    "region": region["name"]
  }

  characters[character_name] = character

  region['characters'] = characters

  print(f'Created "{character_name}" for the region of: {region["name"]}...')


for region in regions.values():    # Create Geralt's profile for White Orchard
  create_main_character(region)

############################################## Save to JSON ##############################################

from helper import save_world

save_world(world, '/mnt/c/Users/Saffron/Documents/Ontario Tech Class Notes/Thesis/AI_Powered_Game/TheContinent.json')

