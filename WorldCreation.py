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
- Only generate in plain text without formatting.
- Use simple clear language without being flowery.
- You must stay below 3-5 sentences for each description.
"""
#Create the World Prompt
world_prompt = """
Generate a description of the virtual world that is explored in "The Witcher 3: Wild Hunt". 

Output content in the form:
World Name: The Continent
World Description: <WORLD DESCRIPTION>
"""

#Create the Region Prompt
region_prompt = """
Generate a description for each of the six major regions in The Witcher 3: Wild Hunt. \
For each hold's description, detail all of the different aspects of the region and all \
of the people who reside in that region.

Output content in the form:
Region 1 Name: White Orchard
Region 1 Description: <HOLD DESCRIPTION>
Region 2 Name: Royal Palace in Vizima
Region 2 Description: <HOLD DESCRIPTION>
Region 3 Name: Velen
Region 3 Description: <HOLD DESCRIPTION>
Region 4 Name: Novigrad
Region 4 Description: <HOLD DESCRIPTION>
Region 5 Name: The Skellige Isles
Region 5 Description: <HOLD DESCRIPTION>
Region 6 Name: Kaer Morhen
Region 6 Description: <HOLD DESCRIPTION>
    """

#Create Character of Geralt
# Geralt starts in White Orchard by default. Depending on which region Geralt starts the chat in, 
# that determines how much information he should know at a given time.
def get_geralt(region):
  return f"""
  Imagine you are portraying the character Geralt of Rivia. Create a description of Geralt \
  based on the region he is in. Describe Geralt's appearance, profession, and purpose for being in \
  that particular region. 

  Output content in the form:
  Character Name: Geralt of Rivia
  Character Description: <CHARACTER DESCRIPTION>

  Region Name: {region['name']}
  Region Description: {region['description']}
  """

############################################## Content Generation ##############################################

from LLM import API_helper

# World Generation
world_messages =[ 
  {"role": "system", "content": system_prompt},
  {"role": "user", "content": world_prompt}
]
world_output = API_helper(world_messages)

world_output = world_output.strip()
world = {
  "name": world_output.split('\n')[0].strip()
  .replace('World Name: ', ''),
  "description": '\n'.join(world_output.split('\n')[1:])
  .replace('World Description:', '').strip()
}
print(world_output)

#Region Generation
region_messages = [
  {"role": "system", "content": system_prompt},
  {"role": "user", "content": region_prompt}
]
regions_output = API_helper(region_messages)
regions = {}

for output in regions_output.split('\n\n'):
  region_name = output.strip().split('\n')[0] \
    .split('Name: ')[1].strip()
  print(f'Created hold "{region_name}" in {world["name"]}')
  
  region_description = output.strip().split('\n')[1] \
    .split('Description: ')[1].strip()
  
  region = {
      "name": region_name,
      "description": region_description,
      "world": world['name']
  }
  regions[region_name] = region

world['regions'] = regions

print(f'\nRegion 1 Description: \
{regions["White Orchard"]["description"]}')


# Main Character Generation
def create_main_character(region):
  print(f'\nCreating Geralt for the region of: {region["name"]}...')
  
  character_messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": get_geralt(region)}
  ]
  character_output = API_helper(character_messages)
  main_character = {}

  for output in character_output.split('\n\n'):
    char_name = output.strip().split('\n')[0]\
    .split('Name: ')[1].strip()
    print(f'- "{char_name}" created')
    
    char_description = output.strip().split('\n')[1\
    ].split('Description: ')[1].strip()
    
    character = {
    "name": char_name,
    "description": char_description,
    "region" : region['name']
    }
    main_character[char_name] = character
  
  region['MainCharacter'] = main_character


for region in regions.values():    # Create Geralt's profile for White Orchard
  create_main_character(region)
  break

character = list(region['MainCharacter'].values())[0]

print(f'\nGeralt in {region["name"]}:\n{character["description"]}')

############################################## Save to JSON ##############################################

from helper import save_world

save_world(world, '/mnt/c/Users/Saffron/Documents/Ontario Tech Class Notes/Thesis/AI_Powered_Game/TheContinent.json')

