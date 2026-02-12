import json

#####################################################################################


world = {}

world['name'] = None
world['description'] = None

world['regions'] = []

world['regions'].append({})

print(f'''
    Great! Let's get started \n
          
          Step 1) Let's start with where your character lives: \n
          World name: \n
          World description: \n

          Step 2) Describe any regions that populate your world:
          Region name: \n
          Region Description: \n

          Step 3) Create a basic NPC profile: \n
          Character name: \n
          Character Description: \n
          Character Backstory: \n

          Step 4) Give your character more details about their place in the world:
          Which region does your character live in? \n
          Personaity traits (max 5): \n
          Relationships: \n
      '''
      )
    