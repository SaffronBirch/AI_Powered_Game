from LLM import API_helper, get_token_budget
from helper import save_world
from pathlib import Path
import json, re

'''
STEPS:
1. 

'''

############################################## Read Game Script File #####################################

# Currently, the script is read only up to the models max character count, and then the script is cut off.
# This should be updates to instead split the text into chunks and use RAG to feed the model the most relevant parts of the script.
def load_script_data(filename):

    base_path = Path(__file__).parent
    file_path = base_path/ "scriptData" / filename

    model = "gpt-oss:120b-cloud"

    char_count = 4
    max_chars = get_token_budget(model) * char_count

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    elif file_path.suffix.lower() == ".txt":
        with open(file_path, 'r', encoding='utf-8') as file:
            return  file.read(max_chars)


################################################### System Prompt #########################################

#Create the System Prompt
def create_system_prompt():
    return f"""
    Your job is to help create an immersive fantasy world based on a video game.

    Instructions:
    - Return ONLY valid JSON (no markdown, no extra text).
    - Only generate plain JSON.
    - Do not include explanations or commentary.
    - Use simple clear language.
    - You must stay below 3-5 sentences for each description.
    """

################################################### Prompts for Script Generation #########################################

#Create the World Prompt
def create_world_from_script(game_data):
    return f"""
    - Generate a description of the world that is being created based on the content in {game_data}. 

    Return ONLY valid JSON.

    Schema:
    {{
    "name": "World Name", 
    "description": "World Description"
    }}
    """

#Create the Region Prompt
def create_regions_from_script(world):
    return f"""
    - Generate a description for all of the major regions in {world}.
    - For each regions's description, detail its location in the world, main features and residents.

    Return ONLY valid JSON.

    Schema:
    {{
    "regions": 
        {{
        "Region Name": 
            {{
            "description": "Region Description"
            }}
        }}
    }}
    """

#Create Characters
def create_characters_from_script(world, region):
    return f""" 
    - Generate a description for all the characters that exist in {region} in the world of {world}. 
    - Describe their appearance, profession, and purpose for being in that particualr region.
    - Describe their backstory.
    - Describe the chartacters 5 most prevelant personality traits.

    Return ONLY valid JSON.
    Schema:
    {{
    "characters": 
        {{
        "Character Name":
            {{
            "description": "Character Description", 
            "backstory": "Character Backstory",
            "personality": [trait1, trait2, trait3, trait4, trait5]
            }}
        }}
    
    }}
  """

################################################### Prompts for User Input Generation #########################################

#Create the World Prompt
def create_world_from_input():
    return f"""
    - Generate a name and description for a fictional fantasy world.

    Return ONLY valid JSON.

    Schema:
    {{
    "name": "World Name", 
    "description": "World Description"
    }}
    """

#Create the Region Prompt
def create_regions_from_input(world):
    return f"""
    - Generate a name and description for a region in {world}.
    - For the regions's description, detail its location in the world, main features and residents.

    Return ONLY valid JSON.

    Schema:
    {{
    "regions": 
        {{
        "Region Name": 
            {{
            "description": "Region Description"
            }}
        }}
    }}
    """

#Create Character
def create_character_from_input(world, region):
    return f""" 
    - Generate a description for a characters that exist in {region} in the world of {world}. 
    - Describe their appearance, profession, and purpose for being in that particualr region.
    - Describe their backstory.
    - Describe the chartacters 5 most prevelant personality traits.

    Return ONLY valid JSON.

    Schema:
    {{
    "characters": 
        {{
        "Character Name":
            {{
            "description": "Character Description", 
            "backstory": "Character Backstory",
            "personality": [trait1, trait2, trait3, trait4, trait5]
            }}
        }}
    
    }}
  """

############################################## Content Generation ##############################################

from abc import ABC, abstractmethod

# Abstract class 
class Generator(ABC):
    def __init__(self):
        self.world = {}
    
    @abstractmethod
    def generate_world(self):
        pass

    @abstractmethod
    def generate_regions(self):
        pass

    @abstractmethod
    def generate_characters(self, region_name):
        pass

    def save_to_file(self, filename):
        if self.world is None:
            print("No world has been generated.")

        base_path = Path(__file__).parent
        save_path = base_path/filename

        save_world(self.world, save_path)


# Generate JSON file from user input data
class inputDataGenerator(Generator):
    def __init__(self):
        super().__init__()

     # Collect world information from user   
    def generate_world(self):
        print('''
            Let's start with where your character lives. Do you want to describe the world yourself,
            or have a world generated for you? \n
            Create world from scratch: Press 1 \n
            Generate world: Press 2 \n
            ''')
        
        choice = input("Enter choice: ").strip()

        if choice == '1':
            self.world['name'] = input("World Name: ").strip()
            self.world['description'] = input("World Description: ").strip()
            self.world['regions'] = {}
            return self.world
       
        elif choice == '2':
            print(f"generating world...")
       
            world_messages = [
                {"role": "system", "content": create_system_prompt()},
                {"role": "user", "content": create_world_from_input()}]
                
            world_output = API_helper(world_messages)
            world_data = json.loads(world_output)

            self.world = {
                "name": world_data["name"].strip(),
                "description": world_data["description"].strip(),
                "regions": {}
            }

            print(f"Created world: {self.world['name']}")
            return self.world


    # Collect region information from user
    def generate_regions(self):
        print('''
            Now let's describe any regions that populate your world. Do you want to describe the regions yourself,
            or have them generated for you? \n
            Create region from scratch: Press 1 \n
            Generate region: Press 2 \n
            ''')
        
        choice = input("Enter choice: ").strip()
        
        if choice == '1':
            add_region = True
            while add_region == True:
                region_name = input("Region Name: ").strip()

                self.world['regions'][region_name] = {
                    "description": input("Region Description: ").strip()
                }
                another = ("Add another region? (y/n)")

                if another == 'y':
                    add_region = True
                elif another == 'n': 
                    add_region = False
                    break
                
            return self.world['regions']

        elif choice == '2':
            add_region = True
            print(f"Generating regions for {self.world['name']}...")

            while add_region == True:
                region_messages = [
                    {"role": "system", "content": create_system_prompt()},
                    {"role": "user", "content": create_regions_from_input(self.world)}
                ]
                region_output = API_helper(region_messages)
                region_data = json.loads(region_output)
                                    
                for region_name, region_info in region_data['regions'].items():
                    self.world['regions'][region_name] = {
                        "description": region_info['description'].strip(),
                        "characters": {}
                    }
                    print(f"created region {region_name}")

                    another = input("Add another region? (y/n) \n").strip()

                    if another == 'y':
                        add_region = True
                    elif another == 'n': 
                        add_region = False
                        break
                
            return self.world['regions']


    # Collect character information from user
    def generate_characters(self, region_name=None):
        print("Finally let's create your NPC profile.  \n")

        print('''
            Do you want to create your character from scratch or have one generated for you? \n
            Create character from scratch: Press 1 \n
            Generate character: Press 2 \n
            ''')
        
        choice = input("Enter choice: ").strip()
        
        if choice == '1':
            add_character = True

            while add_character == True:
                region_name = input(f"Which region does your character live in? \n Available regions: {', '.join(self.world['regions'].keys())} \n")

                character_name = input("Character Name: ").strip()
                character_description = input("Character Description: ").strip()
                character_backstory = input("Character Backstory: ").strip()

                print("Enter 5 personality traits for your character: ")
                character_personality = []
                for i in range(5):
                    trait = input(f"Trait {i+1}: ").strip()
                    character_personality.append(trait)

                region = self.world['regions'][region_name]

                region['characters'][character_name] = {
                    "description": character_description,
                    "backstory": character_backstory,
                    "personality": character_personality
                }
                another = input("Add another character? (y/n) \n").strip()

                if another == 'y':
                    add_character = True
                elif another == 'n': 
                    add_character = False
                    break

            return region['characters']

        elif choice == '2':
            add_character = True

            while add_character == True:
                region_name = input(f"Which region does your character live in? \n Available regions: {', '.join(self.world['regions'].keys())} \n")

                print(f"Generating character for {region_name}...")

                region = self.world['regions'][region_name]

                character_messages = [
                    {"role": "system", "content": create_system_prompt()},
                    {"role": "user", "content": create_character_from_input(self.world, region)}
                ]

                character_output = API_helper(character_messages)
                character_data = json.loads(character_output)

                for character_name, character_info in character_data['characters'].items():     
                    region["characters"][character_name] = {
                        "description": character_info["description"].strip(),
                        "backstory": character_info['backstory'].strip(),
                        "personality": character_info['personality']
                    }
                    print(f"created character {character_name}")

                    another = input("Add another character? (y/n) \n").strip()

                    if another == 'y':
                        add_character = True
                    elif another == 'n': 
                        add_character = False
                        break

            return region['characters']


# Generate JSON file from script data
class ScriptDataGenerator(Generator):
    def __init__(self, data_filename=None):
       super().__init__()
       self.game_data = load_script_data(data_filename)
   
    def generate_world(self):
       print(f"generating world...")
       
       world_messages = [
           {"role": "system", "content": create_system_prompt()},
           {"role": "user", "content": create_world_from_script(self.game_data)}]
           
       world_output = API_helper(world_messages)
       world_data = json.loads(world_output)

       self.world = {
          "name": world_data["name"].strip(),
          "description": world_data["description"].strip(),
          "regions": {}
       }

       print(f"Created world: {self.world['name']}")
       return self.world
    
   
    def generate_regions(self):
        if not self.world:
            raise ValueError("World must be generated first. Call generate_world() first.")
        
        print(f"Generating regions...")

        region_messages = [
            {"role": "system", "content": create_system_prompt()},
            {"role": "user", "content": create_regions_from_script(self.world['name'])}
        ]
        regions_output = API_helper(region_messages)
        region_data = json.loads(regions_output)
        
        for region_name, region_info in region_data['regions'].items():
            self.world['regions'][region_name] = {
                "description": region_info['description'].strip(),
                "characters": {}
            }
            print(f"created region {region_name}")
        
        return self.world['regions']
    
  
    def generate_characters(self, region_name):
        if region_name not in self.world['regions']:
            raise ValueError("Regions must be generated first. Call generate_regions() first.")
        
        print(f"Generating characters for {region_name}...")

        region = self.world['regions'][region_name]

        character_messages = [
            {"role": "system", "content": create_system_prompt()},
            {"role": "user", "content": create_characters_from_script(self.world['name'], region)}
        ]

        character_output = API_helper(character_messages)
        character_data = json.loads(character_output)

        for character_name, character_info in character_data['characters'].items():     
            region["characters"][character_name] = {
                "description": character_info["description"].strip(),
                "backstory": character_info['backstory'].strip(),
                "personality": character_info['personality']
            }
            print(f"created character {character_name}")

        return region['characters']

############################################## User Input and World Generation ##############################################

print(f'''
      
    \n === Welcome to the Smart NPC Generator === \n
      
      If you are creating a character from scratch: Press 1 \n
      If you are creating a character from an exising script: Press 2 \n
    '''
    )

world = None
generator = None

choice = input("Enter choice: ").strip()

# Build NPC from scratch
if choice == '1':
    generator = inputDataGenerator()
    world = generator.generate_world()
    generator.generate_regions()
    generator.generate_characters()

# Build NPC from existing script
elif choice == '2':
    script_filename = input("Please enter the filename of your script (Ex. SkyrimScript.txt): ")
    generator = ScriptDataGenerator(script_filename)

    world = generator.generate_world()
    regions = generator.generate_regions()

    for region_name in regions:
        generator.generate_characters(region_name)

else:
    print("invalid choice, please enter 1 or 2")

if generator and world:
    filename = f"{world['name'].replace(' ', '')}.json"
    generator.save_to_file(filename)
else:
    print("No world has been generated, nothing to save")






        














