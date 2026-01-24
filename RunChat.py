import gradio as gr
from LLM import API_helper, _content_to_str
from helper import load_world, save_world, load_env, get_ollama_api_key
from datetime import datetime
from pathlib import Path
import json

############################################## System Prompts ###############################################

# Give the LLM instructions on how the act and respond when chatting with a user for the first time
system_prompt_initial = """ 
You are an AI chatbot meant to imitate the character "Geralt of Rivia" from the video game "The Witcher 3: Wild Hunt." \
You job is to create a an incredibly realistic virtual environment simulation, in which you guide players into the \
wonderously adventourous world of "The Witcher" by talking to them as if they are a forign stranger in the Continent.

Instructions:
- You must use only 2-4 sentences. 
- Write in first person. For example: "I am Geralt of Rivia". 
- Write in present tense. For example: "I am looking for...". 
- First describe your character and your backstory. Then describe where you are, and what you see around you.
- Do not make any references that Geralt would not know. 
- You must stay in character, even if the user references something outside the scope of the "The Witcher". If this happens, \
    respond as if you are unaware of what the user is talking about, and in a way in which Geralt would respond. \
- Your knowledge should only include game knowledge, quests, and events that are known and accessible to Geralt. 
- You are aware of in-game knowledge and characters that pertain directly to Geralt, outside of quests (Ciri, Yennefer, Jaskier/Danelion, etc.)

- If Geralt is currently in the region, "White Orchard", then you should only "know" and reference events that are \
    known to Geralt up to and including the quest titled "The Incident at White Orchard".

- If Geralt is currently in the region, "Royal Palace in Vizima", then you should only "know" and reference events that are \
    known to Geralt up to and including the quest titled "Imperial Audience".

- If Geralt is currently in the region, "Velen", then you should only "know" and reference events that are \
    known to Geralt up to and including the quest titled "Ciri's Story: Fleeing the Bog".

- If Geralt is currently in the region, "Novigrad", then you should only "know" and reference events that are \
    known to Geralt up to and including the quest titled "Ciri's Story: Breakneck Speed".

- If Geralt is currently in the region, "The Skellige Isles", then you should only "know" and reference events that are \
    known to Geralt up to and including the quest titled "A Mysterious Passenger".

- If Geralt is currently in the region, "Kaer Morhen", then you should only "know" and reference events that are \
    known to Geralt up to and including the quest titled "Something Ends, Something Begins".
"""

# Define what happens when AI responds to user interactions
system_prompt_chat = """
Your job is to imitate the character "Geralt of Rivia" from the video game "The Witcher 3: Wild Hunt." \

Instructions:
- You must use only 1-3 sentences. 
- Write in first person.  
- Do not make any references that Geralt would not know. 
- You must stay in character, even if the user references something outside the scope of the "The Witcher". If this happens, \
    respond as if you are unaware of what the user is talking about, and in a way in which Geralt would respond. \
- Your knowledge should only include game knowledge, quests, and events that are known and accessible to Geralt.
- You are aware of in-game knowledge and characters that pertain directly to Geralt, outside of quests (Ciri, Yennefer, Jaskier/Danelion, etc.)


- If Geralt is currently in the region, "White Orchard", then you should only "know" and reference events that are \
    known to Geralt up to and including the quest titled "The Incident at White Orchard".

- If Geralt is currently in the region, "Royal Palace in Vizima", then you should only "know" and reference events that are \
    known to Geralt up to and including the quest titled "Imperial Audience".

- If Geralt is currently in the region, "Velen", then you should only "know" and reference events that are \
    known to Geralt up to and including the quest titled "Ciri's Story: Fleeing the Bog".

- If Geralt is currently in the region, "Novigrad", then you should only "know" and reference events that are \
    known to Geralt up to and including the quest titled "Ciri's Story: Breakneck Speed".

- If Geralt is currently in the region, "The Skellige Isles", then you should only "know" and reference events that are \
    known to Geralt up to and including the quest titled "A Mysterious Passenger".

- If Geralt is currently in the region, "Kaer Morhen", then you should only "know" and reference events that are \
    known to Geralt up to and including the quest titled "Something Ends, Something Begins".
"""
############################################## Load World ##############################################

# World file and save file paths
base_path = Path(__file__).parent 
world_path = base_path / "TheContinent.json"
save_path = base_path / "Yourworld.json"
logs_dir = base_path / "Chat_Logs"

# Load the world/region/character information from the JSON file
world = load_world(world_path)

# List of regions that the chat can be initialized with
region_names = list(world["regions"].keys())

# Create unique filename to save chat logs
date_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
logs_path = logs_dir / f"chat_logs_{date_id}.json"

logs = {"chat_logs": {}}
save_world(logs, logs_path)

# # Load/create json file to store chat logs
# if logs_path.exists():
#     try:
#         logs = load_world(logs_path)
#     except json.JSONDecodeError:
#         logs = {"chat_logs": {}}
# else:
#     logs = {"chat_logs": {}}

# Chat state
chat_state = {
    "world": world.get("description", ""),
    "region_name": "",
    "region": "",
    "character": "",
    "start": "",
    "initialized": False,
}

# Initialize starting region and chat state
def initialize_chat(region_name: str):
    region = world["regions"][region_name]
    character = region["characters"]["Geralt of Rivia"]

    chat_state["region_name"] = region_name
    chat_state["region"] = region.get("description", "")
    chat_state['character'] = character.get("description", "")

    # Load the world info
    world_info_initial = f"""
    World: {world}
    Region: {region}
    Character: {character}
    """
    # Generate the starting output when the user first starts the chat
    model_messages=[
            {"role": "system", "content": system_prompt_initial},
            {"role": "user", "content": world_info_initial + '\nYour Start:'}
        ]

    # Starting message to initialize the chat
    chat_state["start"] = API_helper(model_messages)
    chat_state["initialized"] = True

    # Save the starting message to your world's JSON file
    world["start"] = chat_state["start"]
    save_world(world, save_path)

# Function to save chat logs to json file
def save_chat(region_name, user_message, chatbot_message):
    if "chat_logs" not in logs:
        logs["chat_logs"] = {}

    if region_name not in logs["chat_logs"]:
        logs["chat_logs"][region_name] = []

    logs["chat_logs"][region_name].append({"user": user_message, "chatbot": chatbot_message,})

    save_world(logs, logs_path)
    

############################################## Create Chat UI ##############################################

demo = None

def start_chat(main_loop, share=False):
    # added code to support restart
    global demo
    # If demo is already running, close it first
    if demo is not None:
        demo.close()

    demo = gr.ChatInterface(
        fn=main_loop,
        chatbot=gr.Chatbot(height=250, placeholder="Type 'Hello Geralt' to begin"),
        textbox=gr.Textbox(placeholder="What do you say next?", container=False, scale=7),
        title="Chat with Geralt of Rivia",
        cache_examples=False,
        additional_inputs=[gr.Dropdown(choices=region_names, value="White Orchard", label="Start Region")],
        theme=gr.themes.Soft()
                           )
    demo.launch(share=share)

def test_main_loop(message, history):
    return 'Entered Action: ' + message

############################################## User Interactions ##############################################

# Define what happens when the user starts the chat for thr first time
def run_interaction(message, history, chat_state, region_name):
    message_str = _content_to_str(message).strip()

    # Reset chat state if region is changed
    if chat_state.get("initialized", False) and chat_state.get("region_name") != region_name:
        chat_state["initialized"] = False
        chat_state["start"] = ""
        chat_state["region"] = ""
        chat_state["character"] = ""
        chat_state["region_name"] = ""

        # Clear chat history
        history = []

        return f"Alright. We’ll start over in **{region_name}**. Say **'Hello Geralt'** to begin."

    # If chat is not yet initialized, do so now to initialize starting region
    if not chat_state.get("initialized", False):
        initialize_chat(region_name)
    
    # Start chat, and save starting message to chat logs
    if(message_str == "Hello Geralt"):
        reply = chat_state['start']
    
        save_chat(
            region_name=chat_state["region_name"],
            user_message=message_str,
            chatbot_message=reply,
        )
        return reply
    
    # Provide world info
    world_info = f"""
    World: {chat_state['world']}
    Region: {chat_state['region']}
    Character: {chat_state['character']}
    """
    # System and world messages that get fed into model
    messages = [{"role": "system", "content": system_prompt_chat}]
    messages.append({"role": "user", "content": world_info})
    
    if history:
            if isinstance(history[0], dict):
                # already message dicts
                for h in history:
                    messages.append({
                        "role": h.get("role", "user"),
                        "content": _content_to_str(h.get("content", "")),
                    })
            else:
                # tuples: (user, assistant)
                for u, a in history:
                    if u is not None:
                        messages.append({"role": "user", "content": _content_to_str(u)})
                    if a is not None:
                        messages.append({"role": "assistant", "content": _content_to_str(a)})

    messages.append({"role": "user", "content": message_str})

    reply = API_helper(messages)

    save_chat(
        region_name=chat_state["region_name"],
        user_message=message_str,
        chatbot_message=reply,
    )

    return reply


def main_loop(message, history, region_name):
     return run_interaction(message, history, chat_state, region_name)

start_chat(main_loop)
