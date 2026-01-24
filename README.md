# Smart NPCs: An Evaluation Framework for LLM-Based NPC Development and Testing
Undergraduate thesis project in collaboration with Cristiano Politowski Mariana Shimabukuro.

## About The Project
As AI becomes more prevelant in the video game industry, there is an increasing amount of research done on Generative AI-powered Non-Player Characters (NPCs). Generative AI-powered NPCs, also known as "Smart NPCs" are connected to a large language to generate textual responses to a player's input, guided by prompts from the developer. This new type of development opens a door for more interactive and dynamic NPC bahaviour, in which smart NPCs can adapt to their environment based on the choices made by a player. This could give each player a unique experience, as no two interaction with a given player would be exactly the same.

As promising as this area of research is, it brings about a new set of challenges that must be addressed. Generative AI is known to hallucinate, falsify information, and lack emotional depth, and so while it's possible to create more dynamic environments, it is challenging to create individual smart NPCs that have consistent personalities. 

AI guardrails can be implemented that act as a digital filter, ensuring user inputs and LLM outputs remain within boundaries established by the developer. These guardrails help maintain appropriate content, improve player safety, and enforce consistent character behaviour.

## Current Features
- AI-based chatbot in a Sandbox environment that connects to an open-source large language model, forming the basis for a Smart NPC.
- System Prompts inform the model about its role in this project, specifically that it will be protraying a designated character from a video game, as chosen by the uer. A series of instructions is included that detail rules for formatting, length of responses, and general contraints.
- Hierarchial Content Generation that builds a world based on a video game: specified b the user:
  - World Generation: Asks the model to generate a description of the world that the the LLM's character exists in.
  - Region Generation: Asks the model to generates a description of each major region/city/location that exists in the video game world. Each region/city/location is subsequently added to the world's JSON data.
  - Character Generation: Asks the model to generate a description of the character that the mdoel will be protraying. A new character description is generated for each of the previously generated regions, outlining the charcter's reason for being in that region, and the role they occupy there.
  - Chat interface built using Gradio.
- Saves chat logs for later review.

## Features To Be Implemented:
- Use guardrails imported from the "Guardrails AI Hub" to govern the AI model, ensuring the AI model follows a specific set of guidelines put forth by the user.
- User interface that allows users to easily select the guardrails they wish to enable from a given library. The selected guardrails will then be automatically implemented into the sandbox environment.

## External Resources That Were Used:
- Gradio
- Ollama
- Guardrails AI

## How to Run
### Create a virtual Environment:

### Download the required dependecies:

### Ensure Ollama is running:

### Run the Script 'WorldCreation.py' in the command line:

This is will generate a JSON file containing the world information.

### Run the script 'RunChat.py' in the command line:

You will be prompted to open the application in your browser. This will open the chat interface where you can begin chatting.



