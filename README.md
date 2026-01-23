# Smart NPCs: An Evaluation Framework for LLM-Based NPC Development and Testing

## About The Project
As AI becomes more prevelant in the video game industry, there is an increasing amount of research done on Generative AI-powered Non-Player Characters (NPCs). 
Generative AI-powered NPCs, also known as "Smart NPCs" are connected to a large language to generate textual responses to a player's input, guided by prompts from the developer. 
This new type of development opens a door for more interactive and dynamic NPC bahaviour, in which smart NPCs can adapt to their environment based on the choices made by a player. 
This could give each player a unique experience, as no two interaction with a given player would be exactly the same.

As promising as this area of research is, it brings about a new set of challenges that must be addressed. Generative AI is known to hallucinate, falsify information, and lack emotional depth, 
and so while it's possible to create more dynamic environments, it is challenging to create individual smart NPCs that have consistent personalities. 

AI guardrails can be implemented that act as a digital filter, ensuring user inputs and LLM outputs remain within boundaries established by the developer. These guardrails help maintain 
appropriate content, improve player safety, and enforce consistent character behaviour.

## Features
- Hierarchial Content Generation that builds content using previously generated content. The hierarchy structure is as follows:
  - System prompts: Imforms the model that it will be protraying a designated character from a video game. (the character being portrayed depends on input from the user). 
    A series of isntructions is included that detail rules for formatting, length of responses, and general contraints.
    - Currently only the character "Geralt of Rivia" from "The Witcher 3: Wild Hunt" is available. More characters to me added soon.
  - World Prompt: Tells the model to generate a description of the world that the chat will take place in.
  - Region Prompt:
  - Character Prompt: 
- Connected to open-source large language models.
- Saves chat logs for later review.

## How to Run

