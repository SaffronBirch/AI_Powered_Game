"""
Prompt templates for world/character generation and NPC chat.
"""

import math
import statistics
from trait_data import LEVEL_QUALIFIERS, TRAIT_SHORT


# ─────────────────────────────────────────────────────────────
# TRAIT PHRASE BUILDER
# ─────────────────────────────────────────────────────────────
def build_trait_phrase(personality: dict) -> str:
    """
    Given a personality dict like:
        { "Extraversion": [{"low":"introverted","high":"extraverted","rating":7}, ...], ... }
    compute the median rating per trait and produce a natural-language phrase.
    """
    phrases = []
    for trait, items in personality.items():
        vals = sorted([it["rating"] for it in items])
        mid = len(vals) // 2
        median = vals[mid] if len(vals) % 2 else math.floor((vals[mid - 1] + vals[mid]) / 2)

        template = LEVEL_QUALIFIERS.get(median, LEVEL_QUALIFIERS[5])
        low = items[0]["low"]
        high = items[0]["high"]
        phrases.append(template.replace("{low}", low).replace("{high}", high))

    return "I am " + ", ".join(phrases) + "."


# ─────────────────────────────────────────────────────────────
# SYSTEM PROMPT: GENERATOR (world/region/character creation)
# ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT_GENERATOR = """You help create immersive fantasy worlds, regions, and characters.

Instructions:
- Return ONLY valid JSON (no markdown fences, no extra text).
- Use clear, vivid language.
- Keep descriptions to 3-5 sentences each.
"""


# ─────────────────────────────────────────────────────────────
# GENERATION PROMPTS
# ─────────────────────────────────────────────────────────────
def world_gen_prompt() -> str:
    return """Generate a unique fantasy world.

Return ONLY valid JSON:
{
  "game_name": "Setting Name",
  "world_name": "World Name",
  "description": "World description (3-5 sentences)"
}"""


def region_gen_prompt(world_name: str, world_desc: str) -> str:
    return f"""Generate a region for the world "{world_name}": {world_desc}

Return ONLY valid JSON:
{{
  "name": "Region Name",
  "description": "Region description (3-5 sentences)"
}}"""


def character_gen_prompt(world_name: str, world_desc: str) -> str:
    return f"""Generate a character for the world "{world_name}": {world_desc}

Return ONLY valid JSON:
{{
  "name": "Character Name",
  "pronouns": "e.g. She/Her",
  "age": "e.g. 47",
  "role": "e.g. Court Wizard",
  "race": "e.g. Human",
  "appearance": "First-person description (1-2 sentences)",
  "backstory": "First-person backstory (2-3 sentences)",
  "skills": "First-person skills (1-2 sentences)",
  "goals": "First-person goals (1-2 sentences)",
  "flaws": "First-person flaws (1-2 sentences)"
}}"""


# ─────────────────────────────────────────────────────────────
# SYSTEM PROMPTS: NPC CHAT
# ─────────────────────────────────────────────────────────────
def build_system_prompt_initial(world: dict, character: dict) -> str:
    char_name = character.get("name", "the character")
    world_name = world.get("world_name", "this world")
    trait_phrase = character.get("traitPhrase", "")
    bio = character.get("bio", "")

    return f"""You must imitate and act as {char_name} from the world of {world_name}.
Your job is to create an incredibly realistic virtual simulation by talking to the user as if they \
are a foreign stranger in {world_name}.

Instructions:
- You MUST use only 2-4 sentences.
- You MUST write in first person and present tense.
- First describe your character and backstory. Then describe where you are and what you see around you.
- Do not make any references that {char_name} would not know.
- Stay in character at all times. If the user references something outside {world_name}, respond as \
if you are unaware of it.
- Your knowledge should only include what {char_name} would know.

Character personality: {trait_phrase}
Character biography: {bio}"""


def build_system_prompt_chat(world: dict, character: dict) -> str:
    char_name = character.get("name", "the character")
    world_name = world.get("world_name", "this world")
    trait_phrase = character.get("traitPhrase", "")

    return f"""You must imitate and act as {char_name} from the world of {world_name}.

Instructions:
- You MUST use only 2-4 sentences.
- You MUST write in first person.
- Do not make any references that {char_name} would not know.
- Stay in character at all times. If the user references something outside {world_name}, respond as \
if you are unaware of it.
- Your knowledge should only include what {char_name} would know.

Character personality: {trait_phrase}
Character details: {character.get('description', '')}
Backstory: {character.get('backstory', '')}"""
