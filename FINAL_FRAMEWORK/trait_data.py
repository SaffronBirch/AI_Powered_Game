"""
Big Five personality trait data.
Adjective pairs, short-form codes, and linguistic level qualifiers.
"""

TRAIT_DATA = {
    "Extraversion": [
        ("introverted", "extraverted"),
        ("unenergetic", "energetic"),
        ("silent", "talkative"),
        ("timid", "bold"),
        ("inactive", "active"),
        ("unassertive", "assertive"),
        ("unfriendly", "friendly"),
        ("unadventurous", "adventurous"),
        ("gloomy", "joyful"),
    ],
    "Agreeableness": [
        ("unkind", "kind"),
        ("uncooperative", "cooperative"),
        ("selfish", "unselfish"),
        ("disagreeable", "agreeable"),
        ("distrustful", "trustful"),
        ("stingy", "generous"),
        ("dishonest", "honest"),
        ("unsympathetic", "sympathetic"),
    ],
    "Conscientiousness": [
        ("disorganized", "organized"),
        ("irresponsible", "responsible"),
        ("negligent", "conscientious"),
        ("impractical", "practical"),
        ("careless", "thorough"),
        ("lazy", "hardworking"),
        ("messy", "orderly"),
        ("undisciplined", "self-disciplined"),
    ],
    "Neuroticism": [
        ("calm", "angry"),
        ("relaxed", "tense"),
        ("at ease", "nervous"),
        ("contented", "discontented"),
        ("easygoing", "anxious"),
        ("patient", "irritable"),
        ("happy", "depressed"),
        ("emotionally stable", "emotionally unstable"),
    ],
    "Openness": [
        ("unintelligent", "intelligent"),
        ("unanalytical", "analytical"),
        ("unreflective", "reflective"),
        ("uninquisitive", "curious"),
        ("unimaginative", "imaginative"),
        ("uncreative", "creative"),
        ("unsophisticated", "sophisticated"),
        ("predictable", "spontaneous"),
    ],
}

TRAIT_SHORT = {
    "Extraversion": "ext",
    "Agreeableness": "agr",
    "Conscientiousness": "con",
    "Neuroticism": "neu",
    "Openness": "ope",
}

# Linguistic qualifiers for each of the 9 rating levels
# Level 1 = extremely low, Level 5 = neutral, Level 9 = extremely high
LEVEL_QUALIFIERS = {
    1: "extremely {low}",
    2: "very {low}",
    3: "{low}",
    4: "a bit {low}",
    5: "neither {low} nor {high}",
    6: "a bit {high}",
    7: "{high}",
    8: "very {high}",
    9: "extremely {high}",
}
