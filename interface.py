# '''
# NOTES
# Users can input their own information into each field, or they can upload a text document which will 
# automatically populate the fields based on the provided information.

# Personality traits are chosen from a drop down menu that filters the traits based on the keyboard inputs
# to make finding specific traits easier.

# Relationships are defined by populating a selection of radio buttons with previously created NPCs (if any).
# If no other NPCs have been created, a button, when clicked, takes the user back to the NPC profile screen where
# they can create another NPC. Or they have the option to just describe a relationship with another character in a text box.
# If the text box is used instead, then that characters attributes are automatically use to generate a new NPC by the system
# and placed in the world.
# When an NPC is clicked on, a text box drops down below where the user can enter the nature of their relationship.
# '''
import gradio as gr
from dataclasses import dataclass, field, replace
import statistics
import math
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
import json
import itertools
import os
from generate_npc_admin_session import build_trait_phrase, build_biography_description, build_item_preambles, generate_admin_session
 

@dataclass
class NPCProfile:
    # Basic
    name: str = ""
    pronouns: str = ""
    age: str = ""
    role: str = ""
    race: str = ""
    appearance: str = ""
    backstory: str = ""
    # Additional
    relationships: str = ""
    skills: str = ""
    opinions: str = ""
    loves: str = ""
    hates: str = ""
    hobbies: str = ""
    goals: str = ""
    flaws: str = ""
    # Personality
    personality: list = field(default_factory=list)
    
    def create_profile_UI(self):
        def section(title, content):
            if not content or not content.strip():
                return ""
            return f"\n{'━' * 25}\n  {title.upper()}\n{'━' * 25}\n{content.strip()}\n"
         
        header = (
            f"\n╔══════════════════════════════════════════════════╗\n"
            f"  CHARACTER PROFILE\n"
            f"╚══════════════════════════════════════════════════╝\n\n"
            f"  Name     :  {self.name or '—'}\n"
            f"  Role     :  {self.role or '—'}\n"
            f"  Gender   :  {self.pronouns or '—'}\n"
            f"  Age      :  {self.age or '—'}\n"
        )

        body = (
            section("Appearance",           self.appearance)
            + section("Backstory",          self.backstory)
            + section("Relationships",      self.relationships)
            + section("Special Skills",     self.skills)
            + section("Opinions & Beliefs", self.opinions)
            + section("Things They Love",   self.loves)
            + section("Things They Hate",   self.hates)
            + section("Hobbies & Interests",self.hobbies)
            + section("Long-Term Goals",    self.goals)
            + section("Flaws",              self.flaws)
        )

        if not body.strip():
            body = "\n  [ No details filled in yet ]\n"
        
        return header + body + f"\n{'═' * 50}\n"
    



trait_adjectives = {
    "Extaversion": [
        ("introverted",            "extraverted"),
        ("unenergetic",            "energetic"),
        ("silent",                 "talkative"),
        ("timid",                  "bold"),
        ("inactive",               "active"),
        ("unassertive",            "assertive"),
        ("unfriendly",             "friendly"),
        ("unadventurous",          "adventurous and daring"),
        ("gloomy",                 "joyful"),
    ],
    "Agreeableness": [
        ("unkind",                 "kind"),
        ("uncooperative",          "cooperative"),
        ("selfish",                "unselfish"),
        ("disagreeable",           "agreeable"),
        ("distrustful",            "trustful"),
        ("stingy",                 "generous"),
        ("immoral",                "moral"),
        ("dishonest",              "honest"),
        ("unaltruistic",           "altruistic"),
        ("self-important",         "humble"),
        ("unsympathetic",          "sympathetic"),
    ],
    "Conscientiousness": [
        ("disorganized",           "organized"),
        ("irresponsible",          "responsible"),
        ("negligent",              "conscientious"),
        ("impractical",            "practical"),
        ("careless",               "thorough"),
        ("lazy",                   "hardworking"),
        ("extravagant",            "thrifty"),
        ("unsure",                 "self-efficacious"),
        ("messy",                  "orderly"),
        ("undisciplined",          "self-discplined"),
    ],
    "Neuroticism": [
        ("calm",                   "angry"),
        ("relaxed",                "tense"),
        ("at ease",                "nervous"),
        ("contented",              "discontented"),
        ("easygoing",              "anxious"),
        ("patient",                "irritable"),
        ("happy",                  "depressed"),
        ("unselfconscious",        "self-conscious"),
        ("level-headed",           "impulsive"),
        ("emotionally stable",     "emotionally unstable"),
    ],
    "Openness": [
        ("unintelligent",          "intelligent"),
        ("unanalytical",           "analytical"),
        ("unreflective",           "reflective"),
        ("uninquisitive",          "curious"),
        ("unimaginative",          "imaginative"),
        ("uncreative",             "creative"),
        ("unsophisticated",        "sophisticated"),
        ("artistically unappreciative", "artistically appreciative"),
        ("unaesthetic",            "aesthetic"),
        ("emotionally closed",     "emotionally aware"),
        ("predictable",            "spontaneous"),
        ("socially conservative",  "socially progressive"),
    ],
}

trait_shortforms = {
    "Extaversion" : "ext",
    "Agreeableness": "agr",
    "Conscientiousness": "con",
    "Neuroticism": "neu",
    "Openness": "ope",
}

# A list of lists --> list of all adjective pairs and their respective ratings --> ("introverted", "extraverted", 1)
all_ratings = []

# A list of all the inputs/ratings
personality_inputs = []

# a list of all of the keys that correspond with the input. The first element is the trait, 
# second element is the low marker, and the third element is the high marker --> ("ext", "introverted, "extraverted")
personality_keys = []

# Dict of the median ratings from create_radar()
median_ratings = {}


# The original expirement set the trait level (Ex. ext0-agr0-con0-neu0-ope1), and has the trait adjective phrase
# build using the level qualifier that correaltes to the traits level. So all the adjectives had the same qualifer for a given
# level. 
#
# My framework instead has the user rate their character's trait level for each individual adjective, and then to get the overall
# level for the trait, I take the median of the trait ratings.
def create_radar(*args):
    *ratings, state = args

    #global all_ratings

    trait_ratings = {}
    for (trait, low, high), rating in zip(personality_keys, ratings):
        if rating is not None:
            trait_ratings.setdefault(trait, []).append(int(rating))
    
    # Radar plot labels and scores
    labels = []
    medians = []
    for trait, scores in trait_ratings.items():
        median = math.floor(statistics.median(scores))
        medians.append(median)
        labels.append(trait)
        median_ratings[trait_shortforms[trait]] = median


        print(f"{trait}: {scores}: {median}")

    personality = []
    for (trait, low, high), rating in zip(personality_keys, ratings):
        personality.append((trait, (low, high), int(rating)))
    updated = replace(state, personality=personality)
    
    print(personality)

    print(f"all_ratings populated: {len(all_ratings)} entries")  # add this
    print(all_ratings[:3]) 
    
    # Create the Radar Plot
    fig = go.Figure(data=go.Scatterpolar(
        r=medians,
        theta=labels,
        fill='toself',
        name='Personality'
    ))

    fig.update_layout(
        height=350,
        margin=dict(t=200, b=200, l=200, r=200),
        polar=dict(
            radialaxis=dict(
            visible=True,
            range=[1,9]
            ),
    ),
    showlegend=False
    )
    generate_session(updated)
    return fig, updated, updated.create_profile_UI()


def update_basic(name, pronouns, age, role, race, appearance, backstory, state):
        updated = replace(state, name=name, pronouns=pronouns, age=age, role=role, race=race,
        appearance=appearance, backstory=backstory)
        return updated, updated.create_profile_UI()
    
def update_additional(relationships, skills, opinions, loves, hates, hobbies, goals, flaws, state):
    updated = replace(state, relationships=relationships, skills=skills, opinions=opinions, loves=loves,
                        hates=hates, hobbies=hobbies, goals=goals, flaws=flaws)
    return updated, updated.create_profile_UI()

def generate_session(state):
            
    trait_phrase = build_trait_phrase(state.personality)
    bio = build_biography_description(state)
    item_preambles = build_item_preambles(trait_phrase, bio, median_ratings)
    generate_admin_session(item_preambles)
    return "✅ Admin session saved successfully."

#######################################################################################################
with gr.Blocks(
    css="""
    .stretch-radio .wrap {
        display: flex !important;
        flex-direction: row !important;
        justify-content: space-between !important;
        width: 100% !important;
    }
    .stretch-radio .wrap label {
        flex: 1 !important;
        text-align: center !important;
    }
    .gap-row {
        margin-top: 16px !important;
        margin-bottom: 16px !important;
    }
               
""") as demo:
# Create a profile state to update the profile overview each the submit button is clicked for each tan
    profile_state = gr.State(NPCProfile())

    gr.Markdown("Smart NPC Generator")
    gr.Markdown("<br>")

    with gr.Row():
        with gr.Column(scale=3):
            with gr.Tab("Basic Profile"):
                gr.Markdown("Please enter the following details about your character. Be sure to write in FIRST PERSON.")
                with gr.Row():
                    name = gr.Textbox(label="Name", placeholder="e.g. Mira Stonehaven")
                    pronouns = gr.Textbox(label="Pronouns", placeholder="e.g. She/Her, He/Him, They/Them...")
                
                with gr.Row():
                    age = gr.Textbox(label="Age", placeholder="e.g. 47, 113...")
                    role = gr.Textbox(label="Role", placeholder="e.g. Court Wizard, Innkeeper...")
                    race = gr.Textbox(label="Race", placeholder="e.g. Human, Orc, Halfling...")
            
                appearance   = gr.Textbox(label="Current Appearance", lines=3, placeholder="e.g. I have a weathered face and a long grey beard. I walk with a slight limp from an old battle wound.")
                backstory      = gr.Textbox(label="Backstory", lines=4, placeholder="e.g. I have defended my village for thirty years. I lost my family to a great war. I have lived alone in the mountains ever since.")

                save_basic = gr.Button("Save")


            with gr.Tab("Additional Information"):
                gr.Markdown("Please enter the following details about your character. Be sure to write in FIRST PERSON.")

                # ── History & Relationships ───────────────────
                relationships  = gr.Textbox(label="Relationships with Other Characters", lines=3, placeholder="e.g. I have one trusted friend, the village elder. I lost my brother to a dragon attack.")

                # ── Traits ────────────────────────────────────
                skills    = gr.Textbox(label="Special Skills", lines=2, placeholder="e.g. I am an expert swordsman. I know the mountain passes better than anyone.")
                opinions  = gr.Textbox(label="Personal Opinions & Beliefs", lines=2, placeholder="e.g. I believe honour is more important than victory. I think most wars are started by cowards")

                with gr.Row():
                    loves  = gr.Textbox(label="Things They Love", lines=3, placeholder="e.g. I love the silence of snowfall. I love a good fire at the end of the day.")
                    hates  = gr.Textbox(label="Things They Hate", lines=3, placeholder="e.g. I hate dishonesty. I hate large crowds.")

                hobbies = gr.Textbox(label="Hobbies & Interests", lines=2, placeholder="e.g. I sharpen my blade every evening. I carve small wooden figures when alone.")
                goals   = gr.Textbox(label="Long-Term Goals & Ambitions", lines=2, placeholder="e.g. I want to ensure no child in my village grows up without a home. I hope to find peace before I die.")
                flaws   = gr.Textbox(label="Flaws", lines=2, placeholder="e.g. I struggle to ask for help. I hold grudges for a very long time.")
                
                save_additional = gr.Button("Save")

           
           
            with gr.Tab("Personality Traits"):
                gr.Markdown("For each of the following traits, rate where your character's personality lies on a scale from 1 to 9.")
                gr.Markdown("For example, rate your character's friendliness on a scale from 1 = unfriendly to 9 = friendly:")
                with gr.Row(elem_classes=["stretch-radio"]):
                    with gr.Column(scale=1, min_width=0):
                        gr.Markdown("1 = extremely unfriendly<br>"
                                    "2 = very unfriendly<br>"
                                    "3 = unfriendly")

                    with gr.Column(scale=1, min_width=0):
                        gr.Markdown("4 = a bit unfriendly<br>"
                                    "5 = neither unfriendly nor friendly<br>"
                                    "6 = a bit friendly")
                        
                    with gr.Column(scale=1, min_width=0):
                        gr.Markdown("7 = friendly<br>"
                                    "8 = very friendly<br>"
                                    "9 = extremely friendly")
                gr.Markdown("<br>")



                for trait, adj_pairs in trait_adjectives.items():

                    gr.Markdown(f"## Category: {trait.upper()}")
                    with gr.Column():
                        for low, high in adj_pairs:
                            adjective_rating = []
                            adjective_rating = [low, high]

                            with gr.Row(elem_classes=["gap-row"]):
                                gr.Markdown(f"### {low}")
                                gr.Markdown(f"### <div style='text-align:right'>{high}</div>")
                            rating = gr.Radio(["1", "2", "3", "4", "5", "6", "7", "8", "9"], 
                                                                value="5",
                                                                label=None,
                                                                show_label=False,
                                                                elem_classes=["stretch-radio"], 
                                                                type="value")
                            adjective_rating.append(rating)
                            personality_inputs.append(rating)
                            personality_keys.append((trait, low, high))
                            
                            all_ratings.append(adjective_rating)
                         
                save_personality = gr.Button("Save")
                


        with gr.Column(scale=2):
            with gr.Column(scale=2):
                radar = gr.Plot(label="Personality Distribution")
                output = gr.Textbox(
                    label=None,
                    lines=48,
                    interactive=False,
                    placeholder="Fill in the fields to generate your NPC Profile.",
                    elem_classes=["output-card"]
                )
                # generate_file = gr.Button("Save as JSON")
                # generate_status = gr.Textbox(label=None, interactive=False, visible=True)

                

    save_basic.click(
        fn=update_basic, 
        inputs=[name, pronouns, age, role, race, appearance, backstory, profile_state], 
        outputs=[profile_state, output])
    
    save_additional.click(
        fn=update_additional, 
        inputs=[relationships, skills, opinions, loves, hates, hobbies, goals, flaws, profile_state],
        outputs=[profile_state, output])
   
    save_personality.click(
        fn=create_radar, 
        inputs=personality_inputs + [profile_state],
        outputs=[radar, profile_state, output])
    
    # generate_file.click(
    #     fn=generate_session,
    #     inputs=[profile_state],
    #     outputs=[generate_status]
        
    # )
    

if __name__ == "__main__":
    demo.launch(share=False)


#######################################################################################################










