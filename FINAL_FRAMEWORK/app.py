"""
Smart NPC Adventure Engine
──────────────────────────
A unified AI-Dungeon-style app combining:
  • World creation (manual or AI-generated)
  • Character creation with Big Five personality traits
  • Immersive NPC chat interface

Run with:  streamlit run app.py
"""

import streamlit as st
import json
import math
import statistics
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
import plotly.graph_objects as go

from llm_helper import call_llm, generate_json, get_provider_info
from trait_data import TRAIT_DATA, TRAIT_SHORT, LEVEL_QUALIFIERS
from prompts import (
    build_trait_phrase,
    build_system_prompt_initial,
    build_system_prompt_chat,
    world_gen_prompt,
    region_gen_prompt,
    character_gen_prompt,
    SYSTEM_PROMPT_GENERATOR,
)
from memory_architecture import GenerativeAgent, MemoryStream

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart NPC Adventure",
    page_icon="⚔",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS — dark fantasy theme
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;700&family=Crimson+Text:ital,wght@0,400;0,600;1,400&family=Fira+Code:wght@400&display=swap');

:root {
    --bg: #0a0a0f;
    --surface: #12121a;
    --surface2: #1a1a28;
    --border: #2a2a3a;
    --accent: #c9a84c;
    --accent-dim: #a0842e;
    --text: #d4d0c8;
    --text-dim: #8a8678;
    --text-bright: #f0ece0;
    --player-color: #4a7c59;
    --npc-color: #c9a84c;
}

.stApp {
    background-color: var(--bg) !important;
    color: var(--text);
    font-family: 'Crimson Text', Georgia, serif;
}

/* Hide default streamlit branding */
#MainMenu, footer, header {visibility: hidden;}
.stDeployButton {display: none;}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] .stMarkdown { color: var(--text); }

/* Headings */
h1, h2, h3 {
    font-family: 'Cinzel', serif !important;
    color: var(--accent) !important;
    letter-spacing: 3px;
}

/* Text inputs & text areas */
.stTextInput input, .stTextArea textarea {
    background-color: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    font-family: 'Crimson Text', serif !important;
    font-size: 15px !important;
    border-radius: 4px !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 1px var(--accent-dim) !important;
}

/* Labels */
.stTextInput label, .stTextArea label, .stSelectbox label, .stSlider label, .stRadio label {
    font-family: 'Cinzel', serif !important;
    font-size: 11px !important;
    color: var(--accent) !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
}

/* Buttons */
.stButton > button {
    font-family: 'Cinzel', serif !important;
    letter-spacing: 2px !important;
    border-radius: 4px !important;
    font-weight: 700 !important;
    transition: all 0.2s !important;
}
.stButton > button[kind="primary"], 
div[data-testid="stFormSubmitButton"] > button {
    background: linear-gradient(135deg, var(--accent), var(--accent-dim)) !important;
    color: var(--bg) !important;
    border: none !important;
}
.stButton > button[kind="secondary"] {
    background: transparent !important;
    color: var(--accent) !important;
    border: 1px solid rgba(201,168,76,0.3) !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    border-bottom: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Cinzel', serif !important;
    font-size: 13px !important;
    letter-spacing: 2px !important;
    color: var(--text-dim) !important;
    background: transparent !important;
}
.stTabs [aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
}

/* Select boxes */
.stSelectbox [data-baseweb="select"] {
    background-color: var(--surface) !important;
}
.stSelectbox [data-baseweb="select"] > div {
    background-color: var(--surface) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
}

/* Sliders */
.stSlider [data-baseweb="slider"] [role="slider"] {
    background-color: var(--accent) !important;
}
.stSlider [data-testid="stTickBarMin"], .stSlider [data-testid="stTickBarMax"] {
    color: var(--text-dim) !important;
    font-size: 12px;
}

/* Dividers */
hr { border-color: var(--border) !important; }

/* Chat messages */
.chat-narrator {
    background: #2a1e2e;
    border-radius: 8px;
    padding: 12px 20px;
    text-align: center;
    font-style: italic;
    color: var(--text-dim);
    margin: 8px auto;
    max-width: 90%;
}
.chat-player {
    background: rgba(74,124,89,0.15);
    border-left: 3px solid var(--player-color);
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin: 8px 0 8px auto;
    max-width: 85%;
}
.chat-npc {
    background: #1e2a3a;
    border-left: 3px solid rgba(201,168,76,0.4);
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin: 8px auto 8px 0;
    max-width: 85%;
}
.chat-label {
    font-family: 'Cinzel', serif;
    font-size: 11px;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 4px;
}
.chat-label-player { color: var(--player-color); }
.chat-label-npc { color: var(--accent); }
.chat-label-narrator { color: var(--text-dim); letter-spacing: 3px; }
.chat-text {
    line-height: 1.7;
    font-size: 15px;
    font-family: 'Crimson Text', serif;
    color: var(--text);
    margin: 0;
}

/* Title screen */
.title-box {
    text-align: center;
    padding: 60px 20px;
}
.title-icon { font-size: 64px; margin-bottom: 16px; }
.title-main {
    font-family: 'Cinzel', serif;
    font-size: 48px;
    color: var(--text-bright);
    letter-spacing: 12px;
    text-shadow: 0 0 40px rgba(201,168,76,0.3);
    margin: 0;
}
.title-sub {
    font-family: 'Cinzel', serif;
    font-size: 16px;
    color: var(--accent);
    letter-spacing: 8px;
    margin-top: 8px;
}
.title-desc {
    color: var(--text-dim);
    font-size: 16px;
    margin-top: 24px;
    line-height: 1.6;
}

/* Region tags */
.region-tag {
    display: inline-block;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 6px 14px;
    font-family: 'Cinzel', serif;
    font-size: 13px;
    color: var(--text-bright);
    letter-spacing: 1px;
    margin: 4px;
}

/* Character roster card */
.roster-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 20px;
    text-align: center;
}
.roster-card h4 {
    font-family: 'Cinzel', serif;
    color: var(--text-bright) !important;
    margin: 8px 0 4px;
}
.roster-card p {
    color: var(--text-dim);
    font-size: 13px;
}

/* Expander styling */
.streamlit-expanderHeader {
    font-family: 'Cinzel', serif !important;
    color: var(--accent) !important;
}

/* Plotly chart background */
.stPlotlyChart { background: transparent !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# SESSION STATE INITIALIZATION
# ─────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "phase": "title",          # title | world | character | roster | adventure
        "world": None,
        "characters": {},
        "chat_messages": [],
        "chat_initialized": False,
        "active_character": None,
        "active_region": None,
        # Character creation temp state
        "char_basic": {
            "name": "", "pronouns": "", "age": "", "role": "",
            "race": "", "appearance": "", "backstory": "",
        },
        "char_additional": {
            "relationships": "", "skills": "", "opinions": "",
            "loves": "", "hates": "", "hobbies": "", "goals": "", "flaws": "",
        },
        "char_personality": {},
        # World creation temp
        "world_draft": {
            "game_name": "", "world_name": "", "description": "",
            "regions": {},
        },
        "world_step": "setup",     # setup | regions
        "chat_logs": [],
        "generative_agents": {},   # name -> GenerativeAgent.to_dict()
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # Initialize personality ratings
    if not st.session_state["char_personality"]:
        p = {}
        for trait, pairs in TRAIT_DATA.items():
            p[trait] = [{"low": l, "high": h, "rating": 5} for l, h in pairs]
        st.session_state["char_personality"] = p

init_state()


# ─────────────────────────────────────────────────────────────
# HELPER: Compute trait medians
# ─────────────────────────────────────────────────────────────
def compute_medians(personality: dict) -> dict:
    medians = {}
    for trait, items in personality.items():
        vals = sorted([it["rating"] for it in items])
        mid = len(vals) // 2
        medians[trait] = vals[mid] if len(vals) % 2 else math.floor((vals[mid - 1] + vals[mid]) / 2)
    return medians


def build_radar_chart(medians: dict):
    labels = list(medians.keys())
    values = list(medians.values())
    # Close the polygon
    labels_closed = labels + [labels[0]]
    values_closed = values + [values[0]]

    fig = go.Figure(data=go.Scatterpolar(
        r=values_closed,
        theta=labels_closed,
        fill="toself",
        fillcolor="rgba(201,168,76,0.15)",
        line=dict(color="#c9a84c", width=2),
        marker=dict(color="#c9a84c", size=6),
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[1, 9], gridcolor="#2a2a3a",
                            tickfont=dict(color="#8a8678", size=10)),
            angularaxis=dict(gridcolor="#2a2a3a",
                             tickfont=dict(color="#c9a84c", size=11, family="Cinzel")),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        margin=dict(t=40, b=40, l=60, r=60),
        height=350,
    )
    return fig


# ─────────────────────────────────────────────────────────────
# HELPER: Render chat message
# ─────────────────────────────────────────────────────────────
def render_message(msg: dict):
    role = msg["role"]
    content = msg["content"]

    if role == "narrator":
        st.markdown(f"""
        <div class="chat-narrator">
            <div class="chat-label chat-label-narrator">— Narrator —</div>
            <p class="chat-text">{content}</p>
        </div>""", unsafe_allow_html=True)
    elif role == "player":
        st.markdown(f"""
        <div class="chat-player">
            <div class="chat-label chat-label-player">▸ You</div>
            <p class="chat-text">{content}</p>
        </div>""", unsafe_allow_html=True)
    elif role == "npc":
        speaker = msg.get("speaker", "NPC")
        st.markdown(f"""
        <div class="chat-npc">
            <div class="chat-label chat-label-npc">▸ {speaker}</div>
            <p class="chat-text">{content}</p>
        </div>""", unsafe_allow_html=True)
    elif role == "system":
        st.markdown(f"<p style='text-align:center;color:#8a8678;font-style:italic;'>{content}</p>",
                    unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# HELPER: Save/Load world JSON
# ─────────────────────────────────────────────────────────────
SAVE_DIR = Path("saved_worlds")
SAVE_DIR.mkdir(exist_ok=True)
LOGS_DIR = Path("chat_logs")
LOGS_DIR.mkdir(exist_ok=True)


def save_world_file(world_data: dict, filename: str = None):
    if not filename:
        safe_name = world_data.get("world_name", "world").replace(" ", "_")
        filename = f"{safe_name}.json"
    path = SAVE_DIR / filename
    with open(path, "w") as f:
        json.dump(world_data, f, indent=2)
    return path


def save_chat_log(messages: list, world_name: str):
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = LOGS_DIR / f"chat_{world_name}_{ts}.json"
    with open(path, "w") as f:
        json.dump(messages, f, indent=2)
    return path


# ═════════════════════════════════════════════════════════════
# PHASE: TITLE SCREEN
# ═════════════════════════════════════════════════════════════
def render_title():
    st.markdown("""
    <div class="title-box">
        <div class="title-icon">⚔</div>
        <h1 class="title-main">SMART NPC</h1>
        <p class="title-sub">ADVENTURE ENGINE</p>
        <p class="title-desc">Craft your world. Shape your characters. Live the story.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        c1, c2 = st.columns(2)
        with c1:
            if st.button("✦ New Adventure", type="primary", use_container_width=True):
                st.session_state["phase"] = "world"
                st.rerun()
        with c2:
            uploaded = st.file_uploader("Load World JSON", type=["json"], label_visibility="collapsed")
            if uploaded:
                try:
                    data = json.loads(uploaded.read())
                    world = {
                        "game_name": data.get("game_name", data.get("name", "Unknown")),
                        "world_name": data.get("world_name", data.get("name", "Unknown")),
                        "description": data.get("description", data.get("world_description", "")),
                        "regions": data.get("regions", {}),
                    }
                    st.session_state["world"] = world

                    # Load characters if present
                    if "characters" in data and data["characters"]:
                        chars = {}
                        for name, c in data["characters"].items():
                            chars[name] = {
                                "name": name,
                                "description": c.get("description", ""),
                                "backstory": c.get("backstory", ""),
                                "traitPhrase": ", ".join(c.get("personality", [])) if isinstance(c.get("personality"), list) else c.get("personality", ""),
                                "bio": c.get("first_person_bio", c.get("description", "")),
                                "role": c.get("role", c.get("daily_plan_req", "")),
                                "age": str(c.get("age", "")),
                            }
                            # Create GenerativeAgent from loaded data
                            ga = GenerativeAgent(scratch={
                                "name": name,
                                "age": str(c.get("age", "")),
                                "innate": c.get("innate", ", ".join(c.get("personality", [])) if isinstance(c.get("personality"), list) else ""),
                                "currently": c.get("currently", ""),
                                "first_person_bio": c.get("first_person_bio", c.get("description", "")),
                                "backstory": c.get("backstory", c.get("learned", "")),
                            })
                            st.session_state["generative_agents"][name] = ga.to_dict()

                        # Also restore full agent state if exported previously
                        if "agents" in data:
                            for name, agent_data in data["agents"].items():
                                st.session_state["generative_agents"][name] = agent_data

                        st.session_state["characters"] = chars
                        st.session_state["active_character"] = list(chars.keys())[0]
                        if world["regions"]:
                            st.session_state["active_region"] = list(world["regions"].keys())[0]
                        st.session_state["phase"] = "adventure"
                    else:
                        st.session_state["phase"] = "character"
                    st.rerun()
                except Exception as e:
                    st.error(f"Invalid JSON: {e}")


# ═════════════════════════════════════════════════════════════
# PHASE: WORLD BUILDER
# ═════════════════════════════════════════════════════════════
def render_world_builder():
    st.markdown("## ⬡ World Creation")
    draft = st.session_state["world_draft"]
    step = st.session_state["world_step"]

    if step == "setup":
        st.markdown("*Define the setting for your adventure.*")

        c1, c2 = st.columns([3, 1])
        with c2:
            if st.button("✧ Auto-Generate", use_container_width=True):
                with st.spinner("Forging reality from the void..."):
                    result = generate_json(
                        SYSTEM_PROMPT_GENERATOR,
                        world_gen_prompt(),
                    )
                    if result:
                        draft["game_name"] = result.get("game_name", "")
                        draft["world_name"] = result.get("world_name", "")
                        draft["description"] = result.get("description", result.get("world_description", ""))
                        st.session_state["world_draft"] = draft
                        st.rerun()

        draft["game_name"] = st.text_input("Game / Setting Name", value=draft["game_name"],
                                           placeholder="e.g. The Elder Scrolls")
        draft["world_name"] = st.text_input("World Name", value=draft["world_name"],
                                            placeholder="e.g. Tamriel")
        draft["description"] = st.text_area("World Description", value=draft["description"],
                                            placeholder="Describe the world, its history, and atmosphere...",
                                            height=120)

        if st.button("Continue to Regions →", type="primary", disabled=not draft["world_name"].strip()):
            st.session_state["world_draft"] = draft
            st.session_state["world_step"] = "regions"
            st.rerun()

    elif step == "regions":
        st.markdown(f"### Regions of {draft['world_name']}")

        # Show existing regions
        if draft["regions"]:
            tags_html = " ".join(
                f'<span class="region-tag">{name}</span>' for name in draft["regions"]
            )
            st.markdown(tags_html, unsafe_allow_html=True)
            st.markdown("")

        # Add region form
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            region_name = st.text_input("Region Name", placeholder="e.g. Whiterun Hold",
                                        key="new_region_name", label_visibility="collapsed")
        with col2:
            add_manual = st.button("+ Add", use_container_width=True)
        with col3:
            gen_region = st.button("✧ Generate", use_container_width=True)

        region_desc = st.text_area("Region Description", placeholder="Describe the region...",
                                   key="new_region_desc", height=80, label_visibility="collapsed")

        if add_manual and region_name.strip():
            draft["regions"][region_name.strip()] = {
                "name": region_name.strip(),
                "description": region_desc,
            }
            st.session_state["world_draft"] = draft
            st.rerun()

        if gen_region:
            with st.spinner("Discovering new lands..."):
                result = generate_json(
                    SYSTEM_PROMPT_GENERATOR,
                    region_gen_prompt(draft["world_name"], draft["description"]),
                )
                if result:
                    name = result.get("name", "Unknown Region")
                    draft["regions"][name] = {
                        "name": name,
                        "description": result.get("description", ""),
                    }
                    st.session_state["world_draft"] = draft
                    st.rerun()

        st.markdown("---")

        c1, c2 = st.columns(2)
        with c1:
            if st.button("← Back"):
                st.session_state["world_step"] = "setup"
                st.rerun()
        with c2:
            if st.button("Continue to Characters →", type="primary",
                         disabled=len(draft["regions"]) == 0):
                # Finalize world
                st.session_state["world"] = {
                    "game_name": draft["game_name"],
                    "world_name": draft["world_name"],
                    "description": draft["description"],
                    "regions": draft["regions"],
                }
                st.session_state["phase"] = "character"
                st.rerun()


# ═════════════════════════════════════════════════════════════
# PHASE: CHARACTER CREATOR
# ═════════════════════════════════════════════════════════════
def render_character_creator():
    world = st.session_state["world"]
    st.markdown("## ⬡ Character Forge")

    c1, c2 = st.columns([3, 1])
    with c2:
        if st.button("✧ Auto-Generate", use_container_width=True):
            with st.spinner("Summoning a soul from the aether..."):
                result = generate_json(
                    SYSTEM_PROMPT_GENERATOR,
                    character_gen_prompt(world["world_name"], world["description"]),
                )
                if result:
                    basic = st.session_state["char_basic"]
                    for key in ["name", "pronouns", "age", "role", "race", "appearance", "backstory"]:
                        if key in result:
                            basic[key] = result[key]
                    st.session_state["char_basic"] = basic
                    add = st.session_state["char_additional"]
                    for key in ["skills", "goals", "flaws"]:
                        if key in result:
                            add[key] = result[key]
                    st.session_state["char_additional"] = add
                    st.rerun()

    # Main content + sidebar preview
    main_col, preview_col = st.columns([3, 2])

    with main_col:
        tab_basic, tab_details, tab_personality = st.tabs(["Basic", "Details", "Personality"])

        with tab_basic:
            st.markdown("*Write descriptions in first person.*")
            basic = st.session_state["char_basic"]

            c1, c2 = st.columns(2)
            with c1:
                basic["name"] = st.text_input("Name", value=basic["name"], placeholder="e.g. Mira Stonehaven")
            with c2:
                basic["pronouns"] = st.text_input("Pronouns", value=basic["pronouns"], placeholder="She/Her")

            c1, c2, c3 = st.columns(3)
            with c1:
                basic["age"] = st.text_input("Age", value=basic["age"], placeholder="47")
            with c2:
                basic["role"] = st.text_input("Role", value=basic["role"], placeholder="Court Wizard")
            with c3:
                basic["race"] = st.text_input("Race", value=basic["race"], placeholder="Human")

            basic["appearance"] = st.text_area("Appearance (1st person)", value=basic["appearance"],
                placeholder="I have a weathered face and a long grey beard...", height=100)
            basic["backstory"] = st.text_area("Backstory (1st person)", value=basic["backstory"],
                placeholder="I have defended my village for thirty years...", height=120)

            st.session_state["char_basic"] = basic

        with tab_details:
            st.markdown("*Additional character details — all optional.*")
            add = st.session_state["char_additional"]

            add["relationships"] = st.text_area("Relationships", value=add["relationships"],
                placeholder="I have one trusted friend, the village elder...", height=80)
            add["skills"] = st.text_area("Special Skills", value=add["skills"],
                placeholder="I am an expert swordsman...", height=70)
            add["opinions"] = st.text_area("Opinions & Beliefs", value=add["opinions"],
                placeholder="I believe honour is more important than victory...", height=70)

            c1, c2 = st.columns(2)
            with c1:
                add["loves"] = st.text_area("Loves", value=add["loves"],
                    placeholder="I love the silence of snowfall...", height=80)
            with c2:
                add["hates"] = st.text_area("Hates", value=add["hates"],
                    placeholder="I hate dishonesty...", height=80)

            add["hobbies"] = st.text_area("Hobbies", value=add["hobbies"],
                placeholder="I carve small wooden figures...", height=70)
            add["goals"] = st.text_area("Goals", value=add["goals"],
                placeholder="I want to ensure no child grows up without a home...", height=70)
            add["flaws"] = st.text_area("Flaws", value=add["flaws"],
                placeholder="I struggle to ask for help...", height=70)

            st.session_state["char_additional"] = add

        with tab_personality:
            st.markdown("*Rate where your character falls on each trait (1 = left extreme, 9 = right extreme).*")
            personality = st.session_state["char_personality"]

            for trait, items in personality.items():
                with st.expander(f"**{trait}**", expanded=False):
                    for i, item in enumerate(items):
                        st.markdown(f"<span style='color:#8a8678;font-size:13px;'>"
                                    f"{item['low']}  ←→  {item['high']}</span>",
                                    unsafe_allow_html=True)
                        item["rating"] = st.slider(
                            f"{item['low']} vs {item['high']}",
                            min_value=1, max_value=9, value=item["rating"],
                            key=f"trait_{trait}_{i}",
                            label_visibility="collapsed",
                        )

            st.session_state["char_personality"] = personality

    with preview_col:
        # Radar chart
        medians = compute_medians(st.session_state["char_personality"])
        fig = build_radar_chart(medians)
        st.plotly_chart(fig, use_container_width=True)

        # Character preview
        basic = st.session_state["char_basic"]
        st.markdown(f"""
        <div class="roster-card">
            <div style="font-size:28px;color:#c9a84c;">⚔</div>
            <h4>{basic['name'] or '—'}</h4>
            <p>{basic['role'] or '—'} · {basic['race'] or '—'} · Age {basic['age'] or '—'}</p>
        </div>
        """, unsafe_allow_html=True)

    # Bottom action buttons
    st.markdown("---")
    c1, c2 = st.columns([1, 1])
    with c2:
        if st.button("Create Character & Continue →", type="primary",
                      disabled=not st.session_state["char_basic"]["name"].strip()):
            basic = st.session_state["char_basic"]
            add = st.session_state["char_additional"]
            personality = st.session_state["char_personality"]
            trait_phrase = build_trait_phrase(personality)
            bio_parts = [
                f"My name is {basic['name']}.",
                f"I am a {basic['age']}-year-old {basic['race']} {basic['role']}." if basic['age'] else "",
                basic["backstory"],
            ]
            bio = " ".join(p for p in bio_parts if p)

            char = {
                **basic,
                **add,
                "description": f"{basic['appearance']} {basic['role']}.",
                "traitPhrase": trait_phrase,
                "bio": bio,
                "traitMedians": compute_medians(personality),
                "personality_raw": personality,
            }

            st.session_state["characters"][char["name"]] = char

            # Create a GenerativeAgent with full memory architecture
            ga = GenerativeAgent(scratch={
                "name": char["name"],
                "age": char.get("age", ""),
                "innate": char.get("traitPhrase", ""),
                "currently": f"a {char.get('role', 'person')} in {st.session_state['world']['world_name']}",
                "first_person_bio": char.get("bio", ""),
                "backstory": char.get("backstory", ""),
                "learned": ". ".join(filter(None, [
                    char.get("skills", ""),
                    char.get("relationships", ""),
                    char.get("opinions", ""),
                ])),
            })
            st.session_state["generative_agents"][char["name"]] = ga.to_dict()

            # Reset char creation fields for potential next character
            st.session_state["char_basic"] = {
                "name": "", "pronouns": "", "age": "", "role": "",
                "race": "", "appearance": "", "backstory": "",
            }
            st.session_state["char_additional"] = {
                "relationships": "", "skills": "", "opinions": "",
                "loves": "", "hates": "", "hobbies": "", "goals": "", "flaws": "",
            }
            p = {}
            for trait, pairs in TRAIT_DATA.items():
                p[trait] = [{"low": l, "high": h, "rating": 5} for l, h in pairs]
            st.session_state["char_personality"] = p

            st.session_state["phase"] = "roster"
            st.rerun()


# ═════════════════════════════════════════════════════════════
# PHASE: CHARACTER ROSTER
# ═════════════════════════════════════════════════════════════
def render_roster():
    st.markdown("## ⬡ Character Roster")
    chars = st.session_state["characters"]

    cols = st.columns(min(len(chars), 4))
    for i, (name, char) in enumerate(chars.items()):
        with cols[i % len(cols)]:
            st.markdown(f"""
            <div class="roster-card">
                <div style="font-size:28px;color:#c9a84c;">⚔</div>
                <h4>{name}</h4>
                <p>{char.get('role', char.get('description','')[:50])}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("+ Add Another Character", use_container_width=True):
            st.session_state["phase"] = "character"
            st.rerun()
    with c2:
        if st.button("Begin Adventure →", type="primary", use_container_width=True):
            st.session_state["active_character"] = list(chars.keys())[0]
            world = st.session_state["world"]
            if world["regions"]:
                st.session_state["active_region"] = list(world["regions"].keys())[0]
            st.session_state["phase"] = "adventure"
            st.rerun()


# ═════════════════════════════════════════════════════════════
# PHASE: ADVENTURE (AI Dungeon-style Chat)
# ═════════════════════════════════════════════════════════════
def render_adventure():
    world = st.session_state["world"]
    characters = st.session_state["characters"]
    active_char = st.session_state["active_character"]
    active_region = st.session_state["active_region"]

    # ── LOAD / CREATE GENERATIVE AGENT ──
    def _get_agent(name: str) -> GenerativeAgent:
        """Load GenerativeAgent from session state, or create one."""
        ga_store = st.session_state["generative_agents"]
        if name in ga_store:
            return GenerativeAgent.from_dict(ga_store[name])
        # Fallback: create from character data
        char = characters[name]
        ga = GenerativeAgent(scratch={
            "name": name,
            "age": char.get("age", ""),
            "innate": char.get("traitPhrase", ""),
            "currently": f"a {char.get('role', 'person')} in {world['world_name']}",
            "first_person_bio": char.get("bio", ""),
            "backstory": char.get("backstory", ""),
        })
        ga_store[name] = ga.to_dict()
        return ga

    def _save_agent(name: str, ga: GenerativeAgent):
        """Persist GenerativeAgent back to session state."""
        st.session_state["generative_agents"][name] = ga.to_dict()

    # ── SIDEBAR ──
    with st.sidebar:
        st.markdown("### ⚔ Adventure")
        pinfo = get_provider_info()
        st.caption(f"🤖 {pinfo['provider']} · `{pinfo['model']}`")
        st.markdown("---")

        st.markdown("##### Characters")
        new_char = st.radio(
            "Talk to:", list(characters.keys()),
            index=list(characters.keys()).index(active_char) if active_char in characters else 0,
            key="sidebar_char", label_visibility="collapsed",
        )

        st.markdown("##### Regions")
        region_names = list(world["regions"].keys()) if world["regions"] else ["Unknown"]
        new_region = st.radio(
            "Region:", region_names,
            index=region_names.index(active_region) if active_region in region_names else 0,
            key="sidebar_region", label_visibility="collapsed",
        )

        # Handle character/region switches
        if new_char != active_char or new_region != active_region:
            st.session_state["active_character"] = new_char
            st.session_state["active_region"] = new_region
            st.session_state["chat_messages"] = []
            st.session_state["chat_initialized"] = False
            st.rerun()

        st.markdown("---")

        # ── Memory Inspector ──
        with st.expander("🧠 Memory Stream", expanded=False):
            ga = _get_agent(active_char)
            mem_count = len(ga.memory.memories)
            obs_count = sum(1 for m in ga.memory.memories if m.memory_type == "observation")
            ref_count = sum(1 for m in ga.memory.memories if m.memory_type == "reflection")
            plan_count = sum(1 for m in ga.memory.memories if m.memory_type == "plan")

            st.markdown(f"""
            **{active_char}'s Memory**
            - Total memories: **{mem_count}**
            - Observations: {obs_count}
            - Reflections: {ref_count}
            - Plans: {plan_count}
            - Importance accumulated: {ga.memory._importance_accumulator}/{ga.memory.reflection_threshold}
            """)

            if mem_count > 0:
                st.markdown("**Recent memories:**")
                for m in ga.memory.memories[-8:]:
                    icon = "👁" if m.memory_type == "observation" else "💭" if m.memory_type == "reflection" else "📋"
                    st.markdown(f"{icon} *{m.content[:80]}{'...' if len(m.content) > 80 else ''}*")

        st.markdown("---")

        if st.button("💾 Save Chat Log", use_container_width=True):
            if st.session_state["chat_messages"]:
                path = save_chat_log(st.session_state["chat_messages"], world["world_name"])
                st.success(f"Saved to {path}")

        if st.button("📥 Export World + Agents", use_container_width=True):
            export = {**world, "characters": {}, "agents": {}}
            for name, c in characters.items():
                export["characters"][name] = {
                    "name": name,
                    "description": c.get("description", ""),
                    "backstory": c.get("backstory", ""),
                    "personality": list(c.get("traitMedians", {}).values()) if "traitMedians" in c else [],
                }
            # Include full agent memory state
            for name, ga_data in st.session_state["generative_agents"].items():
                export["agents"][name] = ga_data
            path = save_world_file(export)
            st.success(f"Saved to {path}")

        if st.button("🔄 Force Reflection", use_container_width=True):
            ga = _get_agent(active_char)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
            with st.spinner(f"{active_char} is reflecting..."):
                reflections = ga.memory.generate_reflections(active_char, timestamp)
            _save_agent(active_char, ga)
            if reflections:
                st.success(f"Generated {len(reflections)} reflection(s)")
            else:
                st.info("No reflections generated")

    # ── HEADER ──
    character = characters[active_char]
    region = world["regions"].get(active_region, {})

    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:12px;padding:8px 0 16px;
                border-bottom:1px solid #2a2a3a;margin-bottom:16px;">
        <span style="font-family:'Cinzel',serif;font-size:18px;color:#f0ece0;letter-spacing:1px;">
            {active_char}
        </span>
        <span style="font-size:14px;color:#8a8678;"> — {active_region}</span>
        <span style="margin-left:auto;color:#4a7c59;font-size:10px;">●</span>
    </div>
    """, unsafe_allow_html=True)

    # ── CHAT HISTORY ──
    chat_container = st.container()
    with chat_container:
        if not st.session_state["chat_initialized"] and not st.session_state["chat_messages"]:
            st.markdown("""
            <div style="text-align:center;padding:80px 20px;color:#8a8678;">
                <div style="font-size:48px;color:#c9a84c;opacity:0.4;margin-bottom:16px;">⚔</div>
                <p>Ready to begin your adventure?</p>
            </div>
            """, unsafe_allow_html=True)

            if st.button("Begin Adventure", type="primary"):
                with st.spinner("Entering the world..."):
                    ga = _get_agent(active_char)
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

                    # Add environmental observation
                    ga.perceive(
                        f"{active_char} is in {active_region}. {region.get('description', '')}",
                        timestamp,
                    )

                    # Build world context
                    world_context = (
                        f"World: {world['world_name']} — {world['description']}\n"
                        f"Region: {active_region} — {region.get('description', '')}"
                    )

                    # Generate initial greeting using memory-aware method
                    reply = ga.generate_initial_greeting(world_context, timestamp)

                    _save_agent(active_char, ga)

                    st.session_state["chat_messages"] = [
                        {"role": "narrator",
                         "content": f"You find yourself in {active_region}. Before you stands {active_char}."},
                        {"role": "npc", "content": reply, "speaker": active_char},
                    ]
                    st.session_state["chat_initialized"] = True
                    st.rerun()
        else:
            for msg in st.session_state["chat_messages"]:
                render_message(msg)

    # ── INPUT BAR ──
    if st.session_state["chat_initialized"]:
        with st.container():
            user_input = st.chat_input("▸ What do you do?")

            if user_input:
                # Add player message to display
                st.session_state["chat_messages"].append(
                    {"role": "player", "content": user_input}
                )

                # Load agent with full memory
                ga = _get_agent(active_char)
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

                # Build world context
                world_context = (
                    f"World: {world['world_name']} — {world['description']}\n"
                    f"Region: {active_region} — {region.get('description', '')}"
                )

                # Build conversation history for the agent
                conversation_history = []
                for m in st.session_state["chat_messages"]:
                    if m["role"] == "player":
                        conversation_history.append({"role": "user", "content": m["content"]})
                    elif m["role"] == "npc":
                        conversation_history.append({"role": "assistant", "content": m["content"]})

                with st.spinner(f"{active_char} is thinking..."):
                    # Use the GenerativeAgent's memory-aware chat_response
                    reply = ga.chat_response(
                        user_message=user_input,
                        world_context=world_context,
                        conversation_history=conversation_history,
                        timestamp=timestamp,
                    )

                # Persist agent state (memories now include this interaction)
                _save_agent(active_char, ga)

                st.session_state["chat_messages"].append(
                    {"role": "npc", "content": reply, "speaker": active_char}
                )

                # Show reflection notification if one was triggered
                if ga.memory.should_reflect():
                    st.toast(f"💭 {active_char} is forming deeper thoughts...")

                st.rerun()


# ═════════════════════════════════════════════════════════════
# ROUTER
# ═════════════════════════════════════════════════════════════
phase = st.session_state["phase"]

if phase == "title":
    render_title()
elif phase == "world":
    render_world_builder()
elif phase == "character":
    render_character_creator()
elif phase == "roster":
    render_roster()
elif phase == "adventure":
    render_adventure()
