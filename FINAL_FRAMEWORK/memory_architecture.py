"""
Generative Agent Memory Architecture
─────────────────────────────────────
Implements the three core components from Park et al. (2023)
"Generative Agents: Interactive Simulacra of Human Behavior":

  1. Memory Stream  — timestamped observations, reflections, and plans
  2. Retrieval      — recency × importance × relevance scoring
  3. Reflection     — periodic synthesis of memories into higher-level insights
  4. Planning       — day plans decomposed into hourly/minute-level actions

This module is designed to plug into the Streamlit NPC app. It stores
everything as plain Python dicts/lists so it can be serialized to JSON.

Compatible with the Smallville repo's persona data format:
  personas/<Name>/bootstrap_memory/scratch.json
"""

import math
import json
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

from llm_helper import call_llm, generate_json


# ─────────────────────────────────────────────────────────────
# MEMORY OBJECT
# ─────────────────────────────────────────────────────────────
class MemoryObject:
    """
    A single entry in the memory stream.
    Mirrors the Smallville repo's ConceptNode / memory_structures.

    Fields:
        content:        Natural language description of the memory
        created:        Simulation timestamp when memory was created
        last_accessed:  Simulation timestamp when memory was last retrieved
        importance:     Integer 1-10 (mundane=1, poignant=10)
        memory_type:    "observation" | "reflection" | "plan"
        depth:          0 for observations, higher for reflections-of-reflections
        evidence:       List of indices into memory_stream that support this memory
        embedding:      Optional cached embedding vector for relevance scoring
    """
    def __init__(
        self,
        content: str,
        created: str,
        importance: int = 5,
        memory_type: str = "observation",
        depth: int = 0,
        evidence: list = None,
    ):
        self.content = content
        self.created = created
        self.last_accessed = created
        self.importance = importance
        self.memory_type = memory_type
        self.depth = depth
        self.evidence = evidence or []
        self.embedding = None  # populated lazily

    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "created": self.created,
            "last_accessed": self.last_accessed,
            "importance": self.importance,
            "memory_type": self.memory_type,
            "depth": self.depth,
            "evidence": self.evidence,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MemoryObject":
        obj = cls(
            content=d["content"],
            created=d["created"],
            importance=d.get("importance", 5),
            memory_type=d.get("memory_type", "observation"),
            depth=d.get("depth", 0),
            evidence=d.get("evidence", []),
        )
        obj.last_accessed = d.get("last_accessed", d["created"])
        return obj


# ─────────────────────────────────────────────────────────────
# MEMORY STREAM
# ─────────────────────────────────────────────────────────────
class MemoryStream:
    """
    The central memory database for a generative agent.
    Stores observations, reflections, and plans as MemoryObjects.

    Implements the retrieval function from Section 4.1 of the paper:
      score = α_recency · recency + α_importance · importance + α_relevance · relevance
    """

    def __init__(self, decay_factor: float = 0.995):
        self.memories: list[MemoryObject] = []
        self.decay_factor = decay_factor
        # Weights for the retrieval scoring function
        self.alpha_recency = 1.0
        self.alpha_importance = 1.0
        self.alpha_relevance = 1.0
        # Track cumulative importance for reflection triggering
        self._importance_accumulator = 0
        self.reflection_threshold = 150  # as in the paper

    def add(self, memory: MemoryObject) -> int:
        """Add a memory and return its index."""
        idx = len(self.memories)
        self.memories.append(memory)
        self._importance_accumulator += memory.importance
        return idx

    def add_observation(self, content: str, timestamp: str, importance: int = None) -> int:
        """Add an observation. Auto-scores importance if not provided."""
        if importance is None:
            importance = self._score_importance(content)
        mem = MemoryObject(
            content=content,
            created=timestamp,
            importance=importance,
            memory_type="observation",
        )
        return self.add(mem)

    def add_plan(self, content: str, timestamp: str) -> int:
        """Add a plan entry to the memory stream."""
        mem = MemoryObject(
            content=content,
            created=timestamp,
            importance=5,
            memory_type="plan",
        )
        return self.add(mem)

    def add_reflection(self, content: str, timestamp: str, evidence: list[int] = None) -> int:
        """Add a reflection (higher-level inference)."""
        # Determine depth: max depth of evidence + 1
        depth = 0
        if evidence:
            depth = max(self.memories[i].depth for i in evidence if i < len(self.memories)) + 1
        mem = MemoryObject(
            content=content,
            created=timestamp,
            importance=8,  # reflections are inherently important
            memory_type="reflection",
            depth=depth,
            evidence=evidence or [],
        )
        return self.add(mem)

    # ─── RETRIEVAL (Section 4.1) ────────────────────────────

    def retrieve(self, query: str, timestamp: str, top_k: int = 10) -> list[tuple[int, MemoryObject, float]]:
        """
        Retrieve the top-k most relevant memories given a query.

        Returns list of (index, MemoryObject, score) tuples, sorted by score descending.

        Scoring formula (from paper):
          score = α_recency · recency + α_importance · importance + α_relevance · relevance

        - Recency: exponential decay based on hours since last access
        - Importance: the memory's importance score (1-10), normalized to [0,1]
        - Relevance: keyword overlap heuristic (or embedding cosine similarity)
        """
        if not self.memories:
            return []

        scores = []
        query_words = set(query.lower().split())

        for i, mem in enumerate(self.memories):
            recency = self._recency_score(mem, timestamp)
            importance = mem.importance / 10.0
            relevance = self._relevance_score(mem, query_words)

            score = (
                self.alpha_recency * recency
                + self.alpha_importance * importance
                + self.alpha_relevance * relevance
            )
            scores.append((i, mem, score))

        # Sort by score descending
        scores.sort(key=lambda x: x[2], reverse=True)

        # Mark retrieved memories as accessed
        for i, mem, _ in scores[:top_k]:
            mem.last_accessed = timestamp

        return scores[:top_k]

    def _recency_score(self, mem: MemoryObject, current_time: str) -> float:
        """Exponential decay based on hours since last access."""
        try:
            current = datetime.strptime(current_time, "%Y-%m-%d %H:%M")
            last = datetime.strptime(mem.last_accessed, "%Y-%m-%d %H:%M")
            hours = max(0, (current - last).total_seconds() / 3600)
        except (ValueError, TypeError):
            hours = 0
        return self.decay_factor ** hours

    def _relevance_score(self, mem: MemoryObject, query_words: set) -> float:
        """
        Simple keyword overlap relevance.
        In production, replace with embedding cosine similarity.
        """
        mem_words = set(mem.content.lower().split())
        if not query_words or not mem_words:
            return 0.0
        overlap = len(query_words & mem_words)
        return min(1.0, overlap / max(1, len(query_words)))

    def _score_importance(self, content: str) -> int:
        """
        Ask the LLM to rate importance on a 1-10 scale.
        Mirrors the exact prompt from Section 4.1 of the paper.
        """
        sys = "You rate the importance of memories on a scale of 1-10. Return ONLY an integer."
        prompt = (
            "On the scale of 1 to 10, where 1 is purely mundane "
            "(e.g., brushing teeth, making bed) and 10 is extremely poignant "
            "(e.g., a break up, college acceptance), rate the likely poignancy "
            f"of the following piece of memory.\n"
            f"Memory: {content}\n"
            f"Rating:"
        )
        try:
            result = call_llm(sys, [{"role": "user", "content": prompt}])
            # Extract first integer from response
            for word in result.split():
                word = word.strip(".,!?")
                if word.isdigit():
                    return max(1, min(10, int(word)))
            return 5
        except Exception:
            return 5

    # ─── REFLECTION (Section 4.2) ───────────────────────────

    def should_reflect(self) -> bool:
        """Check if accumulated importance exceeds threshold."""
        return self._importance_accumulator >= self.reflection_threshold

    def generate_reflections(self, agent_name: str, timestamp: str) -> list[str]:
        """
        Generate higher-level reflections from recent memories.
        Follows the two-step process from Section 4.2:
          1. Generate questions from recent memories
          2. For each question, retrieve relevant memories and synthesize insights
        """
        if not self.memories:
            return []

        # Step 1: Get the 100 most recent memories
        recent = self.memories[-100:]
        recent_text = "\n".join(f"- {m.content}" for m in recent)

        sys = "You help agents reflect on their experiences. Be concise."
        q_prompt = (
            f"Here are recent statements about {agent_name}:\n{recent_text}\n\n"
            "Given only the information above, what are 3 most salient high-level "
            "questions we can answer about the subjects in the statements?"
        )

        try:
            questions_raw = call_llm(sys, [{"role": "user", "content": q_prompt}])
            questions = [q.strip().lstrip("0123456789.)- ") for q in questions_raw.strip().split("\n") if q.strip()]
        except Exception:
            return []

        reflections = []

        for question in questions[:3]:
            # Step 2: Retrieve relevant memories for this question
            retrieved = self.retrieve(question, timestamp, top_k=15)
            if not retrieved:
                continue

            evidence_indices = [idx for idx, _, _ in retrieved]
            evidence_text = "\n".join(
                f"{i+1}. {mem.content}" for i, (_, mem, _) in enumerate(retrieved)
            )

            r_prompt = (
                f"Statements about {agent_name}:\n{evidence_text}\n\n"
                "What 5 high-level insights can you infer from the above statements? "
                "(example format: insight (because of 1, 5, 3))"
            )

            try:
                insights_raw = call_llm(sys, [{"role": "user", "content": r_prompt}])
                for line in insights_raw.strip().split("\n"):
                    line = line.strip().lstrip("0123456789.)- ")
                    if line:
                        self.add_reflection(line, timestamp, evidence=evidence_indices[:5])
                        reflections.append(line)
            except Exception:
                continue

        # Reset accumulator
        self._importance_accumulator = 0
        return reflections

    # ─── SERIALIZATION ──────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "memories": [m.to_dict() for m in self.memories],
            "importance_accumulator": self._importance_accumulator,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MemoryStream":
        stream = cls()
        for md in d.get("memories", []):
            stream.memories.append(MemoryObject.from_dict(md))
        stream._importance_accumulator = d.get("importance_accumulator", 0)
        return stream

    def get_recent_summary(self, n: int = 20) -> str:
        """Get a natural language summary of the n most recent memories."""
        recent = self.memories[-n:]
        return "\n".join(f"- {m.content}" for m in recent)


# ─────────────────────────────────────────────────────────────
# PLANNING (Section 4.3)
# ─────────────────────────────────────────────────────────────
class DayPlanner:
    """
    Generates daily plans in a top-down, recursive fashion:
      1. Broad day plan (5-8 chunks)
      2. Hourly decomposition
      3. 5-15 minute action decomposition (optional)
    """

    @staticmethod
    def generate_day_plan(
        agent_name: str,
        agent_summary: str,
        previous_day_summary: str,
        current_date: str,
    ) -> list[dict]:
        """
        Generate a broad-strokes daily plan.
        Returns list of {"time": "8:00 AM", "activity": "...", "duration_min": 60}
        """
        sys = (
            "You create realistic daily schedules for characters. "
            "Return ONLY valid JSON: a list of objects with 'time', 'activity', and 'duration_min' fields. "
            "No markdown fences."
        )
        prompt = (
            f"Name: {agent_name}\n"
            f"{agent_summary}\n\n"
            f"Yesterday: {previous_day_summary}\n\n"
            f"Today is {current_date}. Create a realistic daily plan for {agent_name} "
            f"with 6-8 activities from morning to night. "
            f"Return as JSON array."
        )

        try:
            result = generate_json(sys, prompt)
            if isinstance(result, list):
                return result
            elif isinstance(result, dict) and "plan" in result:
                return result["plan"]
        except Exception:
            pass

        # Fallback plan
        return [
            {"time": "7:00 AM", "activity": "Wake up and morning routine", "duration_min": 60},
            {"time": "8:00 AM", "activity": "Have breakfast", "duration_min": 30},
            {"time": "9:00 AM", "activity": "Begin daily work", "duration_min": 180},
            {"time": "12:00 PM", "activity": "Lunch break", "duration_min": 60},
            {"time": "1:00 PM", "activity": "Afternoon work", "duration_min": 180},
            {"time": "5:00 PM", "activity": "Evening leisure", "duration_min": 120},
            {"time": "7:00 PM", "activity": "Dinner", "duration_min": 60},
            {"time": "10:00 PM", "activity": "Prepare for bed", "duration_min": 30},
        ]

    @staticmethod
    def decompose_activity(
        agent_name: str,
        activity: str,
        duration_min: int,
    ) -> list[dict]:
        """
        Decompose a broad activity into 5-15 minute sub-actions.
        """
        sys = (
            "You decompose activities into detailed sub-steps. "
            "Return ONLY valid JSON: a list of objects with 'time_offset_min' and 'action' fields. "
            "No markdown fences."
        )
        prompt = (
            f"{agent_name} is going to: {activity} (for {duration_min} minutes). "
            f"Break this into detailed 5-15 minute sub-actions. Return as JSON array."
        )

        try:
            result = generate_json(sys, prompt)
            if isinstance(result, list):
                return result
        except Exception:
            pass

        return [{"time_offset_min": 0, "action": activity}]


# ─────────────────────────────────────────────────────────────
# GENERATIVE AGENT — Full Agent with Memory Architecture
# ─────────────────────────────────────────────────────────────
class GenerativeAgent:
    """
    A complete generative agent with memory stream, reflection, and planning.

    This wraps the memory architecture with the agent's identity (scratch data)
    and provides the high-level API for the chat interface.

    Compatible with Smallville scratch.json format:
    {
        "name": "Isabella Rodriguez",
        "first_name": "Isabella",
        "last_name": "Rodriguez",
        "age": 34,
        "innate": "friendly, outgoing, hospitable",
        "learned": "running Hobbs Cafe, ...",
        "currently": "planning a Valentine's Day party",
        "lifestyle": "...",
        "living_area": "...",
        "daily_plan_req": "Isabella needs to ...",
        "first_person_bio": "I am Isabella Rodriguez. I run Hobbs Cafe..."
    }
    """

    def __init__(self, scratch: dict = None):
        self.scratch = scratch or {}
        self.memory = MemoryStream()
        self.current_plan: list[dict] = []
        self.current_action: str = ""

        # Seed initial memories from scratch data
        if scratch:
            self._seed_memories()

    def _seed_memories(self):
        """
        Initialize the memory stream with seed memories from the scratch data.
        Mirrors how the Smallville repo seeds from semicolon-delimited descriptions.
        """
        timestamp = "2023-02-13 00:00"

        # Seed from first_person_bio or description
        bio = self.scratch.get("first_person_bio", self.scratch.get("bio", ""))
        if bio:
            # Split on sentence boundaries
            for sentence in bio.replace(";", ".").split("."):
                sentence = sentence.strip()
                if sentence:
                    self.memory.add_observation(sentence, timestamp, importance=5)

        # Seed from innate traits
        innate = self.scratch.get("innate", self.scratch.get("traitPhrase", ""))
        if innate:
            self.memory.add_observation(
                f"{self.name} is {innate}", timestamp, importance=3
            )

        # Seed from backstory
        backstory = self.scratch.get("backstory", "")
        if backstory:
            for sentence in backstory.replace(";", ".").split("."):
                sentence = sentence.strip()
                if sentence:
                    self.memory.add_observation(sentence, timestamp, importance=6)

    @property
    def name(self) -> str:
        return self.scratch.get("name", self.scratch.get("first_name", "Agent"))

    def get_summary(self) -> str:
        """
        Generate the [Agent's Summary Description] used in prompts.
        Cached and refreshed periodically in the full Smallville system.
        """
        parts = [f"Name: {self.name}"]

        age = self.scratch.get("age", "")
        if age:
            parts.append(f"Age: {age}")

        innate = self.scratch.get("innate", self.scratch.get("traitPhrase", ""))
        if innate:
            parts.append(f"Personality: {innate}")

        currently = self.scratch.get("currently", "")
        if currently:
            parts.append(f"Currently: {currently}")

        bio = self.scratch.get("first_person_bio", self.scratch.get("bio", ""))
        if bio:
            parts.append(bio)

        return "\n".join(parts)

    def perceive(self, observation: str, timestamp: str):
        """
        Agent perceives something in their environment.
        The observation is added to the memory stream.
        """
        self.memory.add_observation(observation, timestamp)

    def chat_response(
        self,
        user_message: str,
        world_context: str,
        conversation_history: list[dict],
        timestamp: str,
    ) -> str:
        """
        Generate an in-character response using the full memory architecture.

        Steps:
          1. Retrieve relevant memories based on the user's message
          2. Build a context-rich prompt with retrieved memories
          3. Generate response
          4. Store the interaction as new observations
        """
        # 1. Retrieve relevant memories
        retrieved = self.memory.retrieve(user_message, timestamp, top_k=8)
        memory_context = ""
        if retrieved:
            memory_context = "Relevant memories:\n" + "\n".join(
                f"- {mem.content}" for _, mem, _ in retrieved
            )

        # 2. Build system prompt
        agent_summary = self.get_summary()
        system_prompt = (
            f"You must imitate and act as {self.name}.\n\n"
            f"{agent_summary}\n\n"
            f"{world_context}\n\n"
            f"{memory_context}\n\n"
            "Instructions:\n"
            "- Use only 2-4 sentences.\n"
            "- Write in first person and present tense.\n"
            "- Stay in character at all times.\n"
            f"- Your knowledge should only include what {self.name} would know.\n"
            "- If the user references something outside your world, respond as if unaware.\n"
            "- Draw on your memories to make responses specific and grounded."
        )

        # 3. Build messages (filter to user/assistant only for API)
        api_messages = []
        for m in conversation_history:
            if m["role"] in ("user", "assistant"):
                api_messages.append(m)
        api_messages.append({"role": "user", "content": user_message})

        # 4. Generate response
        reply = call_llm(system_prompt, api_messages)

        # 5. Store the interaction as observations
        self.memory.add_observation(
            f"A stranger said: {user_message}", timestamp, importance=4
        )
        self.memory.add_observation(
            f"{self.name} responded: {reply}", timestamp, importance=3
        )

        # 6. Check if reflection is needed
        if self.memory.should_reflect():
            self.memory.generate_reflections(self.name, timestamp)

        return reply

    def generate_initial_greeting(self, world_context: str, timestamp: str) -> str:
        """
        Generate the NPC's opening greeting when the player first approaches.
        """
        retrieved = self.memory.retrieve(
            f"{self.name} current situation", timestamp, top_k=5
        )
        memory_context = ""
        if retrieved:
            memory_context = "Your memories:\n" + "\n".join(
                f"- {mem.content}" for _, mem, _ in retrieved
            )

        agent_summary = self.get_summary()
        system_prompt = (
            f"You must imitate and act as {self.name}.\n\n"
            f"{agent_summary}\n\n"
            f"{world_context}\n\n"
            f"{memory_context}\n\n"
            "Instructions:\n"
            "- Use only 2-4 sentences.\n"
            "- Write in first person and present tense.\n"
            "- Describe your character, where you are, and what you see.\n"
            "- Stay in character at all times."
        )

        reply = call_llm(
            system_prompt,
            [{"role": "user", "content": f"{world_context}\nYour Start:"}],
        )

        self.memory.add_observation(
            f"A stranger approached {self.name}", timestamp, importance=4
        )

        return reply

    # ─── SERIALIZATION ──────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "scratch": self.scratch,
            "memory": self.memory.to_dict(),
            "current_plan": self.current_plan,
            "current_action": self.current_action,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GenerativeAgent":
        agent = cls.__new__(cls)
        agent.scratch = d.get("scratch", {})
        agent.memory = MemoryStream.from_dict(d.get("memory", {}))
        agent.current_plan = d.get("current_plan", [])
        agent.current_action = d.get("current_action", "")
        return agent

    def save(self, filepath: str):
        """Save agent state to JSON file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "GenerativeAgent":
        """Load agent state from JSON file."""
        with open(filepath, "r") as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def from_smallville_scratch(cls, scratch_path: str) -> "GenerativeAgent":
        """
        Load from a Smallville-format scratch.json file.
        Compatible with:
          personas/<Name>/bootstrap_memory/scratch.json
        """
        with open(scratch_path, "r") as f:
            scratch = json.load(f)

        # Map Smallville fields to our format
        mapped = {
            "name": scratch.get("name", scratch.get("first_name", "")),
            "age": scratch.get("age", ""),
            "innate": scratch.get("innate", ""),
            "currently": scratch.get("currently", ""),
            "first_person_bio": scratch.get("first_person_bio", ""),
            "backstory": scratch.get("learned", ""),
            "role": scratch.get("daily_plan_req", ""),
            "living_area": scratch.get("living_area", ""),
        }

        return cls(scratch=mapped)
