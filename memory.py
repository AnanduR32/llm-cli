"""
Two kinds of memory, deliberately kept separate:

1. ConversationMemory -- the raw back-and-forth of the *current* session.
   Persisted to disk so you can resume a conversation across restarts.

2. VectorMemory -- long-term facts the agent chooses to remember, searchable
   later regardless of which conversation they came from. This is what lets
   the agent recall "you told me last week you prefer X" in a brand new session.

   If you don't wire up an embedding model (config.EMBED_URL), this degrades
   gracefully to substring/keyword search -- still useful, just less fuzzy.
"""

import json
import os
import math
import config


def _load_json(path, default):
    if not os.path.exists(path):
        return default
    with open(path, "r") as f:
        return json.load(f)


def _save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _estimate_tokens(obj) -> int:
    """
    Rough heuristic: ~4 chars per token for English text. Deliberately
    conservative (slightly overestimates), since the safe direction to be
    wrong in is trimming a bit early rather than overflowing the server.
    Works on a message dict or a plain string.
    """
    text = obj if isinstance(obj, str) else json.dumps(obj)
    return max(1, len(text) // 4)


class ConversationMemory:
    def __init__(self, path: str = config.CONVO_FILE):
        self.path = path
        self.messages: list[dict] = _load_json(path, [])
        self.last_trim_dropped = 0  # how many messages the last _trim() removed

    def add(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        self._trim()
        _save_json(self.path, self.messages)

    def add_raw(self, message: dict):
        """For appending full message dicts (e.g. assistant messages with
        tool_calls) rather than simple role+content pairs."""
        self.messages.append(message)
        self._trim()
        _save_json(self.path, self.messages)

    def _trim(self):
        system = [m for m in self.messages if m["role"] == "system"]
        rest = [m for m in self.messages if m["role"] != "system"]

        budget = config.MAX_CONTEXT_TOKENS - config.CONTEXT_RESERVE_TOKENS
        budget -= sum(_estimate_tokens(m) for m in system)

        kept = []
        used = 0
        # Walk newest -> oldest, keep whatever fits in the budget
        for m in reversed(rest):
            size = _estimate_tokens(m)
            if used + size > budget:
                break
            kept.append(m)
            used += size
        kept.reverse()

        self.last_trim_dropped = len(rest) - len(kept)
        self.messages = system + kept

    def as_list(self) -> list[dict]:
        return self.messages

    def clear(self):
        self.messages = []
        _save_json(self.path, self.messages)


class VectorMemory:
    def __init__(self, path: str = config.MEMORY_FILE, llm: "LLMClient | None" = None):
        self.path = path
        self.llm = llm
        self.entries: list[dict] = _load_json(path, [])  # [{text, embedding, meta}]

    def remember(self, text: str, meta: dict | None = None):
        embedding = self.llm.embed(text) if self.llm else None
        self.entries.append({"text": text, "embedding": embedding, "meta": meta or {}})
        _save_json(self.path, self.entries)

    def recall(self, query: str, k: int = 3) -> list[str]:
        if not self.entries:
            return []

        query_emb = self.llm.embed(query) if self.llm else None

        if query_emb is not None and all(e["embedding"] for e in self.entries):
            scored = [
                (self._cosine(query_emb, e["embedding"]), e["text"])
                for e in self.entries
            ]
        else:
            # Fallback: naive keyword overlap scoring
            q_words = set(query.lower().split())
            scored = []
            for e in self.entries:
                t_words = set(e["text"].lower().split())
                overlap = len(q_words & t_words)
                scored.append((overlap, e["text"]))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [text for score, text in scored[:k] if score > 0]

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)