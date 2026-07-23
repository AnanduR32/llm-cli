"""
Hosts the agent as a local HTTP service, so anything on your machine
(curl, a browser, a future VS Code extension) can talk to it -- with the
same memory/tools/skills as the CLI, since it reuses agent.run_turn directly.

Run:
    python server.py

Then:
    curl -X POST http://localhost:8090/chat \\
      -H "Content-Type: application/json" \\
      -d '{"message": "read config.py and summarize it"}'
"""

import threading
from flask import Flask, request, jsonify

from agent import run_turn, build_system_prompt
from llm_client import LLMClient
from memory import ConversationMemory, VectorMemory

app = Flask(__name__)

# One shared conversation/memory instance for the life of the server process.
# A lock keeps concurrent requests from corrupting shared state -- this makes
# requests serialize (one at a time), which is fine for personal single-user
# use, and matches how the underlying model can only process one request at
# a time anyway.
_lock = threading.Lock()
_llm = LLMClient()
_convo = ConversationMemory()
_vector_memory = VectorMemory(llm=_llm)

if not _convo.messages:
    _convo.add("system", build_system_prompt())


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    message = data.get("message", "").strip()
    if not message:
        return jsonify({"error": "missing 'message' field"}), 400

    with _lock:
        answer = run_turn(_llm, _convo, _vector_memory, message)

    return jsonify({"answer": answer})


@app.route("/clear", methods=["POST"])
def clear():
    with _lock:
        _convo.clear()
        _convo.add("system", build_system_prompt())
    return jsonify({"status": "conversation cleared"})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    # 127.0.0.1, not 0.0.0.0 -- this stays local-only by default.
    # Change only if you specifically want it reachable from other devices
    # on your network, and add config.API_KEY-style auth first if you do.
    app.run(host="127.0.0.1", port=14557)
