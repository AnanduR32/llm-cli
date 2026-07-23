# local-agent

A minimal, hand-rolled agent loop against local llama-cli. 

## What's here

- `llm_client.py` -- HTTP wrapper around llama-server's OpenAI-compatible API
- `tools.py` -- file/shell tools + their OpenAI-format schemas
- `memory.py` -- `ConversationMemory` (persisted chat history) and
  `VectorMemory` (long-term facts, searchable, embeddings optional)
- `skills.py` -- lazy-loaded skill files from `skills/*.md`, same pattern
  Claude itself uses: a short catalog up front, full content loaded on demand
- `agent.py` -- the loop: send message -> model may call tools -> execute ->
  feed results back -> repeat until plain text answer
- `config.py` -- configuration settings (endpoint URL, model name, paths)

## Setup

```bash
pip install -r requirements.txt
```

1. Confirm `config.py`' set `BASE_URL`, `EMBED_URL`, context window etc
2. Run the agent cli in workspace you want to use it in:

   ```bash
   python agent.py
   ```

3. Done

## Enabling real embedding-based memory

For real semantic search, set `config.EMBED_URL`. :

Runs a second, small llama-server instance with an embedding model (e.g. EmbeddingGemma-300M -- tiny, runs alongside main model on remaining VRAM, or even on CPU).

## Adding a skill

Drop new `.md` file in `skills/` following the format in
`skills/git_commit_style.md` (YAML-ish header + body). It'll show up
automatically in the catalog the model sees, and it can call `load_skill`
to pull in the full content when relevant.

## Adding a tool

Add an entry to `TOOL_SCHEMAS` and a matching function to `TOOL_FUNCTIONS`
in `tools.py`. That's it -- `agent.py` picks up everything in that dict
automatically.

## To do

- add a confirm-before-execute in `run_shell`
