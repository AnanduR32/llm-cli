import os

BASE_URL = "main-model-base-url"  # e.g. "http://localhost:8090"
API_KEY = None 
MODEL = "local-model"

EMBED_URL = "embedding-base-url"  # e.g. "http://localhost:14556"
EMBED_MODEL = "embedding-model"

# ---  Model Path Configurations ---
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_DIR, "models")

EMBEDDING_MODEL_PATH = os.path.join(MODELS_DIR, "model-embedding.gguf")
LLM_MODEL_PATH = os.path.join(MODELS_DIR, "model-llm.gguf")
# ------------------------------------------

MAX_HISTORY_TURNS = 20
MAX_CONTEXT_TOKENS = 16000
CONTEXT_RESERVE_TOKENS = 1500
DEBUG = False

DATA_DIR = "./data"
CONVO_FILE = f"{DATA_DIR}/conversation.json"
MEMORY_FILE = f"{DATA_DIR}/memory.json"
SKILLS_DIR = "./skills"

PROTECTED_FILES = {
    os.path.abspath(os.path.join(PROJECT_DIR, f)) for f in [
        "agent.py", "config.py", "llm_client.py", "memory.py",
        "tools.py", "skills.py", "server.py",
    ]
}