"""
Flow per turn:
  1. User types something.
  2. Agent injects relevant long-term memories (VectorMemory.recall) into context.
  3. Agent calls the model with the full tool list (file/shell tools + 
     memory tools + load_skill).
  4. If the model returns tool_calls, execute them locally,
     and feed the results back, looping until the model returns a plain text answer.
  5. Agent persists the turn to ConversationMemory, and optionally ask the model
     to save anything worth remembering long-term.
"""

import json
import config
import tools
import skills
import ui
import subprocess
import time
import sys
import atexit
from llm_client import LLMClient
from memory import ConversationMemory, VectorMemory


SYSTEM_PROMPT_TEMPLATE = """You are a local coding/dev assistant running on the user's own machine.
You have tools for reading/writing files, running shell commands, and remembering
things across sessions. Use them when they help; don't narrate that you're about to
use a tool, just use it.

{skills_block}
"""


def build_system_prompt() -> str:
    skills_block = skills.catalog_as_prompt_block()
    return SYSTEM_PROMPT_TEMPLATE.format(skills_block=skills_block)


def all_tool_schemas() -> list[dict]:
    memory_tools = [
        {
            "type": "function",
            "function": {
                "name": "remember",
                "description": "Save a fact or preference to long-term memory, to recall in future sessions.",
                "parameters": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "recall",
                "description": "Search long-term memory for relevant facts.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        },
    ]
    return tools.TOOL_SCHEMAS + memory_tools + [skills.SKILL_SCHEMA]


def execute_tool_call(call: dict, vector_memory: VectorMemory) -> str:
    name = call["function"]["name"]
    raw_args = call["function"]["arguments"] or "{}"

    try:
        args = json.loads(raw_args)
    except json.JSONDecodeError as e:
        # Most common cause: the response got cut off by max_tokens mid-string
        # (e.g. a long write_file content field).
        return (f"ERROR: your arguments for '{name}' were not valid JSON "
                f"(likely truncated -- {e}). Raw args received: {raw_args[:200]}... "
                f"Try again, possibly with shorter content or split into multiple calls.")

    if name in tools.TOOL_FUNCTIONS:
        return tools.TOOL_FUNCTIONS[name](**args)
    elif name == "remember":
        vector_memory.remember(args["text"])
        return "Saved to long-term memory."
    elif name == "recall":
        results = vector_memory.recall(args["query"])
        return "\n".join(results) if results else "No relevant memories found."
    elif name == "load_skill":
        return skills.load_skill_body(args["name"])
    else:
        return f"ERROR: unknown tool '{name}'"


def run_turn(llm: LLMClient, convo: ConversationMemory, vector_memory: VectorMemory,
             user_input: str, max_tool_iterations: int = 20) -> str:
    convo.add("user", user_input)

    messages = convo.as_list()

    last_call_signature = None

    for i in range(max_tool_iterations):
        with ui.thinking():
            response = llm.chat(messages, tools=all_tool_schemas())
        if config.DEBUG:
            ui.debug(f"[iteration {i+1}/{max_tool_iterations}] {json.dumps(response, indent=2)}")

        if response.get("tool_calls"):
            # Detect the model calling the exact same tool with the exact same
            # arguments twice in a row 
            # In that case, bail out early with a clear message
            signature = tuple(
                (c["function"]["name"], c["function"]["arguments"])
                for c in response["tool_calls"]
            )
            if signature == last_call_signature:
                return (f"(stopped: model repeated the exact same tool call "
                        f"{signature} twice in a row -- likely stuck, not making progress)")
            last_call_signature = signature

            # Model wants to use one or more tools -- run them and feed results back
            messages.append(response)
            for call in response["tool_calls"]:
                name = call["function"]["name"]
                args = call["function"]["arguments"]
                ui.tool_call(name, args)
                result = execute_tool_call(call, vector_memory)
                if name == "propose_edit":
                    try:
                        path = json.loads(args or "{}").get("path", "")
                    except json.JSONDecodeError:
                        path = ""
                    ui.diff(result, path)
                elif result.startswith("ERROR"):
                    ui.tool_error(result)
                messages.append({
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": result,
                })
            continue  # loop back so the model can react to tool results

        # Plain text answer -- we're done
        answer = response["content"]
        convo.add("assistant", answer)
        if convo.last_trim_dropped > 0:
            ui.system_msg(f"trimmed {convo.last_trim_dropped} older message(s) to stay within context budget")
        return answer

    return "(hit max tool-call iterations without a final answer -- something may be looping)"


server_processes = []

def start_local_servers():
    """Launches the embedding and LLM servers in the background."""
    ui.system_msg("Starting local llama-servers... please wait.")
    
    embed_params = [
        "llama-server",
        "-m", config.EMBEDDING_MODEL_PATH,
        "--embedding",
        "-c", "2048",
        "--port", "14556"
    ]
    
    llm_params = [
        "llama-server",
        "-m", config.LLM_MODEL_PATH,
        "-ngl", "99",
        "--n-cpu-moe", "24",
        "-c", "16000",
        "-fa", "on",
        "-ctk", "q8_0",
        "-ctv", "q8_0",
        "-t", "8",
        "-tb", "16",
        "-ub", "1024",
        "--prio", "1",
        "-np", "1",
        "--port", "14555",
        "--jinja"
    ]
    
    try:
        # Change stdout=None if you need to debug startup issues.
        embed_proc = subprocess.Popen(embed_params, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, 
                                      text=True, encoding="utf-8", errors="replace")
        server_processes.append(embed_proc)
        
        llm_proc = subprocess.Popen(llm_params, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                    text=True, encoding="utf-8", errors="replace")
        server_processes.append(llm_proc)
        
        ui.system_msg("Waiting 5 seconds for servers to initialize...")
        time.sleep(5)
        
    except FileNotFoundError:
        print("ERROR: 'llama-server' executable not found in your system PATH.", file=sys.stderr)
        cleanup_servers()
        sys.exit(1)
    except Exception as e:
        print(f"ERROR starting servers: {e}", file=sys.stderr)
        cleanup_servers()
        sys.exit(1)

def cleanup_servers():
    """Gracefully terminates background server processes."""
    if server_processes:
        ui.system_msg("Shutting down llama-servers...")
        for proc in server_processes:
            if proc.poll() is None:
                proc.terminate()
                proc.wait()
        server_processes.clear()

def main():
    # Start background instances before compiling LLM clients
    start_local_servers()
    
    try:
        llm = LLMClient()
        convo = ConversationMemory()
        vector_memory = VectorMemory(llm=llm)

        if not convo.messages:
            convo.add("system", build_system_prompt())

        ui.banner()

        while True:
            user_input = ui.user_prompt().strip()
            if user_input.lower() in ("exit", "quit"):
                break
            if user_input.lower() == "clear":
                convo.clear()
                convo.add("system", build_system_prompt())
                ui.system_msg("conversation cleared")
                continue
            if not user_input:
                continue

            answer = run_turn(llm, convo, vector_memory, user_input)
            ui.assistant_answer(answer)
            
    finally:
        # Fallback cleanup when exiting the main application loop normally
        cleanup_servers()


if __name__ == "__main__":
    main()