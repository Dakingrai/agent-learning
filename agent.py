import os
from dotenv import load_dotenv
from anthropic import Anthropic
from langsmith import trace

load_dotenv()
client = Anthropic()

# --- ask for target directory at startup ---
TARGET_DIR = input("Enter path to target directory: ").strip()
if not os.path.isdir(TARGET_DIR):
    print(f"Error: '{TARGET_DIR}' is not a valid directory.")
    exit(1)

tools = [
    {
        "name": "list_files",
        "description": "List all files in the target directory",
        "input_schema": {"type": "object", "properties": {}}
    },
    {
        "name": "read_file",
        "description": "Read the contents of a file by name (relative to target directory)",
        "input_schema": {
            "type": "object",
            "properties": {"filename": {"type": "string"}},
            "required": ["filename"]
        }
    }
]

def list_files():
    return "\n".join(os.listdir(TARGET_DIR))

def read_file(filename):
    try:
        with open(os.path.join(TARGET_DIR, filename)) as f:
            return f.read()
    except Exception as e:
        return f"Error: {e}"

def execute_tool(name, input_dict):
    if name == "list_files":
        return list_files()
    elif name == "read_file":
        return read_file(input_dict["filename"])
    return "Unknown tool"

def agent_step(messages):
    """Runs the tool-use loop until Claude produces a final text response."""
    while True:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            tools=tools,
            messages=messages
        )

        for block in response.content:
            if block.type == "text":
                print(f"\n[CLAUDE]: {block.text}")
            elif block.type == "tool_use":
                print(f"\n[TOOL CALL]: {block.name}({block.input})")

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            break

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result = execute_tool(block.name, block.input)
                print(f"[TOOL RESULT]: {result[:300]}")
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result
                })

        messages.append({"role": "user", "content": tool_results})

    return messages

if __name__ == "__main__":
    messages = []  # persists across turns
    print(f"Agent ready. Target directory: {TARGET_DIR}")
    print("Type 'exit' to quit.\n")

    with trace(name="conversation", project_name="agent-learning") as rt:
        while True:
            user_input = input("You: ").strip()
            if user_input.lower() in ("exit", "quit"):
                break
            
            messages.append({"role": "user", "content": user_input})
            messages = agent_step(messages)