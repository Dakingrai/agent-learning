from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
import os
from dotenv import load_dotenv

load_dotenv()
TARGET_DIR = "./sample_repo"

@tool
def list_files() -> str:
    """List all files in the target directory"""
    return "\n".join(os.listdir(TARGET_DIR))

@tool
def read_file(filename: str) -> str:
    """Read the contents of a file by name (relative to target directory)"""
    try:
        with open(os.path.join(TARGET_DIR, filename)) as f:
            return f.read()
    except Exception as e:
        return f"Error: {e}"

model = init_chat_model("anthropic:claude-sonnet-4-6")
agent = create_react_agent(model, tools=[list_files, read_file])

if __name__ == "__main__":
    messages = []
    print(f"Agent ready. Target directory: {TARGET_DIR}")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            break

        messages.append({"role": "user", "content": user_input})

        before = len(messages)
        result = agent.invoke({"messages": messages})
        messages = result["messages"]
        for m in messages[before:]:
            m.pretty_print()