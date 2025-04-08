from agent.agent_core import react_graph
from langchain_core.messages import HumanMessage
from src.langfuse_client import langfuse_handler  # Optional
import os

def run_interactive_chat():
    """
    Interactive loop to chat with Alfred.
    """
    print("🦇 Alfred is at your service, Master Wayne. Type 'exit' to leave.\n")
    
    # Initialize state
    state = {
        "input_file": None,
        "messages": []
    }

    while True:
        user_input = input("🗨️ You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("🦇 Alfred: Very well, sir. Until next time.")
            break

        # Optional: Handle image input dynamically
        if user_input.startswith("load image "):
            path = user_input.replace("load image ", "").strip()
            if os.path.exists(path):
                state["input_file"] = path
                print(f"📷 Alfred: Image loaded: {path}")
                continue
            else:
                print("⚠️ File not found. Try again.")
                continue

        # Add user message to history
        state["messages"].append(HumanMessage(content=user_input))

        try:
            result = react_graph.invoke(
                input=state,
                config={
                    "callbacks": [langfuse_handler],
                    "metadata": {"mode": "interactive"}
                }
            )

            # Show agent reply
            final_message = result["messages"][-1].content
            print(f"🦇 Alfred: {final_message}\n")

            # Update state
            state["messages"] = result["messages"]

        except Exception as e:
            print(f"❌ Error during conversation: {e}")

def main():
    run_interactive_chat()

if __name__ == "__main__":
    main()
