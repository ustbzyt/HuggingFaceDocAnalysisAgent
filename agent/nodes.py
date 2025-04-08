from langchain_core.messages import SystemMessage
from models.agent_state import AgentState
from agent.utils import agent_runnable
from langchain_core.runnables import Runnable

def assistant(state: AgentState) -> AgentState:
    """Let LangGraph handle tool use; return model output directly."""

    # If no previous messages, initialize with system prompt
    if not state['messages']:
        textual_description_of_tool = """
        extract_text(img_path: str) -> str:
        Extract text from an image file using a multimodal model.
        divide(a: int, b: int) -> float:
        Divide a and b
        """
        image = state.get("input_file", "./data/test.png")

        sys_prompt = f"""
        You are a helpful butler named Alfred serving Mr. Wayne and Batman.
        You can analyze documents and perform computations using the tools below:
        {textual_description_of_tool}
        Whenever the user asks you to extract text from an image, you MUST use the 'extract_text' tool.
        The currently loaded image is: {image}
        """

        state["messages"] = [SystemMessage(content=sys_prompt)]

    # Run the model (which may call tools)
    result = agent_runnable.invoke(state["messages"])

    # Append the result to the message history
    state["messages"].append(result)
    return state