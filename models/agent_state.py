from typing import TypedDict, List, Optional, Annotated
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    """
    Represents the state of the agent during conversation.
    
    Attributes:
        input_file: Optional path to a document file (PDF/PNG)
        messages: List of messages in the conversation history
    """
    # The document provided
    input_file: Optional[str]  # Contains file path (PDF/PNG)
    messages: Annotated[List[AnyMessage], add_messages]
