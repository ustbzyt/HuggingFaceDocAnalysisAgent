import base64
from langchain.schema import HumanMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama


@tool
def divide(a: int, b: int) -> float:
    """Divide a and b - for Master Wayne's occasional calculations."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

@tool
def extract_text(img_path: str) -> str:
    """
    Extract text from an image file using a multimodal model.
    """
    try:
        with open(img_path, "rb") as image_file:
            image_bytes = image_file.read()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        message = [
            HumanMessage(
                content=[
                    {"type": "text", "text": "Extract all the text from this image. Return only the extracted text."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
                ]
            )
        ]
        response = vision_llm.invoke(message)
        return response.content.strip()
    except FileNotFoundError:
        return "Error: Image file not found"
    except PermissionError:
        return "Error: Permission denied when accessing the image file"
    except Exception as e:
        return f"Error extracting text: {str(e)}"

# Initialize models with error handling
try:
    vision_llm = ChatOllama(model="granite3.2-vision")  # Used only inside extract_text
    llm = ChatOllama(model="qwen2.5:1.5B")                # Main chat agent
except Exception as e:
    raise RuntimeError(f"Failed to initialize Ollama models: {str(e)}")

# Correct binding:
tools = [divide, extract_text]
llm_with_tools = llm.bind_tools(tools)

# This is the correct agent_runnable
agent_runnable = llm_with_tools
