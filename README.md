# HuggingFace Document Analysis Agent

This project implements a conversational agent, named Alfred, that can analyze documents (like images) and respond to user queries about them. It leverages the power of LangChain and LangGraph to create a sophisticated, context-aware conversational experience.

## Features

*   **Interactive Chat:** Engage in a natural conversation with Alfred.
*   **Document Analysis:** Load and analyze image documents.
*   **Context-Aware:** Alfred remembers the conversation history and uses it to provide relevant responses.
*   **Tool Integration:** The agent can use tools to perform specific tasks, such as searching for information or analyzing images.
*   **Langfuse Integration:** (Optional) Monitor and debug the agent's performance using Langfuse.
* **Graceful Termination:** The agent can gracefully terminate the conversation when the task is completed.

## Getting Started

### Prerequisites

*   Python 3.9+
*   Poetry (for dependency management)

### Installation

1.  Clone the repository:

    ```bash
    git clone <repository_url>
    cd HuggingFaceDocAnalysisAgent
    ```

2.  Install dependencies using Poetry:

    ```bash
    poetry install
    ```

3.  Activate the virtual environment:

    ```bash
    poetry shell
    ```

### Usage

1.  Run the `main.py` script:

    ```bash
    python main.py
    ```

2.  Interact with Alfred in the terminal:

    *   Type your questions or commands.
    *   To load an image, type `load image <path_to_image>`.
    *   Type `exit` or `quit` to end the conversation.

### Example Interaction

🦇 Alfred is at your service, Master Wayne. Type 'exit' to leave.

🗨️ You: load image ./data/test.png 📷 Alfred: Image loaded: ./data/test.png

🗨️ You: What is the image about? 🦇 Alfred: The image appears to be a landscape with mountains and a lake.

🗨️ You: exit 🦇 Alfred: Very well, sir. Until next time.

plaintext

## Project Structure

*   `main.py`: The main entry point for the interactive chat.
*   `agent/`: Contains the agent's core logic.
    *   `agent_core.py`: Defines the LangGraph agent and its nodes.
    *   `nodes.py`: Defines the nodes for the agent.
    * `utils.py`: Defines the tools for the agent.
*   `models/`: Contains the data models.
    *   `agent_state.py`: Defines the `AgentState` for the agent.
*   `src/`: Contains the source code.
    * `langfuse_client.py`: Defines the langfuse client.
*   `data/`: Contains the data.
    * `test.png`: An example image.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a pull request.

## Contact

Your Name - your.email@example.com

## Acknowledgments

*   LangChain
*   LangGraph
*   Hugging Face
*   Langfuse