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

*   **Python 3.9+:**  Make sure you have Python 3.9 or a later version installed on your system. You can check your Python version by running `python --version` or `python3 --version` in your terminal.
*   **Git:** You will need Git to clone the repository. You can check your Git version by running `git --version` in your terminal. If you don't have it, you can install it by following the instructions on the official Git website: https://git-scm.com/book/en/v2/Getting-Started-Installing-Git

### Installation

1.  **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd HuggingFaceDocAnalysisAgent
    ```

    *   Replace `<repository_url>` with the actual URL of your GitHub repository.

2.  **Create a virtual environment:**

    ```bash
    python3 -m venv .venv
    ```
    * This command will create a virtual environment in the `.venv` directory.

3.  **Activate the virtual environment:**

    *   **On Windows:**

        ```bash
        .venv\Scripts\activate
        ```

    *   **On macOS and Linux:**

        ```bash
        source .venv/bin/activate
        ```

    *   You'll need to activate the virtual environment every time you want to work on the project. You should see `(.venv)` at the beginning of your terminal prompt when the virtual environment is active.

4.  **Install dependencies using pip:**

    ```bash
    pip install -r requirements.txt
    ```

    *   This command will install all the project's dependencies listed in the `requirements.txt` file.

### Environment Variables

*   **LANGFUSE_PUBLIC_KEY** (Optional): If you want to use Langfuse for monitoring, set this environment variable to your Langfuse public key.
*   **LANGFUSE_SECRET_KEY** (Optional): If you want to use Langfuse for monitoring, set this environment variable to your Langfuse secret key.

    *   You can set these environment variables in your terminal before running the project, or you can add them to your shell's configuration file (e.g., `.bashrc`, `.zshrc`).

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