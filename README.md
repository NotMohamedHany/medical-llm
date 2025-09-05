# medical-llm

## Overview

This repository contains code for building and experimenting with a medical-focused Language Model (LLM) application. It leverages various LLM providers (Groq, DeepSeek, Ollama) and the Langchain framework for agent creation, prompt management, and tool integration.  The application framework is built using FastAPI.

## Key Features & Benefits

*   **LLM Integration:** Supports multiple LLM providers including Groq, DeepSeek, and Ollama, allowing for flexibility and experimentation.
*   **Langchain Framework:** Utilizes Langchain for creating ReAct agents, managing prompts, and integrating tools.
*   **Tooling Support:** Integrates with Tavily Search for information retrieval and other potential custom tools.
*   **FastAPI Framework:** Built using FastAPI, enabling easy deployment and creation of API endpoints for interacting with the LLM application.
*   **Modular Design:**  Easy to extend with new LLM providers, tools, and functionalities.

## Prerequisites & Dependencies

Before you begin, ensure you have the following installed:

*   **Python:** Version 3.7 or higher
*   **pip:** Python package installer

## Installation & Setup Instructions

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/NotMohamedHany/medical-llm.git
    cd medical-llm
    ```

2.  **Create a Virtual Environment (Recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```

3.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up Environment Variables:**

    *   Create a `.env` file in the root directory of the project.
    *   Define the necessary environment variables in the `.env` file.  At a minimum, the Tavily API key is required, which must be named `TAVILY_API_KEY`.  Other LLM provider keys may be needed depending on the configuration. Example:

        ```
        TAVILY_API_KEY="YOUR_TAVILY_API_KEY"
        GROQ_API_KEY="YOUR_GROQ_API_KEY" # If using Groq
        ```

        **Note:** You will need to obtain API keys from the respective LLM providers if you intend to use them.

## Usage Examples & API Documentation

### Running the Application

1.  **Start the FastAPI server:**

    ```bash
    uvicorn main:app --reload
    ```

    This will start the server on `http://127.0.0.1:8000`.

2.  **Access the API:**

    The application exposes API endpoints for interacting with the LLM agent. Refer to the FastAPI documentation for accessing API endpoints.  (e.g., `http://127.0.0.1:8000/docs` for interactive API documentation).

### Example: Interacting with the Agent (Conceptual)

The `main.py` likely includes an endpoint to query the agent. A simplified example using `curl` (adjust based on the specific API endpoint defined in `main.py`):

```bash
curl -X POST -H "Content-Type: application/json" -d '{"query": "What are the common symptoms of diabetes?"}' http://127.0.0.1:8000/query
```

*Replace `/query` with the actual endpoint and adjust the JSON payload as needed.*

**Note:** Examine the `main.py` file for specific API endpoints and input/output formats.

## Configuration Options

*   **LLM Provider Selection:**  Modify the `main.py` file to choose which LLM provider (Groq, DeepSeek, Ollama) to use.  Configure the appropriate API keys in the `.env` file.
*   **Tool Configuration:** Add or modify tools used by the ReAct agent in the `main.py` file.  Customize tool parameters as needed.
*   **Prompt Customization:** Adjust the prompts used by the LLM in the `main.py` file to fine-tune the agent's behavior.
*   **Environment Variables:** The `.env` file allows configuring various settings. The minimal setting required is `TAVILY_API_KEY`. Other LLM keys might be required depending on use case.

## License Information

The licensing for this project is currently unspecified.  Please check for updates on licensing in the future.

## Acknowledgments

*   Langchain: For providing the framework for building LLM applications.
*   Groq, DeepSeek, and Ollama: For providing LLM services.
*   Tavily Search: For providing a search tool integration.
