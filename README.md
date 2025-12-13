# eu-ai-act-assistant

A Retrieval-Augmented Generation (RAG) agent designed to answer questions about the EU AI Act using a local vector index and specific tools. The project consists of a frontend interface (Gradio) and a backend component providing tools via the Model Context Protocol (MCP).

## Project Structure

The project is split into two main components:

- **Frontend (`/frontend`)**:
    - **Interface**: Built with Gradio (`gradio_agent.py`).
    - **Functionality**:
        - Interacts with the backend MCP server.
        - Uses LangChain for agentic logic.
        - Provides a chat interface with a "LangChain Agent" persona.
        - Includes an integrated **RAG Evaluator** (`rag_evaluator.py`) that scores answers for separate Faithfulness and Answer Relevancy metrics using `ragas`.
        - Supports non-blocking asynchronous evaluation.
    - **Configuration**: Uses `Dockerfile` to build the agent image.

- **Backend (`/backend`)**:
    - **Server**: An MCP server (`rag_mcp_server.py`) exposing retrieval tools.
    - **Index**: Contains the pre-built vector index (`eu_ai_act_index`) and the indexer script (`eu_ai_act_indexer.py`).
    - **Configuration**: Uses `Dockerfile` to build the MCP server image.

## Prerequisites

- **Docker** and **Docker Compose** installed.
- An **OpenAI API Key**.
- (Optional but Recommended) **LangSmith API Key** for observability.

## Getting Started

1.  **Set Environment Variables**
    Ensure your `OPENAI_API_KEY` is available in your shell environment, or set it in the `docker-compose.yml` file directly (not recommended for committed code).

    ```bash
    export OPENAI_API_KEY=your_api_key_here
    
    # Optional: Enable LangSmith Tracing
    export LANGCHAIN_TRACING_V2=true
    export LANGCHAIN_API_KEY=your_langsmith_key
    ```

2.  **Run with Docker Compose**
    Navigate to this directory and run:

    ```bash
    docker-compose up --build
    ```

    This command will:
    - Build the `mcp-server` image from the `backend` directory.
    - Build the `rag-eu-agent` image from the `frontend` directory.
    - Start both containers.

3.  **Access the Application**
    Open your browser and navigate to:
    
    [http://localhost:7860](http://localhost:7860)

4.  **Usage**
    - Type your question about the EU AI Act in the chat box.
    - The agent will use the `eu_ai_act_retriever` tool to find relevant context.
    - It will stream the answer back to you.
    - **Evaluation**: Shortly after the answer is complete, the "Evaluation Metrics" box below the chat will update with scores for the response's faithfulness and relevancy.

## Observability with LangSmith

This project is configured to use [LangSmith](https://smith.langchain.com/) for tracing and monitoring. This is extremely effective for:

- **Tracing LLM Calls**: Inspect the exact inputs and outputs of every chain step.
- **Debugging**: Quickly identify why an agent took a specific path or tool.
- **Recording Evaluations**: View the RAG evaluations (Faithfulness/Relevancy) alongside the trace runs for comprehensive analysis.

To enable it, simply set the `LANGCHAIN_TRACING_V2` and `LANGCHAIN_API_KEY` environment variables before running `docker-compose up`.

## Development

- **Frontend Code**: Located in `frontend/gradio_agent.py`.
- **Evaluator**: `frontend/rag_evaluator.py`.
- **Backend Tools**: Defined in `backend/rag_knowledge_source.py`.

## Notes

- The setup uses a bridge network so the frontend can securely communicate with the backend MCP server.
- The vector index is mounted as a volume to the backend container to ensure data persistence and accessibility.
