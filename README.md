# eu-ai-act-assistant

A Retrieval-Augmented Generation (RAG) agent designed to answer questions about the EU AI Act using a local vector index and specific tools. The project consists of a frontend interface (Gradio) and a backend component providing tools via the Model Context Protocol (MCP).

## Project Structure

The project is split into two main components:

- **Frontend (`/frontend`)**:
    - **Interface**: Built with Gradio (`gradio_agent.py`).
    - **Functionality**:
        - Interacts with the backend MCP server.
        - Uses LangChain for agentic logic.
        - **Intelligent Memory**: Integrated with **Supermemory** for persistent and semantic conversation memory.
        - **RAG Evaluator**: Integrated evaluator (`rag_evaluator.py`) that scores answers for Faithfulness and Answer Relevancy using `ragas`.
    - **Configuration**: Uses `Dockerfile` to build the agent image.

- **Backend (`/backend`)**:
    - **Server**: An MCP server (`rag_mcp_server.py`) exposing retrieval tools.
    - **Index**: Contains the pre-built vector index (`eu_ai_act_index`) and the indexer script (`eu_ai_act_indexer.py`).
    - **Configuration**: Uses `Dockerfile` to build the MCP server image.

## Prerequisites

- **Docker** and **Docker Compose** installed.
- An **OpenAI API Key**.
- A **Supermemory API Key** (optional but recommended for persistent memory).

## Getting Started

1.  **Set Environment Variables**
    Ensure your API keys are available in your shell environment, or set them in the `docker-compose.yml` file.

    ```bash
    export OPENAI_API_KEY=your_openai_key_here
    export SUPERMEMORY_API_KEY=your_supermemory_key_here
    ```

    > [!TIP]
    > You can obtain a Supermemory API key from [console.supermemory.ai](https://console.supermemory.ai).

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
    - **Memory**: The agent retrieves relevant context from past conversations via Supermemory to provide personalized answers.
    - **Evaluation**: Shortly after the answer is complete, the "Evaluation Metrics" box below the chat will update with scores for the response's faithfulness and relevancy.

## Memory Management (Supermemory)

The assistant uses **Supermemory** to move beyond simple chat history. Instead of just remembering the last few messages, it:
- **Indexes Conversations**: Every interaction is stored and semantically indexed.
- **Semantic Retrieval**: Relevant context from any previous chat is retrieved based on your current question.

To disable memory, simply omit the `SUPERMEMORY_API_KEY`.

## Development

- **Frontend Code**: Located in `frontend/gradio_agent.py`.
- **Evaluator**: `frontend/rag_evaluator.py`.
- **Backend Tools**: Defined in `backend/rag_knowledge_source.py`.

## Notes

- The setup uses a bridge network so the frontend can securely communicate with the backend MCP server.
- The vector index is mounted as a volume to the backend container to ensure data persistence and accessibility.
