import os
import logging
from fastmcp import FastMCP
from rag_knowledge_source import RAGKnowledgeSource

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

@mcp.tool
def get_contextual_answer(query: str) -> str:
    """
    Get answer from the EU AI Act RAG system
    """
    rag_system = RAGKnowledgeSource(openai_api_key=OPENAI_API_KEY, persist_dir="./eu_ai_act_index")
    response = rag_system.get_context(query)
    logger.info(f"Query: {query}\nResponse: {response}")
    return response

if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", 8000))
    print(f"Starting MCP server for EU AI Act RAG system on {host}:{port}...")
    mcp.run(transport="http", host=host, port=port, path="/mcp")
