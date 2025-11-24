import os
from fastmcp import FastMCP
from eu_ai_act_rag import EUIActRAGSystem

mcp = FastMCP()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

@mcp.tool
def get_contextual_answer(query: str) -> str:
    """
    Get answer from the EU AI Act RAG system
    """
    rag_system = EUIActRAGSystem(openai_api_key=OPENAI_API_KEY)
    return rag_system.get_context(query)

if __name__ == "__main__":
    print("Starting MCP server for EU AI Act RAG system...")
    mcp.run(transport="http", host="127.0.0.1", port=8000, path="/mcp")
