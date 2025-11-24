from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent

client = MultiServerMCPClient(
    {
        "eu_ai_act_retriever": {
            "transport": "streamable_http",
            "url": "http://localhost:8000/mcp"
        },
    }
)

async def main():
    system_prompt_template = """ Based on the EU AI Act document, 
        please answer the following question based on the context provided by the "eu_ai_act_retriever" tool with specific references to 
        relevant articles, chapters, or sections when possible:
        
        Question: {question}
        context: {context}

        Please do not make up any answers. Please do not provide generic answers.
        Please do not use Internet knowledge.
        Please provide a comprehensive answer citing specific parts of the regulation.
        """
    tools = await client.get_tools()
    print("Available Tools:", tools)
    agent = create_agent(model="openai:gpt-4.1", tools=tools, system_prompt=system_prompt_template)
    response = await agent.ainvoke({"messages":"List me top 3 prohibited AI practices?"})
    print("Agent Response:", response)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main()) 