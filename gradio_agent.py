import gradio as gr
import asyncio
import os

from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient

# Configure your MCP servers
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

MCP_SERVERS = {
   "eu_ai_act_retriever": {
        "transport": "streamable_http",
        "url": "http://localhost:8000/mcp"
    },
}

system_prompt_template = """ Based on the EU AI Act document, 
        please answer the following question based on the context provided by the "eu_ai_act_retriever" tool with specific references to 
        relevant articles, chapters, or sections when possible:
        
        Question: {question}
        context: {context}

        Please do not make up any answers. Please do not provide generic answers.
        Please do not use Internet knowledge.
        Please provide a comprehensive answer citing specific parts of the regulation.

        {chat_history}
"""
# Initialize MCP client and LangChain components
mcp_client = None
agent = None

async def initialize_agent():
    """Initialize the MCP client and LangChain agent"""
    global mcp_client, agent
    
    # Initialize MCP client with multiple servers
    mcp_client = MultiServerMCPClient(MCP_SERVERS)
    
    # Get tools from MCP servers
    tools = await mcp_client.get_tools()
    
    # Create agent
    agent = create_agent(model="openai:gpt-4.1", tools=tools, system_prompt=system_prompt_template, debug=True)

    return "Agent initialized successfully!"

async def chat(message, history):
    """Handle chat messages"""
    global agent
    
    if agent is None:
        return "Please initialize the agent first by clicking 'Initialize Agent' button."
    
    try:
        print("User message:")
        print(message)
        # Run the agent
        response = await agent.ainvoke({
            "messages": message,
            "chat_history": history
        })
        print("User Resppnse:")
        print(response)
        return response
    
    except Exception as e:
        return f"Error: {str(e)}"

async def cleanup():
    """Cleanup MCP client on shutdown"""
    global mcp_client
    if mcp_client:
        await mcp_client.__aexit__(None, None, None)

def sync_initialize():
    """Synchronous wrapper for initialization"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(initialize_agent())
    return result

def sync_chat(message, history):
    """Synchronous wrapper for chat"""
    loop = asyncio.get_event_loop()
    if loop.is_closed():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(chat(message, history))

# Create Gradio interface
with gr.Blocks(title="MCP Chatbot") as demo:
    gr.Markdown("# MCP Chatbot with LangChain and OpenAI")
    gr.Markdown("Chat with an AI assistant that has access to MCP (Model Context Protocol) servers.")
    
    with gr.Row():
        init_btn = gr.Button("Initialize Agent", variant="primary")
        status = gr.Textbox(label="Status", interactive=False)
    
    chatbot = gr.Chatbot(height=400)
    msg = gr.Textbox(
        label="Message",
        placeholder="Type your message here...",
        lines=2
    )
    
    with gr.Row():
        submit = gr.Button("Send", variant="primary")
        clear = gr.Button("Clear")
    
    gr.Markdown("""
    ### Setup Instructions:
    1. Set your OpenAI API key: `export OPENAI_API_KEY=your-key`
    2. Make sure Node.js and npx are installed for MCP servers
    3. Click "Initialize Agent" to connect to MCP servers
    4. Start chatting!
    
    ### Configured MCP Servers:
    - **filesystem**: Access to /tmp directory
    
    You can add more servers by modifying the `MCP_SERVERS` configuration.
    """)
    
    # Event handlers
    init_btn.click(sync_initialize, outputs=status)
    
    def respond(message, chat_history):
        if not message.strip():
            return chat_history, ""
        
        response = sync_chat(message, chat_history)
        chat_history.append((message, response))
        return chat_history, ""
    
    submit.click(respond, [msg, chatbot], [chatbot, msg])
    msg.submit(respond, [msg, chatbot], [chatbot, msg])
    clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    try:
        demo.launch(share=False)
    finally:
        # Cleanup on shutdown
        loop = asyncio.get_event_loop()
        if not loop.is_closed():
            loop.run_until_complete(cleanup())