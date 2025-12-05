import gradio as gr
import asyncio
import os
import atexit
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
MAX_HISTORY_LENGTH = 20  # Keep only last 20 messages in memory

async def initialize_agent():
    """Initialize the MCP client and LangChain agent"""
    global mcp_client, agent

    # Initialize MCP client with multiple servers
    mcp_client = MultiServerMCPClient(MCP_SERVERS)

    # Get tools from MCP servers
    tools = await mcp_client.get_tools()

    # Create agent
    agent = create_agent(model="openai:gpt-4.1", tools=tools, system_prompt=system_prompt_template, debug=False)

    return "Agent initialized successfully!"

async def cleanup():
    """Cleanup MCP client on shutdown"""
    global mcp_client
    if mcp_client:
        await mcp_client.__aexit__(None, None, None)

def sync_initialize_agent():
    """Run the async initializer in a fresh event loop (safe on Windows)."""
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(initialize_agent())
    finally:
        loop.close()

# initialize synchronously so agent is ready for Gradio handlers
sync_initialize_agent()

# register cleanup to run on process exit
def _sync_cleanup():
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        loop.run_until_complete(cleanup())
    finally:
        loop.close()

atexit.register(_sync_cleanup)

def trim_chat_history(chat_history):
    """Keep only the most recent messages to save memory"""
    if len(chat_history) > MAX_HISTORY_LENGTH:
        # Keep the first system message + last MAX_HISTORY_LENGTH messages
        return chat_history[:1] + chat_history[-(MAX_HISTORY_LENGTH - 1):]
    return chat_history

def extract_text_from_chunk(chunk):
    """Extract text content from various chunk formats"""
    txt = None
    
    if isinstance(chunk, dict):
        # Check for 'model' key which contains the response structure
        if "model" in chunk and isinstance(chunk["model"], dict):
            model_data = chunk["model"]
            
            # Extract messages list
            if "messages" in model_data and isinstance(model_data["messages"], list):
                messages = model_data["messages"]
                
                # Iterate through messages and extract AIMessage content
                for msg in messages:
                    if hasattr(msg, "content"):
                        txt = msg.content
                        break
        
        # Fallback: Check for AIMessage object directly
        if not txt and "output" in chunk and chunk["output"]:
            output_val = chunk["output"]
            if hasattr(output_val, "content"):
                txt = output_val.content
            else:
                txt = str(output_val)
        
        # Fallback: Try other common keys
        if not txt:
            for key in ("text", "result", "final_output", "content"):
                if key in chunk and chunk[key]:
                    val = chunk[key]
                    if hasattr(val, "content"):
                        txt = val.content
                    else:
                        txt = str(val)
                    break
    
    elif isinstance(chunk, str):
        txt = chunk
    
    return txt

async def interact_with_langchain_agent(user_message, chat_history):
    """
    Gradio Chatbot expects chat_history as a list of [user, assistant] pairs.
    This function appends a new pair for the incoming user message, streams
    agent output and updates the last assistant text so Gradio receives the
    correct tuple-format history. Memory-efficient version.
    """
    global agent

    if chat_history is None:
        chat_history = []

    # Trim history to save memory
    chat_history = trim_chat_history(chat_history)

    # Add the user message as a new pair with empty assistant placeholder
    chat_history.append([user_message, ""])
    yield chat_history

    # Ensure agent is available
    if agent is None:
        await initialize_agent()

    assistant_text = ""

    # Stream the agent response; update the last assistant text progressively
    async for chunk in agent.astream({"messages": user_message}):
        
        txt = extract_text_from_chunk(chunk)

        if txt:
            # Normalize to string
            if not isinstance(txt, str):
                try:
                    txt = str(txt)
                except Exception:
                    txt = ""

            assistant_text += txt
            
            # Update the last assistant placeholder with accumulated text
            if chat_history and chat_history[-1][1] == "":
                chat_history[-1][1] = assistant_text
                yield chat_history

def sync_interact_with_langchain_agent(user_message, chat_history):
    """Synchronous wrapper for the async agent interaction that retains chat history"""
    global agent
    
    if chat_history is None:
        chat_history = []
    
    # Make a copy to avoid reference issues
    chat_history = list(chat_history)
    
    # Trim history to save memory
    chat_history = trim_chat_history(chat_history)
    
    # The user message was already added by add_user_message()
    # Just ensure the last entry has an empty string for the assistant response
    if chat_history and chat_history[-1][1] is None:
        chat_history[-1][1] = ""
    
    # Ensure agent is available
    if agent is None:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            loop.run_until_complete(initialize_agent())
        finally:
            loop.close()
    
    # Get response from async agent
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        assistant_text = loop.run_until_complete(get_agent_response(user_message))
    finally:
        loop.close()
    
    # Update last message with full response
    if chat_history and chat_history[-1][1] == "":
        chat_history[-1][1] = assistant_text
    
    return chat_history

async def get_agent_response(user_message):
    """Get full response from agent"""
    assistant_text = ""
    
    async for chunk in agent.astream({"messages": user_message}):
        txt = extract_text_from_chunk(chunk)
        
        if txt:
            if not isinstance(txt, str):
                try:
                    txt = str(txt)
                except Exception:
                    txt = ""
            
            assistant_text += txt
    
    return assistant_text

# Build Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# Chat with a LangChain Agent 🦜⛓️ and see its thoughts 💭")
    
    # Store the user message in state so it's available in the next step
    user_message_state = gr.State("")
    
    chatbot = gr.Chatbot(
        label="Agent",
        avatar_images=(
            None,
            "https://em-content.zobj.net/source/twitter/141/parrot_1f99c.png",
        ),
    )
    input_box = gr.Textbox(lines=1, label="Chat Message", placeholder="Type your question...")
    
    def add_user_message(user_message, chat_history):
        """Add user message to chat immediately for better UX"""
        if chat_history is None:
            chat_history = []
        
        chat_history = list(chat_history)
        # Only add user message, let sync_interact_with_langchain_agent handle the assistant placeholder
        chat_history.append([user_message, None])
        return chat_history, "", user_message  # Return user_message for state
    
    def get_response(user_message, chat_history):
        """Get agent response using the stored user message"""
        return sync_interact_with_langchain_agent(user_message, chat_history)
    
    # First step: Display user message immediately and clear input
    input_box.submit(
        add_user_message,
        inputs=[input_box, chatbot],
        outputs=[chatbot, input_box, user_message_state]
    ).then(
        # Second step: Get agent response using the stored user message
        get_response,
        inputs=[user_message_state, chatbot],
        outputs=[chatbot]
    )

demo.launch()