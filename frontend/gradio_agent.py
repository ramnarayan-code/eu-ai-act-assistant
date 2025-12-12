import gradio as gr
import asyncio
import os
import atexit
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import ToolMessage, HumanMessage
from rag_evaluator import RAGEvaluator

class RAGAgent:
    def __init__(self):
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.MCP_SERVER_URL = os.getenv("MCP_SERVER_URL")
        self.MCP_SERVERS = {
            "eu_ai_act_retriever": {
                "transport": "streamable_http",
                "url": self.MCP_SERVER_URL
            },
        }
        self.system_prompt_template = """ Based on the EU AI Act document, 
        please answer the following question based on the context provided by the "eu_ai_act_retriever" tool with specific references to 
        relevant articles, chapters, or sections when possible:
        
        Question: {question}
        context: {context}

        Please do not make up any answers. Please do not provide generic answers.
        Please do not use Internet knowledge.
        Please provide a comprehensive answer citing specific parts of the regulation.

        {chat_history}
"""
        self.MAX_HISTORY_LENGTH = 20
        self.mcp_client = None
        self.agent = None
        self.evaluator = RAGEvaluator()

    async def initialize(self):
        """Initialize the MCP client and LangChain agent"""
        # Initialize MCP client with multiple servers
        self.mcp_client = MultiServerMCPClient(self.MCP_SERVERS)

        # Get tools from MCP servers
        tools = await self.mcp_client.get_tools()

        # Create agent
        self.agent = create_agent(model="openai:gpt-4.1", tools=tools, system_prompt=self.system_prompt_template, debug=False)

        return "Agent initialized successfully!"

    async def cleanup(self):
        """Cleanup MCP client on shutdown"""
        if self.mcp_client:
            # MultiServerMCPClient no longer supports context manager
            # We assume there is a close method, or we simply don't need to exit if it manages itself.
            # However, looking at source typically helps. Let's try close() if it exists or nothing.
            # Best guess: use close()
            if hasattr(self.mcp_client, "close"):
                await self.mcp_client.close()
            elif hasattr(self.mcp_client, "__aexit__"):
                 # Fallback if the user was wrong about version or it's a different error, 
                 # but correct fix for "cannot be used as context manager" is typically removing the WITH or manual call.
                 # If it says "cannot be used", likely __aexit__ RAISES the error.
                 pass

    def trim_chat_history(self, chat_history):
        """Keep only the most recent messages to save memory"""
        if len(chat_history) > self.MAX_HISTORY_LENGTH:
            # chat_history is a list of dicts, no system message usually in gradio history unless explicitly added
            return chat_history[-(self.MAX_HISTORY_LENGTH):]
        return chat_history

    def extract_tool_messages_from_chunk(self, chunk):
        """Extract tool message content from chunks"""
        tool_contents = []
        
        if isinstance(chunk, dict):
            # Check for 'tools' key (langchain graph output often uses this)
            if "tools" in chunk and isinstance(chunk["tools"], dict):
                tools_data = chunk["tools"]
                if "messages" in tools_data and isinstance(tools_data["messages"], list):
                     for msg in tools_data["messages"]:
                        if isinstance(msg, ToolMessage):
                            tool_contents.append(msg.content)

            if "messages" in chunk and isinstance(chunk["messages"], list):
                 for msg in chunk["messages"]:
                    if isinstance(msg, ToolMessage):
                        tool_contents.append(msg.content)
            
            # Also check model -> messages path
            if "model" in chunk and isinstance(chunk["model"], dict):
                model_data = chunk["model"]
                if "messages" in model_data and isinstance(model_data["messages"], list):
                    for msg in model_data["messages"]:
                        if isinstance(msg, ToolMessage):
                            tool_contents.append(msg.content)
        
        return tool_contents

    def extract_text_from_chunk(self, chunk):
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
                        if hasattr(msg, "content") and not isinstance(msg, ToolMessage):
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

    async def interact_with_langchain_agent(self, user_message, chat_history):
        """
        Gradio Chatbot expects chat_history as a list of [user, assistant] pairs.
        This function appends a new pair for the incoming user message, streams
        agent output and updates the last assistant text so Gradio receives the
        correct tuple-format history. Memory-efficient version.
        """
        if chat_history is None:
            chat_history = []

        # Trim history to save memory
        chat_history = self.trim_chat_history(chat_history)

        # Add the user message as a new dict
        chat_history.append({"role": "user", "content": user_message})
        chat_history.append({"role": "assistant", "content": ""})
        
        # Tool outputs for this turn
        tool_outputs = []
        
        yield chat_history, tool_outputs

        # Ensure agent is available
        if self.agent is None:
            await self.initialize()

        assistant_text = ""

        # Stream the agent response; update the last assistant text progressively
        async for chunk in self.agent.astream({"messages": [{"role": "user", "content": user_message}]}):
            
            # Capture tool messages
            new_tool_outputs = self.extract_tool_messages_from_chunk(chunk)
            if new_tool_outputs:
                tool_outputs.extend(new_tool_outputs)
                
            txt = self.extract_text_from_chunk(chunk)

            if txt:
                # Normalize to string
                if not isinstance(txt, str):
                    try:
                        txt = str(txt)
                    except Exception:
                        txt = ""

                assistant_text += txt
                
                # Update the last assistant placeholder with accumulated text
                if chat_history and chat_history[-1]["role"] == "assistant":
                    chat_history[-1]["content"] = assistant_text
                    yield chat_history, tool_outputs

    async def get_agent_response(self, user_message):
        """Get full response from agent"""
        assistant_text = ""
        tool_outputs = []
        
        if self.agent is None:
            await self.initialize()

        print(f"DEBUG INPUT: type={type(user_message)}, value={user_message}")
            
        async for chunk in self.agent.astream({"messages": [{"role": "user", "content": user_message}]}):
            new_tool_outputs = self.extract_tool_messages_from_chunk(chunk)
            if new_tool_outputs:
                tool_outputs.extend(new_tool_outputs)
                
            txt = self.extract_text_from_chunk(chunk)
            
            if txt:
                if not isinstance(txt, str):
                    try:
                        txt = str(txt)
                    except Exception:
                        txt = ""
                
                assistant_text += txt
        
        return assistant_text, tool_outputs

    def sync_interact(self, user_message, chat_history):
        """Synchronous wrapper for the async agent interaction that retains chat history"""
        if chat_history is None:
            chat_history = []
        
        # Make a copy to avoid reference issues
        chat_history = list(chat_history)
        
        # Trim history to save memory
        chat_history = self.trim_chat_history(chat_history)
        
        # The user message was already added by add_user_message()
        # Just ensure the last entry has an empty string for the assistant response
        # The user message was already added by add_user_message()
        # Just ensure the last entry is an empty assistant response if not present
        if chat_history and chat_history[-1]["role"] == "user":
             chat_history.append({"role": "assistant", "content": ""})
        
        # Ensure agent is available
        if self.agent is None:
            loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self.initialize())
            finally:
                loop.close()
        
        # Get response from async agent
        tool_outputs = []
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            assistant_text, tool_outputs = loop.run_until_complete(self.get_agent_response(user_message))
        finally:
            loop.close()
        
        # Update last message with full response
        # Update last message with full response
        # Update last message with full response
        if chat_history and chat_history[-1]["role"] == "assistant":
            chat_history[-1]["content"] = assistant_text
        
        return chat_history, tool_outputs

    def evaluate_response(self, user_message, chat_history, tool_outputs):
        """Evaluate the response - intended to be run in a separate thread (Gradio def)"""
        if not tool_outputs or not chat_history:
             return {"message": "No context or history to evaluate"}
        
        
        assistant_text = chat_history[-1]["content"]
        
        try:
            results = self.evaluator.evaluate(
                query=user_message,
                response=assistant_text,
                context_array=tool_outputs
            )
            return results
        except Exception as e:
            return {"error": str(e)}

# Instantiate the agent
rag_agent = RAGAgent()

def sync_initialize_agent():
    """Run the async initializer in a fresh event loop (safe on Windows)."""
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(rag_agent.initialize())
    finally:
        loop.close()

# initialize synchronously so agent is ready for Gradio handlers
sync_initialize_agent()

# register cleanup to run on process exit
def _sync_cleanup():
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        loop.run_until_complete(rag_agent.cleanup())
    finally:
        loop.close()

atexit.register(_sync_cleanup)

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
    
    tool_outputs_state = gr.State([])
    evaluation_json = gr.JSON(label="Evaluation Metrics")

    def add_user_message(user_message, chat_history):
        """Add user message to chat immediately for better UX"""
        if chat_history is None:
            chat_history = []
        
        chat_history = list(chat_history)
        # Add user message as a dict
        chat_history.append({"role": "user", "content": user_message})
        return chat_history, "", user_message  # Return user_message for state
    
    def get_response(user_message, chat_history):
        """Get agent response using the stored user message"""
        return rag_agent.sync_interact(user_message, chat_history)

    def run_evaluation(user_message, chat_history, tool_outputs):
        """Trigger evaluation"""
        return rag_agent.evaluate_response(user_message, chat_history, tool_outputs)
    
    # First step: Display user message immediately and clear input
    input_box.submit(
        add_user_message,
        inputs=[input_box, chatbot],
        outputs=[chatbot, input_box, user_message_state]
    ).then(
        # Second step: Get agent response using the stored user message
        get_response,
        inputs=[user_message_state, chatbot],
        outputs=[chatbot, tool_outputs_state]
    ).then(
        # Third step: Evaluate (non-blocking for the generation)
        run_evaluation,
        inputs=[user_message_state, chatbot, tool_outputs_state],
        outputs=[evaluation_json]
    )

demo.launch()