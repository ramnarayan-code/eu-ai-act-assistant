import gradio as gr
import asyncio
import os
import atexit
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import ToolMessage
from rag_evaluator import RAGEvaluator
from supermemory import Supermemory

class RAGAgent:
    """Agent that interacts with EU AI Act documentation via MCP and manages memory with Supermemory."""
    
    def __init__(self):
        # Configuration
        self.mcp_url = os.getenv("MCP_SERVER_URL")
        self.user_id = "default_user"
        self.max_history = 20
        
        # Clients
        self.mcp_client = None
        self.agent = None
        self.evaluator = RAGEvaluator()
        self.memory = Supermemory(api_key=os.getenv("SUPERMEMORY_API_KEY")) if os.getenv("SUPERMEMORY_API_KEY") else None

        self.system_prompt = """ Based on the EU AI Act document, 
        please answer the following question based on the context provided by the "eu_ai_act_retriever" tool 
        with specific references to relevant articles, chapters, or sections when possible.
        
        Question: {question}
        context: {context}

        Rules:
        - Do not make up answers or provide generic ones.
        - Do not use Internet knowledge.
        - Provide a comprehensive answer citing specific parts of the regulation.
        """

    async def initialize(self):
        """Initialize MCP client and LangChain agent."""
        if self.agent: return
        
        servers = {"eu_ai_act_retriever": {"transport": "streamable_http", "url": self.mcp_url}}
        self.mcp_client = MultiServerMCPClient(servers)
        tools = await self.mcp_client.get_tools()
        
        self.agent = create_agent(
            model="openai:gpt-4.1", 
            tools=tools, 
            system_prompt=self.system_prompt, 
            debug=False
        )
        print("Agent initialized.")

    async def cleanup(self):
        """Close MCP client connections."""
        if self.mcp_client and hasattr(self.mcp_client, "close"):
            await self.mcp_client.close()

    async def get_context(self, query):
        """Fetch relevant memories from Supermemory."""
        if not self.memory: return ""
        try:
            results = self.memory.search.memories(container_tag=self.user_id, q=query, limit=5, threshold=0.7, rerank=True)
            return "\n".join(r.content for r in results) if results else ""
        except Exception as e:
            print(f"Memory error: {e}")
            return ""

    def save_memory(self, user_msg, assistant_msg):
        """Save conversation turn to Supermemory."""
        if not self.memory: return
        try:
            self.memory.add(content=f"user: {user_msg}\nassistant: {assistant_msg}", container_tag=self.user_id)
        except Exception as e:
            print(f"Save memory error: {e}")

    def parse_chunk(self, chunk):
        """Extract text and tool outputs from an agent stream chunk."""
        text, tool_calls = "", []
        
        if not isinstance(chunk, dict): return text, tool_calls
            
        # Extract messages from various possible keys in the chunk
        messages = []
        for key in ["messages", "model", "tools"]:
            data = chunk.get(key)
            if isinstance(data, dict):
                messages.extend(data.get("messages", []))
            elif isinstance(data, list):
                messages.extend(data)

        for msg in messages:
            if isinstance(msg, ToolMessage):
                tool_calls.append(msg.content)
            elif hasattr(msg, "content") and msg.content:
                text += str(msg.content)
        
        # Fallback for direct output
        if not text and "output" in chunk:
            out = chunk["output"]
            text = out.content if hasattr(out, "content") else str(out)
            
        return text, tool_calls

    async def respond(self, message, history):
        """Main interaction stream for Gradio."""
        if not self.agent: await self.initialize()
        
        # 1. Prepare messages with memory context
        context = await self.get_context(message)
        prompt_msgs = []
        if context:
            prompt_msgs.append({"role": "system", "content": f"User Context:\n{context}"})
        prompt_msgs.append({"role": "user", "content": message})

        # 2. Update UI with user message and placeholder
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": ""})
        tool_outputs = []
        yield history, tool_outputs

        # 3. Stream agent response
        assistant_text = ""
        async for chunk in self.agent.astream({"messages": prompt_msgs}):
            txt, tools = self.parse_chunk(chunk)
            tool_outputs.extend(tools)
            if txt:
                assistant_text += txt
                history[-1]["content"] = assistant_text
                yield history, tool_outputs
        
        # 4. Finalize
        self.save_memory(message, assistant_text)

    def evaluate(self, message, history, tools):
        """Run RAG evaluation on the last turn."""
        if not history or not tools: return {}
        try:
            return self.evaluator.evaluate(
                query=message, 
                response=history[-1]["content"], 
                context_array=tools
            )
        except Exception as e:
            return {"error": str(e)}

# --- Startup & Cleanup ---
rag_agent = RAGAgent()

def run_sync(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()

run_sync(rag_agent.initialize())
atexit.register(lambda: run_sync(rag_agent.cleanup()))

# --- Gradio UI ---
with gr.Blocks(title="EU AI Act Agent") as demo:
    gr.Markdown("# Chat with a LangChain Agent 🦜⛓️ and see its thoughts 💭")
    
    state_msg = gr.State("")
    state_tools = gr.State([])
    
    chatbot = gr.Chatbot(label="Agent", avatar_images=(None, "https://em-content.zobj.net/source/twitter/141/parrot_1f99c.png"))
    input_txt = gr.Textbox(label="Message", placeholder="Ask about the EU AI Act...", lines=1)
    eval_output = gr.JSON(label="RAG Evaluation Metrics")

    def user_start(txt, history):
        return "", txt # Clear input, set state_msg

    input_txt.submit(
        user_start, [input_txt, chatbot], [input_txt, state_msg]
    ).then(
        rag_agent.respond, [state_msg, chatbot], [chatbot, state_tools]
    ).then(
        rag_agent.evaluate, [state_msg, chatbot, state_tools], [eval_output]
    )

if __name__ == "__main__":
    demo.launch()
