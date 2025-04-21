import os

from fastapi import FastAPI
from pydantic import BaseModel

from llama_index.core.workflow import Context
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.llms.openai import OpenAI

# Import our custom tools from their modules
from tools import search_tool, weather_info_tool, hub_stats_tool
from retriever import guest_info_tool

os.environ["OPENAI_API_KEY"] = ""

# Initialize the Hugging Face model
llm = OpenAI(model="gpt-4o-mini")

# Create Alfred with all the tools
alfred = AgentWorkflow.from_tools_or_functions(
    [guest_info_tool, search_tool, weather_info_tool, hub_stats_tool],
    llm=llm,
)
ctx = Context(alfred)

class ChatInput(BaseModel):
    query: str

app = FastAPI()

# fastapi dev .\app.py    
@app.post("/chat_with_agent")
async def chat_with_agent(input: ChatInput):
    response = await alfred.run(input.query, ctx=ctx)
    print("🎩 Alfred's Response:")
    print(response)
    return response

