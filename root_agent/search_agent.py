from google.adk.agents import Agent
from google.adk.tools import google_search, VertexAiSearchTool 

search_agent_tool = Agent(
    name="google_search_agent",
    model="gemini-2.0-flash",  
    # model="gemini-2.5-flash",  
    description="Agent to answer questions using Google Search.",
    instruction="You are an expert researcher. You always stick to the facts.",
    # tools=[google_search, ask_vertex_retrieval]
    tools=[google_search]
)

