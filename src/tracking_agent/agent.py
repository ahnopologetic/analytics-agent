from google.adk.agents import Agent

from tracking_agent.prompts import return_instructions_root
from tracking_agent.tools.rag_query import rag_query
from tracking_agent.tools.list_corpora import list_corpora

root_agent = Agent(
    model="gemini-2.0-flash-001",
    name="ask_rag_agent",
    instruction=return_instructions_root(),
    tools=[list_corpora, rag_query],
)
