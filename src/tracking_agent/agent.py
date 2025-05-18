from drtail_prompt import load_prompt
from google.adk.agents import Agent
from pydantic import BaseModel

from tracking_agent.tools import (
    add_corpus_from_github,
    add_corpus_from_local_path,
    list_corpora,
    list_files,
    rag_query,
)

instruction = load_prompt("./tracking_agent/agent.prompt.yaml")


class TrackingAgentOutputItem(BaseModel):
    event_name: str
    properties: dict
    context: str
    location: str


class TrackingAgentOutput(BaseModel):
    items: list[TrackingAgentOutputItem]


root_agent = Agent(
    model="gemini-2.0-flash-001",
    name="tracking_agent",
    instruction=instruction.messages[0].content,
    tools=[
        list_corpora,
        rag_query,
        add_corpus_from_github,
        add_corpus_from_local_path,
        list_files,
    ],
)
