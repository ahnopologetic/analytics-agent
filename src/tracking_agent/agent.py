from drtail_prompt import load_prompt
from google.adk.agents import Agent
from pydantic import BaseModel

from tracking_agent.tools import (
    add_corpus_from_github,
    define_pattern,
    list_corpora,
    list_files,
    rag_query_with_semantic_search,
)
from pathlib import Path


ROOT_DIR = Path(__file__).parent.parent.parent
AGENT_DIR = ROOT_DIR / "src" / "tracking_agent"
instruction = load_prompt(AGENT_DIR / "agent.prompt.yaml")


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
        rag_query_with_semantic_search,
        add_corpus_from_github,
        list_files,
        define_pattern,
    ],
)
