from drtail_prompt import load_prompt
from google.adk.agents import Agent

from tracking_agent.tools import (
    add_corpus_from_github,
    add_corpus_from_local_path,
    list_corpora,
    list_files,
    rag_query,
)

instruction = load_prompt("./tracking_agent/agent.prompt.yaml")

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
