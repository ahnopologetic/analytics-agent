from google.adk.agents import Agent
from google.adk.tools.retrieval.vertex_ai_rag_retrieval import VertexAiRagRetrieval
from pydantic_settings import BaseSettings, SettingsConfigDict
from vertexai.preview import rag

from tracking_agent.prompts import return_instructions_root


class AgentConfig(BaseSettings):
    rag_corpus: str

    model_config = SettingsConfigDict(env_file=".env")


config = AgentConfig(rag_corpus="projects/1013401147098/locations/us-central1/ragCorpora/1152921504606846976")

ask_vertex_retrieval = VertexAiRagRetrieval(
    name="retrieve_rag_documentation",
    description=(
        "Use this tool to retrieve documentation and reference materials for the question from the RAG corpus,"
    ),
    rag_resources=[rag.RagResource(rag_corpus=config.rag_corpus)],
    similarity_top_k=10,
    vector_distance_threshold=0.6,
)


root_agent = Agent(
    model="gemini-2.0-flash-001",
    name="ask_rag_agent",
    instruction=return_instructions_root(),
    tools=[
        ask_vertex_retrieval,
    ],
)
