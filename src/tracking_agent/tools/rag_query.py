"""
Tool for querying Vertex AI RAG corpora and retrieving relevant information.
"""

from typing import Optional
from google.adk.tools.tool_context import ToolContext
from pydantic import BaseModel
from vertexai import rag

from tracking_agent.config import config
from tracking_agent.logger import structlog

from .utils import check_corpus_exists, get_corpus_resource_name

logger = structlog.get_logger()


class RagQueryResult(BaseModel):
    source_uri: str
    source_name: str
    text: str
    score: float


class RagQueryOut(BaseModel):
    status: str
    message: str
    query: str
    corpus_name: str
    results: list[RagQueryResult]
    results_count: int


def rag_query(
    corpus_name: str,
    query: str,
    tool_context: ToolContext,
    file_names: Optional[list[str]] = None,
) -> dict:
    """
    Query a Vertex AI RAG corpus with a user question and return relevant information.

    Args:
        corpus_name (str): The name of the corpus to query. If empty, the current corpus will be used.
                          Preferably use the resource_name from list_corpora results.
        query (str): The text query to search for in the corpus
        tool_context (ToolContext): The tool context
        file_names (list[str]): The names of the files to query. If empty, all files in the corpus will be used.
    Returns:
        dict: The query results and status
    """
    try:
        # Check if the corpus exists
        if not check_corpus_exists(corpus_name, tool_context):
            return {
                "status": "error",
                "message": f"Corpus '{corpus_name}' does not exist. Please create it first using the create_corpus tool.",
                "query": query,
                "corpus_name": corpus_name,
            }

        # Get the corpus resource name
        corpus_resource_name = get_corpus_resource_name(corpus_name)

        # Configure retrieval parameters
        rag_retrieval_config = rag.RagRetrievalConfig(
            top_k=config.top_k,
            filter=rag.Filter(vector_distance_threshold=config.distance_threshold),
        )

        logger.info("Performing retrieval query...")
        response = rag.retrieval_query(
            rag_resources=[
                rag.RagResource(
                    rag_corpus=corpus_resource_name,
                    rag_file_ids=file_names,
                )
            ],
            text=query,
            rag_retrieval_config=rag_retrieval_config,
        )

        # Process the response into a more usable format
        results = []
        if hasattr(response, "contexts") and response.contexts:
            for ctx_group in response.contexts.contexts:
                result = RagQueryResult(
                    source_uri=(
                        ctx_group.source_uri if hasattr(ctx_group, "source_uri") else ""
                    ),
                    source_name=(
                        ctx_group.source_display_name
                        if hasattr(ctx_group, "source_display_name")
                        else ""
                    ),
                    text=(ctx_group.text if hasattr(ctx_group, "text") else ""),
                    score=(ctx_group.score if hasattr(ctx_group, "score") else 0.0),
                )
                results.append(result)

        # If we didn't find any results
        if not results:
            return RagQueryOut(
                status="warning",
                message=f"No results found in corpus '{corpus_name}' for query: '{query}'",
                query=query,
                corpus_name=corpus_name,
                results=[],
                results_count=0,
            ).model_dump(mode="json")

        return RagQueryOut(
            status="success",
            message=f"Successfully queried corpus '{corpus_name}'",
            query=query,
            corpus_name=corpus_name,
            results=results,
            results_count=len(results),
        ).model_dump(mode="json")

    except Exception as e:
        error_msg = f"Error querying corpus: {str(e)}"
        logger.error(error_msg)
        return RagQueryOut(
            status="error",
            message=error_msg,
            query=query,
            corpus_name=corpus_name,
            results=[],
            results_count=0,
        ).model_dump(mode="json")
