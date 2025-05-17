"""
Tool for listing all available Vertex AI RAG corpora.
"""

from datetime import datetime

from pydantic import BaseModel
from vertexai import rag


class Corpus(BaseModel):
    resource_name: str
    display_name: str
    create_time: datetime
    update_time: datetime


class ListCorporaOut(BaseModel):
    status: str
    message: str
    corpora: list[Corpus]


def list_corpora() -> dict:
    """
    List all available Vertex AI RAG corpora.

    Returns:
        dict: A list of available corpora and status, with each corpus containing:
            - resource_name: The full resource name to use with other tools
            - display_name: The human-readable name of the corpus
            - create_time: When the corpus was created
            - update_time: When the corpus was last updated
    """
    try:
        # Get the list of corpora
        corpora = rag.list_corpora()

        # Process corpus information into a more usable format
        corpus_info = []
        for corpus in corpora:
            corpus_data: Corpus = Corpus(
                resource_name=corpus.name,
                display_name=corpus.display_name,
                create_time=corpus.create_time,
                update_time=corpus.update_time,
            )

            corpus_info.append(corpus_data)

        return ListCorporaOut(
            status="success",
            message=f"Found {len(corpus_info)} available corpora",
            corpora=corpus_info,
        ).model_dump(mode="json")
    except Exception as e:
        return ListCorporaOut(
            status="error",
            message=f"Error listing corpora: {str(e)}",
            corpora=[],
        ).model_dump(mode="json")
