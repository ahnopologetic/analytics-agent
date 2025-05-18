from typing import MutableSequence
import structlog
from google.adk.tools.tool_context import ToolContext
from pydantic import BaseModel
from vertexai import rag
from google.cloud.aiplatform_v1.types.vertex_rag_data import RagFile
from tracking_agent.tools.utils import check_corpus_exists

logger = structlog.get_logger(__name__)


class RagFileItem(BaseModel):
    name: str
    display_name: str
    description: str | None = None
    create_time: str | None = None
    update_time: str | None = None
    file_status: str | None = None
    gcs_source: str | None = None
    google_drive_source: str | None = None
    direct_upload_source: str | None = None
    slack_source: str | None = None
    jira_source: str | None = None
    share_point_sources: str | None = None


class ListFilesOutput(BaseModel):
    files: list[RagFileItem]
    next_page_token: str


def list_files(
    corpus_name: str,
    page_size: int,
    page_token: str,
    tool_context: ToolContext,
) -> dict:
    """
    List all files in the given corpus.

    Args:
        corpus_name (str): The name of the corpus to list files from.
        page_size (int): The number of files to return per page.
        page_token (str): The token to use for pagination.

    Returns:
        list: A list of file paths in the corpus.
    """

    if not check_corpus_exists(corpus_name, tool_context):
        return {
            "status": "error",
            "message": f"Corpus '{corpus_name}' does not exist. Please create it first using the create_corpus tool.",
        }

    next_page_token = True
    files = []
    while next_page_token:
        logger.info(
            "Listing files",
            corpus_name=corpus_name,
            files=files,
        )
        pager = rag.list_files(corpus_name, page_size, page_token)
        rag_files: MutableSequence[RagFile] = pager.rag_files
        for rag_file in rag_files:
            files.append(
                RagFileItem(
                    name=rag_file.name,
                    display_name=rag_file.display_name,
                    description=rag_file.description,
                    # create_time=rag_file.create_time,
                    # update_time=rag_file.update_time,
                ).model_dump(mode="json")
            )
        next_page_token = pager.next_page_token

    return {
        "status": "success",
        "message": "Files listed successfully",
        "files": files,
    }
