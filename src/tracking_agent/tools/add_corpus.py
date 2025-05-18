import uuid

import git
import structlog

from tracking_agent.ingestion.rag_corpus import (
    clone_github_repo,
    create_rag_corpus,
    get_gcs_bucket,
    import_files_to_vertex_rag,
    upload_repo_to_gcs,
)

logger = structlog.get_logger(__name__)


def add_corpus_from_github(github_url: str) -> dict:
    """
    Add a RAG corpus from a GitHub repository.

    Args:
        github_url (str): The URL of the GitHub repository.

    Returns:
        dict: A dictionary containing the status and message of the operation.
    """
    logger.info("Adding corpus from GitHub", github_url=github_url)
    if not github_url.startswith("https://github.com/"):
        return {
            "status": "error",
            "message": "Invalid GitHub URL",
        }

    repo_name = github_url.split("/")[-1]
    repo_org_and_name = github_url.split("/")[-2:]

    try:
        local_repo_path = clone_github_repo(github_url)
    except git.GitCommandError as e:
        logger.error("Error cloning repository", error=str(e))
        return {
            "status": "error",
            "message": str(e),
        }
    bucket = get_gcs_bucket()
    gcs_folder_prefix = upload_repo_to_gcs(bucket, local_repo_path)

    rag_corpus_name = f"trk-rag-corpus-{repo_name}-{uuid.uuid4()}"
    rag_corpus = create_rag_corpus(rag_corpus_name, repo_org_and_name)

    response = import_files_to_vertex_rag(rag_corpus.name, gcs_folder_prefix)

    if response.imported_rag_files_count > 0:
        return {
            "status": "success",
            "message": "RAG corpus created and files imported successfully",
        }
    else:
        return {
            "status": "error",
            "message": "Failed to create RAG corpus",
        }


def add_corpus_from_local_path(local_path: str) -> dict:
    """
    Add a RAG corpus from a local path.

    Args:
        local_path (str): The path to the local directory.

    Returns:
        dict: A dictionary containing the status and message of the operation.
    """
    bucket = get_gcs_bucket()
    gcs_folder_prefix = upload_repo_to_gcs(bucket, local_path)

    rag_corpus_name = f"trk-rag-corpus-{local_path.split('/')[-1]}-{uuid.uuid4()}"
    rag_corpus = create_rag_corpus(rag_corpus_name, local_path)

    response = import_files_to_vertex_rag(rag_corpus.name, gcs_folder_prefix)

    if response.imported_rag_files_count > 0:
        return {
            "status": "success",
            "message": "RAG corpus created and files imported successfully",
        }
    else:
        return {
            "status": "error",
            "message": "Failed to create RAG corpus",
        }
