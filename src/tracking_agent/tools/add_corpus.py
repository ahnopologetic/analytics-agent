import argparse
import structlog
from typing import Optional

from tracking_agent.ingestion.rag_corpus import ingest_repository_to_rag_corpus
from tracking_agent.ingestion.config import (
    RAGIngestConfig,
)  # For default project/location

logger = structlog.get_logger(__name__)
config = RAGIngestConfig()  # Load config for default project_id and location if needed


def add_corpus_from_github(
    github_url: str,
    corpus_display_name: Optional[str] = None,
    project_id: Optional[str] = None,
    location: Optional[str] = None,
) -> dict:
    """
    CLI tool to add a RAG corpus from a GitHub repository using the ingestion pipeline.

    Args:
        github_url (str): The URL of the GitHub repository.
        corpus_display_name (Optional[str]): Optional display name for the RAG corpus.
        project_id (Optional[str]): Google Cloud Project ID. Defaults to config.
        location (Optional[str]): Google Cloud Location. Defaults to config.

    Returns:
        dict: A dictionary containing the status, message, and corpus details if successful.
    """
    logger.info(
        "Adding corpus from GitHub via tool",
        github_url=github_url,
        corpus_display_name=corpus_display_name,
        project_id=project_id,
        location=location,
    )

    # Use provided project_id/location or fallback to config defaults
    effective_project_id = project_id if project_id else config.google_config.project_id
    effective_location = location if location else config.google_config.location

    if not github_url or not github_url.startswith("https://github.com/"):
        logger.error("Invalid GitHub URL provided.", github_url=github_url)
        return {
            "status": "error",
            "message": "Invalid GitHub URL. Must start with https://github.com/",
        }

    try:
        corpus_object, import_response = ingest_repository_to_rag_corpus(
            github_url=github_url,
            rag_corpus_display_name=corpus_display_name,
            project_id=effective_project_id,
            location=effective_location,
            # Other parameters like gcs_bucket_name, embedding_model, chunk_size/overlap
            # will use defaults from the RAGIngestConfig within ingest_repository_to_rag_corpus
        )

        if (
            corpus_object
            and import_response
            and hasattr(corpus_object, "name")
            and import_response.imported_rag_files_count > 0
        ):
            logger.info(
                "RAG corpus created and files imported successfully.",
                corpus_name=corpus_object.name,
                corpus_display_name=corpus_object.display_name,
                imported_count=import_response.imported_rag_files_count,
                failed_count=import_response.failed_rag_files_count,
            )
            return {
                "status": "success",
                "message": "RAG corpus created and files imported successfully.",
                "corpus_name": corpus_object.name,
                "corpus_display_name": corpus_object.display_name,
                "imported_files_count": import_response.imported_rag_files_count,
                "failed_files_count": import_response.failed_rag_files_count,
            }
        elif (
            corpus_object
            and import_response
            and hasattr(corpus_object, "name")
            and import_response.imported_rag_files_count == 0
            and import_response.failed_rag_files_count == 0
        ):
            logger.warn(
                "RAG corpus created, but no files were imported (possibly no supported files found or all files skipped).",
                corpus_name=corpus_object.name,
                corpus_display_name=corpus_object.display_name,
            )
            return {
                "status": "warning",
                "message": "RAG corpus created, but no files were imported. Check repository for supported file types and sizes.",
                "corpus_name": corpus_object.name,
                "corpus_display_name": corpus_object.display_name,
                "imported_files_count": 0,
                "failed_files_count": 0,
            }
        else:
            logger.error(
                "Failed to create RAG corpus or import files.",
                corpus_object=corpus_object,
                import_response=import_response,
            )
            return {
                "status": "error",
                "message": "Failed to create RAG corpus or import files. Check logs for details.",
                "corpus_details": str(corpus_object) if corpus_object else "N/A",
                "import_details": str(import_response) if import_response else "N/A",
            }

    except Exception as e:
        logger.error("Error adding corpus from GitHub", error=str(e), exc_info=True)
        return {
            "status": "error",
            "message": f"An unexpected error occurred: {str(e)}",
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Add a RAG corpus from a GitHub repository."
    )
    parser.add_argument(
        "github_url",
        type=str,
        help="The URL of the GitHub repository (e.g., https://github.com/user/repo).",
    )
    parser.add_argument(
        "--name", type=str, help="Optional display name for the RAG corpus."
    )
    parser.add_argument(
        "--project-id",
        type=str,
        help=f"Google Cloud Project ID. (default: {config.google_config.project_id})",
    )
    parser.add_argument(
        "--location",
        type=str,
        help=f"Google Cloud Location for Vertex AI. (default: {config.google_config.location})",
    )

    args = parser.parse_args()

    result = add_corpus_from_github(
        github_url=args.github_url,
        corpus_display_name=args.name,
        project_id=args.project_id,
        location=args.location,
    )

    print("Operation Result:")
    for key, value in result.items():
        print(f"  {key}: {value}")

    if result["status"] == "error":
        exit(1)
    elif result["status"] == "warning":
        exit(2)  # Different exit code for warnings
