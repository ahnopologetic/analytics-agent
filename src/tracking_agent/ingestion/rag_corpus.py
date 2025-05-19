import argparse
import tempfile
import uuid
from pathlib import Path
from typing import Any, List, Optional, Tuple

import git
import vertexai
from google.cloud import aiplatform_v1beta1, storage
from google.cloud.aiplatform_v1.types.vertex_rag_data_service import (
    ImportRagFilesResponse,
)
from vertexai.preview import rag
from vertexai.preview.rag.rag_data import (
    RagCorpus,
    TransformationConfig,
)
from vertexai.preview.rag.utils.resources import (
    ChunkingConfig,
    EmbeddingModelConfig,
    RagResource,
)

from tracking_agent.ingestion.config import config
from tracking_agent.logger import structlog

logger = structlog.get_logger()


def clone_github_repo(github_url: str) -> Path:
    # TODO: validate github_url
    repo_path = Path(config.local_repo_path) / github_url.split("/")[-1]
    if repo_path.exists():
        logger.info("Repo already cloned", path=str(repo_path))
        return repo_path
    logger.info("Cloning repo", github_url=github_url, path=str(repo_path))
    try:
        repo_path.parent.mkdir(parents=True, exist_ok=True)
        git.Repo.clone_from(github_url, repo_path)
        logger.info("Repo cloned successfully", path=str(repo_path))
    except git.GitCommandError as e:
        logger.error("Error cloning repository", error=str(e))
        raise
    return repo_path


def get_gcs_bucket(
    project_id: str = config.google_config.project_id,
    bucket_name_to_get: str = config.bucket_name,
) -> storage.Bucket:
    logger.info(
        "Connecting to GCS bucket",
        bucket_name=bucket_name_to_get,
        project_id=project_id,
    )
    storage_client = storage.Client(project=project_id)
    try:
        bucket_obj = storage_client.get_bucket(bucket_name_to_get)
        logger.info("Connected to bucket", bucket_name=bucket_obj.name)
        return bucket_obj
    except Exception as e:
        logger.error(
            "Error accessing GCS bucket", bucket_name=bucket_name_to_get, error=str(e)
        )
        raise


def chunk_file_by_lines(file_path: Path, chunk_size: int = 20):
    """
    Splits a file into chunks of N lines. Returns a list of (start_line, end_line, chunk_text).
    """
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    chunks = []
    for i in range(0, len(lines), chunk_size):
        chunk_lines = lines[i : i + chunk_size]
        start_line = i + 1
        end_line = i + len(chunk_lines)
        chunks.append((start_line, end_line, "".join(chunk_lines)))
    return chunks


try:
    import tree_sitter

    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False


def chunk_file_by_tree_sitter(file_path: Path, language: str = "python"):
    """
    Chunks a file using tree-sitter to create semantically meaningful chunks.
    Returns a list of (start_line, end_line, chunk_text).
    Supported languages: python, typescript, java, kotlin, go
    """
    import tree_sitter_python as python_tree_sitter
    import tree_sitter_typescript as typescript_tree_sitter
    import tree_sitter_java as java_tree_sitter
    import tree_sitter_kotlin as kotlin_tree_sitter
    import tree_sitter_go as go_tree_sitter
    import tree_sitter_javascript as javascript_tree_sitter

    if not TREE_SITTER_AVAILABLE:
        logger.warn(
            "tree-sitter not available; falling back to line-based chunking.",
            file=str(file_path),
        )
        return chunk_file_by_lines(file_path)

    supported_langs = {"python", "typescript", "java", "kotlin", "go", "javascript"}
    if language not in supported_langs:
        logger.warn(
            f"Language {language} not supported for tree-sitter chunking; falling back to line-based chunking.",
            file=str(file_path),
        )
        return chunk_file_by_lines(file_path)

    try:
        # Load the appropriate language parser
        if language == "python":
            language_parser = python_tree_sitter.language()
        elif language == "typescript":
            language_parser = typescript_tree_sitter.language()
        elif language == "java":
            language_parser = java_tree_sitter.language()
        elif language == "kotlin":
            language_parser = kotlin_tree_sitter.language()
        elif language == "go":
            language_parser = go_tree_sitter.language()
        elif language == "javascript":
            language_parser = javascript_tree_sitter.language()
        else:
            raise ValueError(
                f"Language {language} not supported for tree-sitter chunking."
            )

        language = tree_sitter.Language(language_parser)
        parser = tree_sitter.Parser(language=language)

        # Read file content
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            lines = content.splitlines()

        # Parse the file
        tree = parser.parse(bytes(content, "utf-8"))

        chunks = []
        root_node = tree.root_node

        for child in root_node.children:
            if child.type in (
                "function_definition",
                "class_definition",
                "method_definition",
                "function_declaration",
                "class_declaration",
                "method_declaration",
            ):
                start_line = child.start_point[0] + 1  # 1-indexed
                end_line = child.end_point[0] + 1
                chunk_text = "\n".join(lines[start_line - 1 : end_line])
                chunks.append((start_line, end_line, chunk_text))

        if not chunks:
            logger.info(
                "No suitable tree-sitter chunks found; falling back to line-based chunking.",
                file=str(file_path),
            )
            return chunk_file_by_lines(file_path)

        return chunks

    except Exception as e:
        logger.error(
            "Error during tree-sitter chunking; falling back to line-based chunking.",
            file=str(file_path),
            error=str(e),
        )
        return chunk_file_by_lines(file_path)


def upload_repo_to_gcs(
    bucket: storage.Bucket,
    local_repo_path: Path,
    chunking_strategy: str = "tree-sitter",  # 'lines' or 'tree-sitter'
    chunk_size: int = 20,  # Only used for line-based chunking
) -> str:
    """
    Uploads supported files from a local repo to GCS, chunking them as specified.
    Each chunk is uploaded as a separate file named <relative_path>__lines_<start>-<end>.txt.
    chunking_strategy: 'lines' (default) or 'tree-sitter'.
    """

    def is_supported_file(file_path: Path) -> bool:
        file_lower = file_path.name.lower()
        return (
            any(file_lower.endswith(ext.lower()) for ext in config.supported_extensions)
            or file_path.name in config.supported_extensions
        )

    def upload_file(blob, src_path: Path):
        if blob.exists():
            return False
        blob.upload_from_filename(str(src_path))
        return True

    gcs_folder_prefix = config.gcs_folder_prefix + local_repo_path.name
    logger.info(
        "Uploading repo files to GCS (chunked)",
        local_repo_path=str(local_repo_path),
        bucket_name=config.bucket_name,
        gcs_folder_prefix=gcs_folder_prefix,
        chunking_strategy=chunking_strategy,
        chunk_size=chunk_size,
    )

    max_bytes = (
        config.max_file_size_mb * 1024 * 1024 if config.max_file_size_mb > 0 else 0
    )
    uploaded, skipped = 0, 0

    for file_path in Path(local_repo_path).rglob("*"):
        if file_path.is_dir() or ".git" in file_path.parts:
            continue
        if not is_supported_file(file_path):
            continue
        if max_bytes > 0 and file_path.stat().st_size > max_bytes:
            skipped += 1
            logger.info(
                "Skipping large file",
                file=str(file_path),
                size=file_path.stat().st_size,
            )
            continue

        rel_path = file_path.relative_to(local_repo_path)
        ext = file_path.suffix.lower()

        # For .md/.txt, treat as a single chunk
        if ext in [".md", ".txt"]:
            with open(file_path, "r", encoding="utf-8") as f:
                chunk_text = f.read()
            start_line = 1
            end_line = chunk_text.count("\n") + 1
            chunks = [(start_line, end_line, chunk_text)]
        else:
            # For code, chunk by lines or tree-sitter
            if chunking_strategy == "tree-sitter":
                if TREE_SITTER_AVAILABLE:
                    # Detect language from file extension
                    ext = file_path.suffix.lower()
                    language = None
                    if ext == ".py":
                        language = "python"
                    elif ext in [".js", ".jsx"]:
                        language = "javascript"
                    elif ext in [".ts", ".tsx"]:
                        language = "typescript"
                    elif ext == ".java":
                        language = "java"
                    elif ext == ".kt":
                        language = "kotlin"
                    elif ext == ".go":
                        language = "go"

                    if language:
                        chunks = chunk_file_by_tree_sitter(file_path, language=language)
                    else:
                        # Fall back to line-based chunking for unsupported languages
                        logger.warn(
                            "Unsupported language for tree-sitter chunking, falling back to line-based.",
                            file=str(file_path),
                            extension=ext,
                        )
                        chunks = chunk_file_by_lines(file_path, chunk_size=chunk_size)
                else:
                    logger.warn(
                        "tree-sitter not available, falling back to line-based chunking.",
                        file=str(file_path),
                    )
                    chunks = chunk_file_by_lines(file_path, chunk_size=chunk_size)
            else:
                chunks = chunk_file_by_lines(file_path, chunk_size=chunk_size)

        for start_line, end_line, chunk_text in chunks:
            chunk_file_name = f"{rel_path}__lines_{start_line}-{end_line}.txt"
            # Replace / with __ in temp file name to avoid subdirs in temp
            temp_chunk_path = Path(tempfile.gettempdir()) / chunk_file_name.replace(
                "/", "__"
            )
            with open(temp_chunk_path, "w", encoding="utf-8") as f:
                f.write(chunk_text)
            gcs_blob_name = str(Path(gcs_folder_prefix) / chunk_file_name).replace(
                "\\", "/"
            )
            blob = bucket.blob(gcs_blob_name)
            try:
                if upload_file(blob, temp_chunk_path):
                    uploaded += 1
                    if uploaded % 50 == 0:
                        logger.info("Uploaded chunks so far", uploaded=uploaded)
                else:
                    skipped += 1
            except Exception as e:
                logger.error(
                    "Error uploading chunk to GCS",
                    file=str(file_path),
                    chunk_file=chunk_file_name,
                    gcs_path=f"gs://{config.bucket_name}/{gcs_blob_name}",
                    error=str(e),
                )
            finally:
                if temp_chunk_path.exists():
                    temp_chunk_path.unlink()

    logger.info("Upload complete", uploaded=uploaded, skipped=skipped)
    return gcs_folder_prefix


def create_rag_corpus(
    rag_corpus_display_name: str,
    description: str,
    project_id: str = config.google_config.project_id,
    location: str = config.google_config.location,
    embedding_model_name: str = config.embedding_model,
) -> RagCorpus:
    """Creates a Vertex AI RAG Corpus."""
    logger.info(
        "Creating RAG corpus",
        display_name=rag_corpus_display_name,
        project_id=project_id,
        location=location,
        embedding_model=embedding_model_name,
    )
    vertexai.init(project=project_id, location=location)

    corpus: RagCorpus = rag.create_corpus(
        display_name=rag_corpus_display_name,
        description=description,
        embedding_model_config=EmbeddingModelConfig(
            publisher_model=embedding_model_name
        ),
    )
    logger.info("RAG corpus created", corpus_name=corpus.name)
    return corpus


def import_files_to_vertex_rag(
    rag_corpus_name: str,
    gcs_uris: List[str],
    project_id: str = config.google_config.project_id,
    location: str = config.google_config.location,
    chunk_size: int = config.chunk_size,
    chunk_overlap: int = config.chunk_overlap,
) -> ImportRagFilesResponse:
    """Imports files from GCS to the specified Vertex AI RAG Corpus."""
    logger.info(
        "Importing files from GCS to Vertex RAG corpus",
        rag_corpus_name=rag_corpus_name,
        gcs_uris=gcs_uris,
        project_id=project_id,
        location=location,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    vertexai.init(project=project_id, location=location)

    formatted_gcs_uris = []
    for uri in gcs_uris:
        if not uri.startswith("gs://"):
            logger.warn(
                "GCS URI does not start with gs://, attempting to format.",
                original_uri=uri,
            )
            formatted_gcs_uris.append(f"gs://{config.bucket_name}/{uri.strip('/')}/")
        else:
            formatted_gcs_uris.append(uri)

    result = rag.import_files(
        corpus_name=rag_corpus_name,
        paths=formatted_gcs_uris,
        transformation_config=TransformationConfig(
            chunking_config=ChunkingConfig(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        ),
    )
    logger.info(
        "Import to Vertex RAG completed",
        imported_files_count=result.imported_rag_files_count,
        failed_files_count=result.failed_rag_files_count,
    )
    return result


def ingest_repository_to_rag_corpus(
    github_url: str,
    rag_corpus_display_name: Optional[str] = None,
    project_id: str = config.google_config.project_id,
    location: str = config.google_config.location,
    gcs_bucket_name: str = config.bucket_name,
    embedding_model_to_use: str = config.embedding_model,
    chunk_size_to_use: int = config.chunk_size,
    chunk_overlap_to_use: int = config.chunk_overlap,
) -> Tuple[Optional[RagCorpus], Optional[ImportRagFilesResponse]]:
    """
    Orchestrates cloning a GitHub repo, uploading to GCS, creating a RAG Corpus, and importing files.
    Relies on Vertex AI's built-in chunking.
    """
    logger.info(
        "Starting RAG ingestion pipeline for repository",
        github_url=github_url,
        project_id=project_id,
        location=location,
    )
    vertexai.init(project=project_id, location=location)

    try:
        repo_name_from_url = github_url.split("/")[-1].replace(".git", "")
        repo_org_and_name = "/".join(github_url.split("/")[-2:]).replace(".git", "")

        local_repo_path = clone_github_repo(github_url)

        gcs_bucket_object = get_gcs_bucket(
            project_id=project_id, bucket_name_to_get=gcs_bucket_name
        )

        gcs_folder_prefix_uploaded = upload_repo_to_gcs(
            gcs_bucket_object, local_repo_path
        )

        gcs_uri_for_import = [
            f"gs://{gcs_bucket_name}/{gcs_folder_prefix_uploaded.strip('/')}/"
        ]

        if not rag_corpus_display_name:
            rag_corpus_display_name = (
                f"rag-corpus-{repo_name_from_url}-{str(uuid.uuid4())[:8]}"
            )

        description = f"RAG corpus for {repo_org_and_name} from {github_url}"

        created_corpus = create_rag_corpus(
            rag_corpus_display_name=rag_corpus_display_name,
            description=description,
            project_id=project_id,
            location=location,
            embedding_model_name=embedding_model_to_use,
        )

        if not created_corpus or not hasattr(created_corpus, "name"):
            logger.error(
                "Failed to create RAG corpus or corpus has no name.",
                corpus_object=created_corpus,
            )
            return None, None

        import_response = import_files_to_vertex_rag(
            rag_corpus_name=created_corpus.name,
            gcs_uris=gcs_uri_for_import,
            project_id=project_id,
            location=location,
            chunk_size=chunk_size_to_use,
            chunk_overlap=chunk_overlap_to_use,
        )
        logger.info(
            "RAG ingestion pipeline completed for repository.",
            github_url=github_url,
            corpus_name=created_corpus.name,
            imported_count=import_response.imported_rag_files_count
            if import_response
            else "N/A",
            failed_count=import_response.failed_rag_files_count
            if import_response
            else "N/A",
        )
        return created_corpus, import_response

    except Exception as e:
        logger.error("Error during RAG ingestion pipeline", error=str(e), exc_info=True)
        return None, None


def main():
    argparser = argparse.ArgumentParser(
        description="Ingest a GitHub repository into Vertex AI RAG or query a corpus."
    )
    argparser.add_argument(
        "github_url", type=str, help="URL of the GitHub repository for ingestion."
    )
    argparser.add_argument(
        "--corpus-display-name",
        type=str,
        help="Optional display name for the RAG corpus during ingestion.",
    )
    argparser.add_argument(
        "--project-id",
        type=str,
        default=config.google_config.project_id,
        help="Google Cloud Project ID.",
    )
    argparser.add_argument(
        "--location",
        type=str,
        default=config.google_config.location,
        help="Google Cloud Location.",
    )

    args = argparser.parse_args()

    logger.info(
        "Running RAG ingestion via CLI for repository:", github_url=args.github_url
    )

    corpus_obj, import_result = ingest_repository_to_rag_corpus(
        github_url=args.github_url,
        rag_corpus_display_name=args.corpus_display_name,
        project_id=args.project_id,
        location=args.location,
    )

    if corpus_obj and import_result:
        logger.info(
            "RAG ingestion successful.",
            corpus_name=corpus_obj.name,
            corpus_display_name=corpus_obj.display_name,
            imported_count=import_result.imported_rag_files_count,
            failed_count=import_result.failed_rag_files_count,
        )
    else:
        logger.error("RAG ingestion failed for repository.", github_url=args.github_url)


if __name__ == "__main__":
    main()
