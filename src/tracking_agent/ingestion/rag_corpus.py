import argparse
import os
from pathlib import Path
from typing import Any
import uuid
import shutil
import tempfile

import git
import vertexai
from google import genai
from google.cloud import storage
from google.genai.types import GenerateContentConfig, Retrieval, Tool, VertexRagStore
from google.cloud.aiplatform_v1.types.vertex_rag_data_service import (
    ImportRagFilesResponse,
)
from vertexai import rag
from vertexai.rag.rag_data import RagCorpus

from tracking_agent.ingestion.config import RAGIngestConfig
from tracking_agent.logger import structlog

logger = structlog.get_logger()
config = RAGIngestConfig()


def clone_github_repo(github_url: str) -> Path:
    # TODO: validate github_url
    repo_path = Path(config.local_repo_path) / github_url.split("/")[-1]
    if repo_path.exists():
        logger.info("Repo already cloned", path=str(repo_path))
        return repo_path
    logger.info("Cloning repo", github_url=github_url, path=str(repo_path))
    try:
        git.Repo.clone_from(github_url, repo_path)
        logger.info("Repo cloned successfully", path=str(repo_path))
    except git.GitCommandError as e:
        logger.error("Error cloning repository", error=str(e))
        raise
    return repo_path


def get_gcs_bucket() -> storage.Bucket:
    logger.info("Connecting to GCS bucket", bucket_name=config.bucket_name)
    storage_client = storage.Client(project=config.google_config.project_id)
    try:
        bucket = storage_client.get_bucket(config.bucket_name)
        logger.info("Connected to bucket", bucket_name=bucket.name)
        return bucket
    except Exception as e:
        logger.error(
            "Error accessing GCS bucket", bucket_name=config.bucket_name, error=str(e)
        )
        raise


def upload_repo_to_gcs(bucket: storage.Bucket, local_repo_path: Path) -> str:
    """
    Uploads supported files from a local repo to GCS. Non .md/.txt files are converted to .txt with original name preserved (e.g., main.js -> main.js.txt).
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

    def convert_to_txt(src_path: Path) -> Path:
        temp_dir = Path(tempfile.gettempdir())
        temp_txt_path = temp_dir / f"{src_path.name}.txt"
        shutil.copyfile(src_path, temp_txt_path)
        return temp_txt_path

    gcs_folder_prefix = config.gcs_folder_prefix + local_repo_path.name
    logger.info(
        "Uploading repo files to GCS",
        local_repo_path=str(local_repo_path),
        bucket_name=config.bucket_name,
        gcs_folder_prefix=gcs_folder_prefix,
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

        ext = file_path.suffix.lower()
        needs_conversion = ext not in [".md", ".txt"]
        upload_path = file_path
        cleanup_temp = False

        if needs_conversion:
            upload_path = convert_to_txt(file_path)
            # Place in GCS with .txt appended to original filename
            rel_path = file_path.relative_to(local_repo_path).with_name(
                file_path.name + ".txt"
            )
            cleanup_temp = True
        else:
            rel_path = file_path.relative_to(local_repo_path)

        gcs_blob_name = str(Path(gcs_folder_prefix) / rel_path).replace("\\", "/")
        blob = bucket.blob(gcs_blob_name)
        try:
            if upload_file(blob, upload_path):
                uploaded += 1
                if uploaded % 50 == 0:
                    logger.info("Uploaded files so far", uploaded=uploaded)
            else:
                skipped += 1
        except Exception as e:
            logger.error(
                "Error uploading file to GCS",
                file=str(file_path),
                gcs_path=f"gs://{config.bucket_name}/{gcs_blob_name}",
                error=str(e),
            )
        finally:
            if cleanup_temp and upload_path.exists():
                upload_path.unlink()

    logger.info("Upload complete", uploaded=uploaded, skipped=skipped)
    return gcs_folder_prefix


def create_rag_corpus(rag_corpus_name: str, repo_org_and_name: str) -> RagCorpus:
    logger.info("Creating RAG corpus", rag_corpus_name=rag_corpus_name)
    corpus = rag.create_corpus(
        display_name=rag_corpus_name,
        description=f"RAG corpus from {repo_org_and_name}",
        backend_config=rag.RagVectorDbConfig(
            rag_embedding_model_config=rag.RagEmbeddingModelConfig(
                vertex_prediction_endpoint=rag.VertexPredictionEndpoint(
                    publisher_model=config.embedding_model
                )
            )
        ),
    )
    logger.info("RAG corpus created", corpus_name=corpus.name)
    return corpus


def import_files_to_vertex_rag(
    rag_corpus_name: str, gcs_folder_prefix: str
) -> ImportRagFilesResponse:
    logger.info(
        "Importing files from GCS to Vertex RAG corpus", rag_corpus_name=rag_corpus_name
    )
    result = rag.import_files(
        corpus_name=rag_corpus_name,
        paths=[f"gs://{config.bucket_name}/{gcs_folder_prefix}/"],
        transformation_config=rag.TransformationConfig(
            chunking_config=rag.ChunkingConfig(
                chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap
            )
        ),
    )
    logger.info("Import to Vertex RAG completed", result=result)
    return result


def create_rag_retrieval_tool(rag_corpus_name: str) -> Tool:
    logger.info("Creating RAG retrieval tool", rag_corpus_name=rag_corpus_name)
    tool = Tool(
        retrieval=Retrieval(
            vertex_rag_store=VertexRagStore(
                rag_corpora=[rag_corpus_name],
                similarity_top_k=config.similarity_top_k,
                vector_distance_threshold=config.vector_distance_threshold,
            )
        )
    )
    logger.info("RAG retrieval tool created")
    return tool


def generate_rag_response(rag_retrieval_tool: Tool, prompt: str) -> Any:
    logger.info("Generating RAG response for prompt", prompt=prompt)
    vertexai.init(
        project=config.google_config.project_id, location=config.google_config.location
    )
    client = genai.Client(
        vertexai=True,
        project=config.google_config.project_id,
        location=config.google_config.location,
    )
    response = client.models.generate_content(
        model=config.model_id,
        contents=prompt,
        config=GenerateContentConfig(tools=[rag_retrieval_tool]),
    )
    logger.info("RAG response generated")
    return response


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("github_url", type=str)
    args = argparser.parse_args()

    logger.info("Starting RAG ingestion pipeline...")

    repo_name = args.github_url.split("/")[-1]
    repo_org_and_name = args.github_url.split("/")[-2:]
    local_repo_path = clone_github_repo(args.github_url)
    bucket = get_gcs_bucket()
    gcs_folder_prefix = upload_repo_to_gcs(bucket, local_repo_path)

    rag_corpus_name = f"trk-rag-corpus-{repo_name}-{uuid.uuid4()}"
    rag_corpus = create_rag_corpus(rag_corpus_name, repo_org_and_name)
    result = import_files_to_vertex_rag(rag_corpus.name, gcs_folder_prefix)
    logger.info("RAG corpus ingestion completed", result=result)


if __name__ == "__main__":
    main()
