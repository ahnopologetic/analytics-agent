import argparse
import os
from pathlib import Path
from typing import Any
import uuid

import git
import vertexai
from google import genai
from google.cloud import storage
from google.genai.types import GenerateContentConfig, Retrieval, Tool, VertexRagStore
from vertexai import rag
from vertexai.rag.rag_data import RagCorpus

from tracking_agent.ingestion.config import RAGIngestConfig
from tracking_agent.logger import structlog

logger = structlog.get_logger()
config = RAGIngestConfig()


def clone_github_repo(github_url: str) -> Path:
    # TODO: validate github_url
    repo_path = Path(config.local_repo_path / github_url.split("/")[-1])
    if repo_path.exists():
        logger.info("Repo already cloned", path=str(repo_path))
        return  # Already cloned
    logger.info("Cloning repo", github_url=github_url, path=str(repo_path))
    try:
        git.Repo.clone_from(github_url, config.local_repo_path)
        logger.info("Repo cloned successfully")
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


def upload_repo_to_gcs(bucket: storage.Bucket, local_repo_path: Path) -> None:
    logger.info(
        "Uploading repo files to GCS",
        local_repo_path=local_repo_path,
        bucket_name=config.bucket_name,
        gcs_folder_path=config.gcs_folder_path,
    )
    max_bytes = (
        config.max_file_size_mb * 1024 * 1024 if config.max_file_size_mb > 0 else 0
    )
    uploaded = 0
    skipped = 0
    for root, dirs, files in os.walk(local_repo_path):
        if ".git" in dirs:
            dirs.remove(".git")
        for file in files:
            file_lower = file.lower()
            local_file_path = os.path.join(root, file)
            is_supported = (
                any(
                    file_lower.endswith(ext.lower())
                    for ext in config.supported_extensions
                )
                or file in config.supported_extensions
            )
            if not is_supported:
                continue
            if max_bytes > 0:
                file_size_bytes = os.path.getsize(local_file_path)
                if file_size_bytes > max_bytes:
                    skipped += 1
                    logger.info(
                        "Skipping large file",
                        file=local_file_path,
                        size=file_size_bytes,
                    )
                    continue
            relative_path = os.path.relpath(local_file_path, local_repo_path)
            gcs_blob_name = (
                os.path.join(config.gcs_folder_path, relative_path)
                if config.gcs_folder_path
                else relative_path
            )
            gcs_blob_name = gcs_blob_name.replace("\\", "/")
            blob = bucket.blob(gcs_blob_name)
            if blob.exists():
                skipped += 1
                continue
            try:
                blob.upload_from_filename(local_file_path)
                uploaded += 1
                if uploaded % 50 == 0:
                    logger.info("Uploaded files so far", uploaded=uploaded)
            except Exception as e:
                logger.error(
                    "Error uploading file to GCS",
                    file=local_file_path,
                    gcs_path=f"gs://{config.bucket_name}/{gcs_blob_name}",
                    error=str(e),
                )
    logger.info("Upload complete", uploaded=uploaded, skipped=skipped)


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


def import_files_to_vertex_rag(rag_corpus_name: str) -> Any:
    logger.info(
        "Importing files from GCS to Vertex RAG corpus", rag_corpus_name=rag_corpus_name
    )
    result = rag.import_files(
        corpus_name=rag_corpus_name,
        paths=[f"gs://{config.bucket_name}/{config.gcs_folder_path}/"],
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

    logger.info("Starting RAG pipeline...")
    # config = RAGIngestConfig(
    #     github_url="https://github.com/google/adk-python",
    #     project_id="ahnopologetic",
    #     location="us-central1",
    #     bucket_name="tracking-agent",
    #     gcs_folder_path="rag-code-data",
    #     max_file_size_mb=10,
    #     embedding_model="publishers/google/models/text-embedding-005",
    #     model_id="gemini-2.5-flash-preview-04-17",
    #     local_repo_path="./cloned_repo",
    # )

    repo_name = args.github_url.split("/")[-1]
    repo_org_and_name = args.github_url.split("/")[-2:]
    local_repo_path = clone_github_repo(args.github_url)
    bucket = get_gcs_bucket()
    upload_repo_to_gcs(bucket, local_repo_path)

    rag_corpus_name = f"trk-rag-corpus-{repo_name}-{uuid.uuid4()}"
    rag_corpus = create_rag_corpus(rag_corpus_name, repo_org_and_name)

    import_files_to_vertex_rag(rag_corpus.name)


if __name__ == "__main__":
    main()
