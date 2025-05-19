import argparse
from typing import Any, Dict, List, Optional

import vertexai
from google.cloud import aiplatform_v1beta1
from vertexai.preview import rag
from vertexai.preview.rag.utils import resources

from tracking_agent.config import config
from tracking_agent.logger import structlog

logger = structlog.get_logger()


def retrieve_contexts_from_corpus(
    rag_corpus_name: str,
    query_text: str,
    project_id: str = config.google_config.project_id,
    location: str = config.google_config.location,
    top_k_results: int = config.top_k,
    vector_distance_threshold_val: Optional[float] = config.distance_threshold,
) -> Optional[List[dict[str, Any]]]:
    """
    Retrieves relevant contexts (chunks) from a specified RAG corpus based on a query.
    """
    logger.info(
        "Retrieving contexts from RAG corpus",
        rag_corpus_name=rag_corpus_name,
        query=query_text,
        project_id=project_id,
        location=location,
    )
    vertexai.init(project=project_id, location=location)

    try:
        if vector_distance_threshold_val is not None:
            logger.warn(
                "Vector distance threshold is not directly supported by vertexai.preview.rag.retrieval_query. It might be part of corpus/engine config."
            )
            pass

        response: aiplatform_v1beta1.RetrieveContextsResponse = rag.retrieval_query(
            rag_resources=[
                resources.RagResource(
                    rag_corpus=rag_corpus_name,
                )
            ],
            text=query_text,
            similarity_top_k=top_k_results,
        )

        contexts = []
        if response and hasattr(response, "contexts") and response.contexts:
            for context in response.contexts.contexts:
                contexts.append(
                    {
                        "source_uri": context.source_uri,
                        "display_name": context.source_display_name,
                        "text": context.text,
                        "distance": context.distance,
                        "score": context.score,
                        "sparse_distance": context.sparse_distance,
                        "chunk": context.chunk.text,  # TODO: resolve this class
                    }
                )
        logger.info(f"Retrieved {len(contexts)} contexts for query.")
        return contexts
    except Exception as e:
        logger.error(
            "Error retrieving contexts from RAG corpus", error=str(e), exc_info=True
        )
        return None


class VertexAISemanticSearch:
    """
    Semantic code search using Vertex AI RAG service.
    Ingestion and chunking are handled by the pipeline in rag_corpus.py.
    This class provides a search interface over an existing RAG corpus.
    """

    def __init__(self, rag_corpus_name: str, project_id: str, location: str):
        self.rag_corpus_name = rag_corpus_name
        self.project_id = project_id
        self.location = location

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search the RAG corpus for code chunks semantically similar to the query.
        Returns a list of dicts with 'source_uri' and 'text'.
        If you encoded metadata (e.g., file path, start/end lines) in the GCS file name or chunk text,
        you can parse it here.
        """
        results = retrieve_contexts_from_corpus(
            rag_corpus_name=self.rag_corpus_name,
            query_text=query,
            project_id=self.project_id,
            location=self.location,
            top_k_results=top_k,
        )
        # Optionally parse metadata from source_uri or text
        if results:
            for r in results:
                r.update(self._parse_metadata_from_source_uri(r["source_uri"]))
        return results or []

    @staticmethod
    def _parse_metadata_from_source_uri(source_uri: str) -> Dict[str, Any]:
        """
        Example: Parse file path and (optionally) line numbers from the GCS source_uri.
        You must encode this info during ingestion (e.g., in the file name or folder structure).
        """
        # Example: gs://bucket/prefix/path/to/file.py__lines_10-30.txt
        import os
        import re

        uri_path = source_uri.split("/", 3)[-1]  # Remove gs://bucket/
        file_info = os.path.basename(uri_path)
        # Try to extract lines info if present
        m = re.match(r"(.+?)(__lines_(\d+)-(\d+))?\.txt$", file_info)
        if m:
            file_path = m.group(1)
            start_line = int(m.group(3)) if m.group(3) else None
            end_line = int(m.group(4)) if m.group(4) else None
            return {
                "file_path": file_path,
                "start_line": start_line,
                "end_line": end_line,
            }
        return {"file_path": file_info}


def main():
    """Example usage of the VertexAISemanticSearch class."""
    from tracking_agent.config import config

    args = argparse.ArgumentParser()
    args.add_argument("query", type=str)
    args.add_argument("--rag_corpus_name", type=str, required=True)
    args = args.parse_args()

    RAG_CORPUS_NAME = args.rag_corpus_name

    searcher = VertexAISemanticSearch(
        rag_corpus_name=RAG_CORPUS_NAME,
        project_id=config.google_config.project_id,
        location=config.google_config.location,
    )
    results = searcher.search(args.query, top_k=5)
    for idx, res in enumerate(results):
        print(f"Result {idx + 1}:")
        print(f"  File: {res.get('file_path')}")
        print(f"  Lines: {res.get('start_line')} - {res.get('end_line')}")
        print(f"  Source URI: {res['source_uri']}")
        print(f"  Text: {res['text'][:200]}{'...' if len(res['text']) > 200 else ''}")
        print()


if __name__ == "__main__":
    main()
