from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from tracking_agent.config import GoogleConfig


class RAGIngestConfig(BaseSettings):
    google_config: GoogleConfig = Field(default_factory=GoogleConfig)

    bucket_name: str = Field("tracking-agent", description="GCS Bucket Name")
    gcs_folder_path: str = Field(
        "rag-code-data", description="GCS folder path (prefix)"
    )
    max_file_size_mb: int = Field(
        10, description="Maximum file size in MB to upload (0 for no limit)"
    )
    embedding_model: str = Field(
        "publishers/google/models/text-embedding-005",
        description="Embedding model name",
    )
    model_id: str = Field(
        "gemini-2.5-flash-preview-04-17", description="Vertex model ID"
    )
    supported_extensions: list[str] = Field(
        default_factory=lambda: [
            ".py",
            ".java",
            ".js",
            ".ts",
            ".tsx",
            ".go",
            ".c",
            ".cpp",
            ".h",
            ".hpp",
            ".cs",
            ".rb",
            ".php",
            ".swift",
            ".kt",
            ".scala",
            ".md",
            ".txt",
            ".rst",
            ".html",
            ".css",
            ".scss",
            ".yaml",
            ".yml",
            ".json",
            ".xml",
            ".proto",
            "Dockerfile",
            ".sh",
            ".tf",
            ".tfvars",
            ".bicep",
            ".gradle",
            "pom.xml",
            "requirements.txt",
            "package.json",
            "go.mod",
            "go.sum",
            "Cargo.toml",
        ],
        description="Supported file extensions",
    )
    local_repo_path: str = Field(
        "./cloned_repo", description="Local path to clone repo"
    )
    chunk_size: int = Field(500, description="Chunk size for RAG import")
    chunk_overlap: int = Field(100, description="Chunk overlap for RAG import")
    similarity_top_k: int = Field(10, description="Top K for similarity search")
    vector_distance_threshold: float = Field(
        0.5, description="Vector distance threshold"
    )

    model_config = SettingsConfigDict(
        env_prefix="TRK_INGEST_", case_sensitive=False, env_file=".env", extra="allow"
    )
