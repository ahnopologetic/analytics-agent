from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class GoogleConfig(BaseSettings):
    project_id: str
    location: str

    model_config = SettingsConfigDict(
        env_prefix="TRK_GOOGLE_", env_file=".env", extra="allow"
    )


class AgentConfig(BaseSettings):
    distance_threshold: float = Field(default=0.6)
    top_k: int = Field(default=10)

    google_config: GoogleConfig = Field(default_factory=GoogleConfig)

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="TRK_AGENT_", extra="allow"
    )


config = AgentConfig()
