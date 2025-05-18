from google.adk.tools.tool_context import ToolContext
from google.genai import Client
from pydantic import BaseModel

from tracking_agent.config import config


class DefinePatternOut(BaseModel):
    pattern: str
    language: str


def define_pattern(
    matching_contents: list[str], language: str, tool_context: ToolContext
) -> dict:
    """
    Define a pattern for tracking.
    """
    client = Client(
        vertexai=True,
        project=config.google_config.project_id,
        location=config.google_config.location,
    )
    response = client.models.generate_content(
        model="gemini-2.0-flash-001",
        contents=f"Define a pattern for tracking the following contents: {matching_contents=} {language=} {tool_context=}",
        config={
            "response_mime_type": "application/json",
            "response_schema": DefinePatternOut,
        },
    )

    return response.parsed.model_dump(mode="json")
