import base64
import os
import time
from typing import Any
import json
from google.genai import types

from ..types import MessageList, SamplerBase, SamplerResponse


class GeminiVertexSampler(SamplerBase):
    """
    Sample from Google Gemini via Vertex AI using the Google Gen AI SDK.
    Ref: Vertex AI SDK migration guide:
    https://docs.cloud.google.com/vertex-ai/generative-ai/docs/deprecations/genai-vertexai-sdk#text-generation
    """

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        system_message: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        project: str | None = None,
        location: str | None = None,
        thinking_budget_tokens: int | None = None,
        thinking_level: str | None = None,
    ):
        # Lazy import new SDK
        from google import genai  # type: ignore

        self.model_name = model
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.thinking_budget_tokens = thinking_budget_tokens
        self.thinking_level = thinking_level

        resolved_project = project or os.environ.get("VERTEXAI_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCP_PROJECT")
        resolved_location = location or os.environ.get("VERTEXAI_LOCATION") or os.environ.get("GOOGLE_CLOUD_LOCATION") or os.environ.get("GCP_LOCATION") or "us-central1"

        if not resolved_project:
            raise ValueError("VERTEXAI_PROJECT must be set (or pass project=...).")

        # Initialize client targeting Vertex AI backend
        self._genai_client = genai.Client(vertexai=True, project=resolved_project, location=resolved_location)

    def _pack_message(self, role: str, content: Any) -> dict[str, Any]:
        return {"role": str(role), "content": content}

    def _messages_to_contents(self, message_list: MessageList, system_message: str | None) -> list[dict]:
        """
        Convert internal MessageList into google-genai 'contents' format:
        [{'role': 'user'|'model', 'parts': [{'text': '...'}]}]
        System message is passed via config.system_instruction.
        """
        contents = []
        for message in message_list:
            role = message.get("role")
            if role == "system":
                continue
            content = message.get("content")
            parts = []
            if isinstance(content, str):
                parts.append({"text": content})
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") in ("text", "input_text"):
                        parts.append({"text": part.get("text", "")})
                    else:
                        # Coerce other parts to text for minimal compatibility
                        parts.append({"text": str(part)})
            else:
                parts.append({"text": str(content)})
            contents.append({"role": "user" if role == "user" else "model", "parts": parts})
        return contents

    def __call__(self, message_list: MessageList, response_schema: object | None = None) -> SamplerResponse:
        # Collate system message
        system_text = self.system_message
        if system_text is None:
            system_msgs = [m.get("content") for m in message_list if m.get("role") == "system"]
            if system_msgs:
                system_text = "\n\n".join(str(c) for c in system_msgs)

        trial = 0
        while True:
            try:
                contents = self._messages_to_contents(message_list, system_text)
                config: dict[str, Any] = {}
                if system_text:
                    config["system_instruction"] = system_text
                if self.temperature is not None:
                    config["temperature"] = self.temperature
                if self.max_tokens is not None:
                    config["max_output_tokens"] = self.max_tokens
                if self.thinking_budget_tokens is not None:
                    config["thinking_config"] = types.ThinkingConfig(thinking_budget=self.thinking_budget_tokens)  # , include_thoughts=True
                # from gemini-3, switch from thinking_budget to thinking_level
                if self.thinking_level is not None:
                    config["thinking_config"] = types.ThinkingConfig(thinking_level=self.thinking_level)
                if response_schema is not None and hasattr(response_schema, "model_json_schema"):
                    # Gemini structured outputs
                    config["response_mime_type"] = "application/json"
                    config["response_schema"] = response_schema.model_json_schema()

                response = self._genai_client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(**config) if config else None,
                )
                response_text = getattr(response, "text", None) or ""

                actual_messages = message_list
                if system_text and not any(m.get("role") == "system" for m in message_list):
                    actual_messages = [self._pack_message("system", system_text)] + message_list

                return SamplerResponse(
                    response_text=response_text,
                    response_metadata={},
                    actual_queried_message_list=actual_messages,
                )
            except Exception as e:
                exception_backoff = 2**trial
                print(
                    f"GenAI Vertex request failed; retry {trial} after {exception_backoff} sec",
                    e,
                )
                time.sleep(exception_backoff)
                trial += 1


