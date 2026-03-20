import os
import time
from typing import Any

from litellm import completion

from ..types import MessageList, SamplerBase, SamplerResponse


class LiteLLMSampler(SamplerBase):
    """
    Sample responses using LiteLLM's `completion` helper.
    Useful for Anthropic (and other) backends supported by LiteLLM.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-5-20250929",
        system_message: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
    ):
        self.model = model
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.api_base = api_base or os.environ.get("ANTHROPIC_API_BASE")
        self.image_format = "base64"

    def _pack_message(self, role: str, content: Any) -> dict[str, Any]:
        return {"role": str(role), "content": content}

    def __call__(self, message_list: MessageList, response_schema: object | None = None) -> SamplerResponse:
        if self.system_message:
            message_list = [self._pack_message("system", self.system_message)] + message_list
        trial = 0
        while True:
            try:
                kwargs: dict[str, Any] = {
                    "model": self.model,
                    "messages": message_list,
                }
                if self.temperature is not None:
                    kwargs["temperature"] = self.temperature
                if self.max_tokens is not None:
                    kwargs["max_tokens"] = self.max_tokens
                if self.reasoning_effort is not None:
                    kwargs["reasoning_effort"] = self.reasoning_effort
                if self.api_key is not None:
                    kwargs["api_key"] = self.api_key
                if self.api_base is not None:
                    kwargs["api_base"] = self.api_base
                if response_schema is not None:
                    # LiteLLM supports passing Pydantic models directly as response_format
                    kwargs["response_format"] = response_schema

                response = completion(**kwargs)

                if hasattr(response, "choices"):
                    response_text = response.choices[0].message.content
                    # usage = getattr(response, "usage", None)
                else:
                    message = response["choices"][0]["message"]
                    response_text = message.get("content")
                    # usage = response.get("usage")

                response_text = response_text or ""
                if not response_text:
                    print(f"LiteLLM returned empty response: {response}")
                
                # response_metadata = {}
                # if usage is not None:
                #     response_metadata["usage"] = usage

                return SamplerResponse(
                    response_text=response_text,
                    # response_metadata=response_metadata,
                    response_metadata={},
                    actual_queried_message_list=message_list,
                )
            except Exception as e:
                exception_backoff = 2**trial  # exponential backoff
                print(
                    f"LiteLLM request failed; retry {trial} after {exception_backoff} sec",
                    e,
                )
                time.sleep(exception_backoff)
                trial += 1
            # unknown error shall throw exception

