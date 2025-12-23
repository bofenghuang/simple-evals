import os
import time
from typing import Any

import openai
from openai import AzureOpenAI
from openai import OpenAI

from ..types import MessageList, SamplerBase, SamplerResponse

OPENAI_SYSTEM_MESSAGE_API = "You are a helpful assistant."
OPENAI_SYSTEM_MESSAGE_CHATGPT = (
    "You are ChatGPT, a large language model trained by OpenAI, based on the GPT-4 architecture."
    + "\nKnowledge cutoff: 2023-12\nCurrent date: 2024-04-01"
)


class ChatCompletionSampler(SamplerBase):
    """
    Sample from OpenAI's chat completion API
    """

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        system_message: str | None = None,
        temperature: float = 0.5,
        max_tokens: int = 1024,
        use_gateway: bool = False,
        gateway_base_url: str = "https://genai-gateway-shared-nl-gcp.doctolib.ai",
    ):
        # self.api_key_name = "OPENAI_API_KEY"
        # self.client = OpenAI()
        # using api_key=os.environ.get("OPENAI_API_KEY")  # please set your OPENAI_API_KEY
        self.api_key_name = "AZURE_OPENAI_API_KEY"
        self.client = AzureOpenAI(
            azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
            api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
        )
        self.model = model
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.image_format = "url"
        self.use_gateway = use_gateway

        if use_gateway:
            # Use GenAI Gateway
            self.api_key_name = "GATEWAY_API_KEY"
            gateway_api_key = os.environ.get("GATEWAY_API_KEY")
            if not gateway_api_key:
                raise ValueError("GATEWAY_API_KEY environment variable must be set when use_gateway=True")

            self.client = OpenAI(
                api_key=gateway_api_key,
                base_url=gateway_base_url
            )
        else:
            # Use Azure OpenAI
            self.api_key_name = "AZURE_OPENAI_API_KEY"
            self.client = AzureOpenAI(
                azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
                api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
                api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
            )

    def _handle_image(
        self,
        image: str,
        encoding: str = "base64",
        format: str = "png",
        fovea: int = 768,
    ):
        new_image = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/{format};{encoding},{image}",
            },
        }
        return new_image

    def _handle_text(self, text: str):
        return {"type": "text", "text": text}

    def _pack_message(self, role: str, content: Any):
        return {"role": str(role), "content": content}

    def __call__(self, message_list: MessageList, response_schema: object | None = None) -> SamplerResponse:
        if self.system_message:
            message_list = [
                self._pack_message("system", self.system_message)
            ] + message_list
        trial = 0
        while True:
            try:
                # Prefer Pydantic structured output when model is provided
                if response_schema is not None:
                    response = self.client.chat.completions.parse(
                        model=self.model,
                        messages=message_list,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        response_format=response_schema,
                    )
                    msg = response.choices[0].message
                    parsed = msg.parsed  # a Pydantic model instance
                    content = parsed.model_dump_json()  # JSON string of the structured output
                else:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=message_list,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                    )
                    content = response.choices[0].message.content
                if content is None:
                    raise ValueError("OpenAI API returned empty response; retrying")
                return SamplerResponse(
                    response_text=content,
                    response_metadata={"usage": response.usage},
                    actual_queried_message_list=message_list,
                )
            # NOTE: BadRequestError is triggered once for MMMU, please uncomment if you are reruning MMMU
            except openai.BadRequestError as e:
                print("Bad Request Error", e)
                return SamplerResponse(
                    response_text="No response (bad request).",
                    response_metadata={"usage": None},
                    actual_queried_message_list=message_list,
                )
            except Exception as e:
                exception_backoff = 2**trial  # expontial back off
                print(
                    f"Rate limit exception so wait and retry {trial} after {exception_backoff} sec",
                    e,
                )
                time.sleep(exception_backoff)
                trial += 1
            # unknown error shall throw exception
