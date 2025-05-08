import logging
import sys
from typing import Annotated

from fastapi import Body, Depends, FastAPI
from llama_index.core import Settings
from llama_index.core.chat_engine.types import BaseChatEngine
from llama_index.llms.openrouter import OpenRouter

from .pipeline import get_chat_engine
from .settings import settings

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

Settings.llm = OpenRouter(
    model=settings.get_model_name(),
    api_key=settings.get_model_key(),
    max_tokens=10000,
    context_window=128000,
    temperature=0.4,
    timeout=60,
)


app = FastAPI()


@app.post("/")
async def main(
    message: Annotated[str, Body()],
    chat_engine: BaseChatEngine = Depends(get_chat_engine),
):
    streaming_response = await chat_engine.achat(message)
    response = streaming_response.response

    logging.info(f"Тип streaming_response: {type(streaming_response)}")
    logging.info(f"Атрибуты streaming_response: {dir(streaming_response)}")
    logging.info(f"Источники: {streaming_response.sources}")

    sources = []
    for source in streaming_response.sources:
        if hasattr(source, "raw_output"):
            for node in source.raw_output:
                if hasattr(node, "node") and hasattr(node.node, "metadata"):
                    if "source_url" in node.node.metadata:
                        sources.append(node.node.metadata["source_url"])

    return {"response": response, "source_urls": sources}
