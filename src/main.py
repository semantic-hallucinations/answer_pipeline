import logging
import sys

from fastapi import Depends, FastAPI
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
def main(message: str, chat_engine: BaseChatEngine = Depends(get_chat_engine)):
    streaming_response = chat_engine.chat(message)
    return streaming_response
