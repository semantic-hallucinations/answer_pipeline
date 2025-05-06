from typing import Dict
from uuid import uuid4
from llama_index.core.memory import ChatSummaryMemoryBuffer
from llama_index.llms.openrouter import OpenRouter
import tiktoken
from settings import settings

class MemoryManager:
    def __init__(self):
        self.memory = ChatSummaryMemoryBuffer.from_defaults(
            llm=OpenRouter(
                model=settings.get_model_name(),
                api_key=settings.get_model_key()
            ),
            token_limit=1024,
            tokenizer_fn=tiktoken.get_encoding("cl100k_base").encode
        )

memory_manager = MemoryManager()