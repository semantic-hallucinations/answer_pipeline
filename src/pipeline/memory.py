import tiktoken
from llama_index.core.memory import ChatSummaryMemoryBuffer
from llama_index.llms.openrouter import OpenRouter

from ..settings import settings


class MemoryManager:
    def __init__(self):
        self.memory = None
        self.initialize_memory()

    def initialize_memory(self):
        llm = OpenRouter(
            model=settings.get_model_name(),
            api_key=settings.get_model_key(),
            max_tokens=10000,
            context_window=128000,
            temperature=0.4,
            timeout=60,
        )
        self.memory = ChatSummaryMemoryBuffer.from_defaults(
            llm=llm,
            token_limit=1024,
            tokenizer_fn=tiktoken.get_encoding("cl100k_base").encode,
        )


memory_manager = MemoryManager()
