from qdrant_client import QdrantClient, models
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.core.chat_engine.types import ChatMode
# from llama_index.core.memory import ChatSummaryMemoryBuffer
from llama_index.core.indices.base import BaseIndex
from llama_index.core.vector_stores.types import BasePydanticVectorStore
from fastapi import Depends, Request
# from llama_index.llms.openrouter import OpenRouter
# import tiktoken
from settings import settings
from .memory import memory_manager
from uuid import uuid4

def get_qdrant_client():
    client =   QdrantClient(
        settings.get_qdrant_url()
    )
    if not client.collection_exists(settings.QDRANT_COLLECTION):
        client.create_collection(
            collection_name=settings.QDRANT_COLLECTION,
            vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE),
        )

    return client


def get_vector_store(
    client: QdrantClient = Depends(get_qdrant_client)
):
    return QdrantVectorStore(
        settings.get_qdrant_collection(),
        client=client
    )


def get_index(
    vector_store: BasePydanticVectorStore = Depends(get_vector_store)
):
    return VectorStoreIndex.from_vector_store(vector_store)


def get_chat_engine(index: BaseIndex = Depends(get_index)):
    return index.as_chat_engine(
        chat_mode=ChatMode.CONTEXT,
        memory=memory_manager.memory,
        verbose=True
    )