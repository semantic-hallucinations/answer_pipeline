from qdrant_client import QdrantClient, models
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.core.chat_engine.types import ChatMode
from llama_index.core.indices.base import BaseIndex
from llama_index.core.vector_stores.types import BasePydanticVectorStore
from fastapi import Depends
from llama_index.core import PromptTemplate
from ..settings import settings
from .memory import memory_manager

qa_prompt = PromptTemplate("""
Ты — помощник по университету, который отвечает на вопросы студентов, преподавателей и сотрудников. Тебе предоставлен контекст из документов университета, который содержит актуальную информацию. Твоя задача — внимательно изучить этот контекст и предоставить точный ответ на заданный вопрос.

Правила работы:
1. Ответ должен быть основан ТОЛЬКО на информации из предоставленного контекста.
2. Если в контексте есть только упоминание темы без подробностей, ты ДОЛЖЕН сказать: "В предоставленном контексте есть только упоминание этой темы, но нет подробной информации."
3. ЗАПРЕЩЕНО придумывать или додумывать информацию, которой нет в контексте.
4. Если в контексте нет информации для ответа на вопрос, ты ДОЛЖЕН сказать: "В предоставленном контексте нет информации для ответа на этот вопрос."
5. Если контекст неполный или требует уточнения, укажи это.
6. Если ты видишь, что контекст содержит информацию о другом предмете (например, о фильме), а не о запрошенной теме, ты ДОЛЖЕН это указать.

Контекст:
{context}

Вопрос:
{question}


Ответ начинай сразу с сути, без вводных фраз. Ответ должен быть коротким и точным.
""")

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
        text_qa_template=qa_prompt,
        temperature=0.4,
        similarity_top_k=3,
        memory=memory_manager.memory,
        verbose=True
    )