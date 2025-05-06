from qdrant_client import QdrantClient, models
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.core.chat_engine.types import ChatMode
from llama_index.core.indices.base import BaseIndex
from llama_index.core.vector_stores.types import BasePydanticVectorStore
from fastapi import Depends
from llama_index.core import PromptTemplate
from settings import settings
from .memory import memory_manager

qa_prompt = PromptTemplate("""
Ты — помощник по университету, который отвечает на вопросы студентов, преподавателей и сотрудников. Тебе предоставлен контекст из документов университета, который содержит актуальную информацию. Твоя задача — внимательно изучить этот контекст и предоставить точный ответ на заданный вопрос.

Правила работы:
1. Ответ должен быть основан исключительно на информации из предоставленного контекста.
2. Если ответ на вопрос содержится в контексте, формулируй его четко и лаконично.
3. Если в контексте нет информации, которая позволяет ответить на вопрос, ты обязан сообщить: "Ответ на ваш вопрос не найден в предоставленном контексте."
4. Не придумывай ответы или дополнительные детали, если их нет в контексте.
5. Если вопрос требует уточнения или контекст неполный, укажи это.

Контекст:
{context}

Вопрос:
{question}

Ответ:
"""
)

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
        # text_qa_template=qa_prompt,
        # similarity_top_k=7,
        memory=memory_manager.memory,
        verbose=True
    )