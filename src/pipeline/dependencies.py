from fastapi import Depends
from llama_index.core import PromptTemplate, VectorStoreIndex
from llama_index.core.chat_engine.types import ChatMode
from llama_index.core.indices.base import BaseIndex
from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import AsyncQdrantClient, models

from ..log_utils import logger
from ..settings import settings
from .llm import get_llm
from .memory import memory_manager

qa_prompt = PromptTemplate(
    """
Ты — помощник по университету, который отвечает на вопросы поступающих, студентов, преподавателей и сотрудников. Тебе предоставлен контекст из документов университета, который содержит актуальную информацию. Твоя задача — внимательно изучить этот контекст и предоставить точный ответ на заданный вопрос.
Собирай только актуальные данные, если в контексте видишь много слов в прошедшем времени то уточни что данные могут быть устаревшими.
Правила работы:
1. Ответ должен быть основан на только информации из предоставленного контекста.
2. Если контекст неполный или требует уточнения, укажи это.
3. Твоя задача также фильтровать ответ чтобы в нем не было лишних символов и не относящихся к сути ответа слов.
4. Также после всего всего ответа выведи список ссылок на источники откуда была взята информация.
Контекст:
{context}

Вопрос:
{question}

Ответ начинай сразу с сути, без вводных фраз.
"""
)


embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-m3",
)


async def get_qdrant_client():
    client = AsyncQdrantClient(url=settings.get_qdrant_url())
    collection_name = settings.QDRANT_COLLECTION
    logger.info(f"Подключение к коллекции: {collection_name}")

    if not await client.collection_exists(collection_name):
        logger.warning(f"Коллекция {collection_name} не существует")
        await client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=1024, distance=models.Distance.COSINE
            ),
        )
        logger.info(f"Создана новая коллекция: {collection_name}")
    else:
        collection_info = await client.get_collection(collection_name)
        logger.info(
            f"Коллекция {collection_name} содержит {collection_info.points_count} точек"
        )

    return client


async def get_vector_store(client: AsyncQdrantClient = Depends(get_qdrant_client)):
    collection_name = settings.QDRANT_COLLECTION
    logger.info(f"Инициализация векторного хранилища для коллекции: {collection_name}")
    return QdrantVectorStore(
        collection_name=collection_name, aclient=client, text_key="content"
    )


async def get_index(vector_store: BasePydanticVectorStore = Depends(get_vector_store)):
    logger.info("Создание индекса из векторного хранилища")
    return VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)


async def get_chat_engine(index: BaseIndex = Depends(get_index), llm=Depends(get_llm)):
    logger.info("Инициализация чат-движка")
    logger.info(f"\n\nПАМЯТЬ{memory_manager.memory}\n\n")
    return index.as_chat_engine(
        chat_mode=ChatMode.CONTEXT,
        text_qa_template=qa_prompt,
        temperature=0.3,
        similarity_top_k=7,
        memory=memory_manager.memory,
        llm=llm,
    )
