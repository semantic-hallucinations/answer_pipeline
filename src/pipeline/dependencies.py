import logging
import sys

from fastapi import Depends
from llama_index.core import PromptTemplate, VectorStoreIndex
from llama_index.core.chat_engine.types import ChatMode
from llama_index.core.indices.base import BaseIndex
from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models

from ..settings import settings
from .memory import memory_manager

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

qa_prompt = PromptTemplate(
    """
Ты — помощник по университету, который отвечает на вопросы поступающих, студентов, преподавателей и сотрудников. Тебе предоставлен контекст из документов университета, который содержит актуальную информацию. Твоя задача — внимательно изучить этот контекст и предоставить точный ответ на заданный вопрос.
Собирай только актуальные данные, если в контексте видишь много слов в прошедшем времени то уточни что данные могут быть устаревшими.
Правила работы:
1. Ответ должен быть основан на информации из предоставленного контекста.
2. Ты можешь анализировать контекст и логически додумывать, что хочет узнать пользователь.Этот пункт не касается конкретных дат, номеров групп, телефонов, почт и другой важной информации.
3. Если контекст неполный или требует уточнения, укажи это.
4. Твоя задача также фильтровать ответ чтобы в нем не было лишних символов и не относящихся к сути ответа слов.

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


def get_qdrant_client():
    client = QdrantClient(url=settings.get_qdrant_url())
    collection_name = settings.QDRANT_COLLECTION
    logger.info(f"Подключение к коллекции: {collection_name}")

    if not client.collection_exists(collection_name):
        logger.warning(f"Коллекция {collection_name} не существует")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=1024, distance=models.Distance.COSINE
            ),
        )
        logger.info(f"Создана новая коллекция: {collection_name}")
    else:
        collection_info = client.get_collection(collection_name)
        logger.info(
            f"Коллекция {collection_name} содержит {collection_info.points_count} точек"
        )

    return client


def get_vector_store(client: QdrantClient = Depends(get_qdrant_client)):
    collection_name = settings.QDRANT_COLLECTION
    logger.info(f"Инициализация векторного хранилища для коллекции: {collection_name}")
    return QdrantVectorStore(
        collection_name=collection_name, client=client, text_key="content"
    )


def get_index(vector_store: BasePydanticVectorStore = Depends(get_vector_store)):
    logger.info("Создание индекса из векторного хранилища")
    return VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)


def get_chat_engine(index: BaseIndex = Depends(get_index)):
    logger.info("Инициализация чат-движка")
    logger.info(f"\n\nПАМЯТЬ{memory_manager.memory}\n\n")
    return index.as_chat_engine(
        chat_mode=ChatMode.CONTEXT,
        text_qa_template=qa_prompt,
        temperature=0.4,
        similarity_top_k=3,
        memory=memory_manager.memory,
        chunk_size_limit=512,
        context_window=3900,
    )
