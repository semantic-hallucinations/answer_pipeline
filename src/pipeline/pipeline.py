from llama_index.core.indices.base import BaseIndex
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core import PromptTemplate
from fastapi import Depends

from .dependencies import get_index


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


def create_ingestion_pipeline():
    return IngestionPipeline()



def setup_rag_pipeline(
    index: BaseIndex = Depends(get_index)
):
    return index.as_query_engine(
        text_qa_template=qa_prompt,
        similarity_top_k=7,
        response_mode="compact"
    )
