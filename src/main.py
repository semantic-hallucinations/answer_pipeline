from typing import Annotated

from fastapi import Body, Depends, FastAPI
from llama_index.core.chat_engine.types import BaseChatEngine

from .log_utils import logger
from .pipeline import get_chat_engine, get_llm, llm_settings

app = FastAPI()


@app.post("/")
async def main(
    message: Annotated[str, Body()],
    chat_engine: BaseChatEngine = Depends(get_chat_engine),
):
    try:
        streaming_response = await chat_engine.achat(message)
        response = streaming_response.response

        if response.strip().lower() == "empty response":
            logger.warning(
                "Получен 'Empty response' - переключаем API токен и повторяем запрос"
            )
            switched = llm_settings.switch_key()
            if switched:
                new_llm = get_llm()
                chat_engine._llm = new_llm
                streaming_response = await chat_engine.achat(message)
                response = streaming_response.response

        logger.info(f"Тип streaming_response: {type(streaming_response)}")
        logger.info(f"Источники: {streaming_response.sources}")

        sources = []
        for source in streaming_response.sources:
            if hasattr(source, "raw_output"):
                for node in source.raw_output:
                    if hasattr(node, "node") and hasattr(node.node, "metadata"):
                        if "source_url" in node.node.metadata:
                            sources.append(node.node.metadata["source_url"])

        return {"response": response, "source_urls": sources}
    except Exception as e:
        try:
            logger.error(f"Error during chat: {str(e)}")
            switched = llm_settings.switch_key()
            if switched:
                new_llm = get_llm()
                chat_engine._llm = new_llm
                streaming_response = await chat_engine.achat(message)
                response = streaming_response.response

            sources = []
            for source in streaming_response.sources:
                if hasattr(source, "raw_output"):
                    for node in source.raw_output:
                        if hasattr(node, "node") and hasattr(node.node, "metadata"):
                            if "source_url" in node.node.metadata:
                                sources.append(node.node.metadata["source_url"])
            logger.info(f"Тип streaming_response: {type(streaming_response)}")
            logger.info(f"Источники: {streaming_response.sources}")

            return {"response": response, "source_urls": sources}
        except Exception:
            return {
                "response": "Извиняемся, в данный момент бот не доступен",
                "source_urls": ["None"],
            }


@app.post("/switch_api_key")
async def switch_api_key():
    switched = llm_settings.switch_key()
    return {
        "switched": switched,
        "current_key": (
            "backup"
            if llm_settings.current_key == llm_settings.backup_api_key
            else "primary"
        ),
    }
