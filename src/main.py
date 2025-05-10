from typing import Annotated, Any, Dict, List

from fastapi import Body, Depends, FastAPI
from llama_index.core.chat_engine.types import BaseChatEngine

from .log_utils import logger
from .pipeline import get_chat_engine, get_llm, llm_settings

app = FastAPI()


def extract_sources(streaming_response: Any) -> List[str]:
    sources = []
    for source in streaming_response.sources:
        if hasattr(source, "raw_output"):
            for node in source.raw_output:
                if hasattr(node, "node") and hasattr(node.node, "metadata"):
                    if "source_url" in node.node.metadata:
                        sources.append(node.node.metadata["source_url"])
    return sources


async def ask_chat_engine(chat_engine: BaseChatEngine, message: str) -> Dict[str, Any]:
    streaming_response = await chat_engine.achat(message)
    response = streaming_response.response
    sources = extract_sources(streaming_response)
    logger.info(f"Тип streaming_response: {type(streaming_response)}")
    logger.info(f"Источники: {streaming_response.sources}")
    return {
        "response": response,
        "source_urls": sources,
        "streaming_response": streaming_response,
    }


@app.post("/")
async def main(
    message: Annotated[str, Body()],
    chat_engine: BaseChatEngine = Depends(get_chat_engine),
):
    try:
        result = await ask_chat_engine(chat_engine, message)

        if result["response"].strip().lower() == "empty response":
            logger.warning(
                "Получен 'Empty response' - переключаем API токен и повторяем запрос"
            )
            if llm_settings.switch_key():
                chat_engine._llm = get_llm()
                result = await ask_chat_engine(chat_engine, message)

        return {"response": result["response"], "source_urls": result["source_urls"]}

    except Exception as e:
        logger.error(f"Error during chat: {str(e)}")
        try:
            if llm_settings.switch_key():
                chat_engine._llm = get_llm()
                result = await ask_chat_engine(chat_engine, message)
                return {
                    "response": result["response"],
                    "source_urls": result["source_urls"],
                }
        except Exception:
            pass

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
