from typing import Optional

from llama_index.llms.openrouter import OpenRouter
from pydantic import BaseModel

from ..log_utils import logger
from ..settings import settings


class LLMSettings(BaseModel):
    primary_api_key: str
    backup_api_key: Optional[str] = None
    current_key: str
    model_name: str

    def switch_key(self):
        if self.current_key == self.primary_api_key:
            if self.backup_api_key:
                self.current_key = self.backup_api_key
                logger.info("Switched to backup API key")
                return True
            else:
                logger.warning("Backup API key отсутствует, переключение невозможно")
                return False
        else:
            self.current_key = self.primary_api_key
            logger.info("Switched to primary API key")
            return True


llm_settings = LLMSettings(
    primary_api_key=settings.get_model_key(),
    backup_api_key=settings.get_backup_model_key(),
    current_key=settings.get_model_key(),
    model_name=settings.get_model_name(),
)


def get_llm():
    logger.info(f"Creating LLM with model: {llm_settings.model_name}")
    return OpenRouter(
        model=llm_settings.model_name,
        api_key=llm_settings.current_key,
        max_tokens=10000,
        context_window=128000,
        temperature=0.3,
        timeout=60,
    )
