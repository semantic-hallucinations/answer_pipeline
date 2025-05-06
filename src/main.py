import logging
import pathlib
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index.core.chat_engine.types import BaseChatEngine
from llama_index.core import Settings
from llama_index.llms.openrouter import OpenRouter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from fastapi import FastAPI, Depends

from settings import settings
from pipeline.dependencies import get_chat_engine


Settings.llm = OpenRouter(
    model=settings.get_model_name(),
    api_key=settings.get_model_key(),
)
Settings.embed_model = HuggingFaceEmbedding(
    model_name="thenlper/gte-large",

)

app = FastAPI()


def get_mock_index():
    from llama_index.core import SimpleKeywordTableIndex, SimpleDirectoryReader

    documents = SimpleDirectoryReader("data").load_data()
    return SimpleKeywordTableIndex.from_documents(documents) 

from pipeline.dependencies import get_index
app.dependency_overrides[get_index] = get_mock_index

@app.get("/")
def main(
    message: str,
    chat_engine: BaseChatEngine = Depends(get_chat_engine)
):
    return chat_engine.chat(message) 
