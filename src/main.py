from fastapi import FastAPI, Header
from pydantic import BaseModel

app = FastAPI()


class TextRequest(BaseModel):
    text: str


@app.post("/length/")
async def calculate_length(
    request: TextRequest, x_multiplier: int = Header(1, alias="multiplier")
):
    length = len(request.text)
    result = length * x_multiplier
    return {"length": length, "result": result}
