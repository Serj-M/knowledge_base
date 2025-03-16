import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from app.question import handler

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


class HumanMessage(BaseModel):
    human_message: str

@app.post("/question")
async def question(params: HumanMessage):
    human_message = params.human_message
    return await handler(human_message)


# fastapi dev app/main.py
# uvicorn app.main:app --reload

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
