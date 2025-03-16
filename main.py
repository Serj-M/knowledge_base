import uvicorn
from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel
from typing import Annotated

from app.question import handler
from app.question_file import handler_file

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


class HumanMessage(BaseModel):
    human_message: str

@app.post("/question")
async def question(params: HumanMessage) -> dict:
    human_message = params.human_message
    response: str = await handler(human_message)
    result = {
        "question": human_message,
        "answer": response
    }
    return result


# endpoint for question with file
@app.post("/question_with_file")
async def upload_question_with_file(
    question: str = Form(..., description="Текст вопроса"), 
    file: UploadFile = File(..., description="Загружаемый файл")
) -> dict:
    response: str = await handler_file(question, file)
    result = {
        "question": question,
        "answer": response,
        "filename": file.filename
    }
    return result




# fastapi dev app/main.py
# uvicorn app.main:app --reload

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
