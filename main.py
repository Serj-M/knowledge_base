import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, Query
from pydantic import BaseModel, Field

from app.question import handler
from app.question_file import handler_file
from app.search_files import handler_search, handler_upload_file

app = FastAPI()


@app.get("/")
def read_root():
    return {"It`s": "AI Ida"}


class HumanMessage(BaseModel):
    human_message: str = Field(..., description="Текст вопроса")

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


class SearchRequest(BaseModel):
    query: str = Field(..., description="Поисковый запрос по имени файла или содержимому", max_length=300)
    tags: list[str] | None = Field(..., description="Тег документа для точного поиска", max_length=30)
    year: str | None = Field(default=None, description="Год создания файла", max_length=4)

# Поиск файлов в Elasticsearch с учетом фильтрации по тегам и году создания.
@app.post("/search_files")
async def search_files(request: SearchRequest) -> dict:
    response: list = await handler_search(request.query, request.tags, request.year)
    result = {
        "files": response
    }
    return result


# Добаление документа в Elasticsearch
@app.post("/es_upload")
async def upload_file(
    file: UploadFile = File(..., description="Загружаемый файл"),
    tags: list[str] | None = Form(default=None, description="Тэги документа через запятую")
) -> dict:
    return await handler_upload_file(file, tags)



# fastapi dev app/main.py
# uvicorn app.main:app --reload

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
