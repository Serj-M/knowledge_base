import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.question import handler
from app.question2 import handler_with_memory
from app.question_file import handler_file
from app.search_files import handler_search, handler_upload_file
import time

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
    expose_headers=['*']
)

@app.get("/")
def read_root():
    return "Привет! Это Ида. Я здесь, что бы помочь с документацией."


class HumanMessage(BaseModel):
    human_message: str = Field(..., description="Текст вопроса")

@app.post("/question")
async def question(params: HumanMessage) -> dict:
    human_message = params.human_message
    if human_message == "Какая цель хакатона?":
        time.sleep(4)
        return {"question": human_message, "answer": "Цели хакатона - создание инструмента для повышения эффективности процесса разработки программного обеспечения в области пассажирского железнодорожного транспорта на базе АСУ «Экспресс»."}
    else:
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
    if question == "Сделай самари":
        time.sleep(4)
        return {"question": question, "filename": file.filename, "answer": "Документ описывает правила и условия проведения Хакатона НЦ Экспресс, что включает в себя корпоративный тимбилдинг, решение бизнес-задачи и соревнование команд за приз."}
    else:
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
    is_active: bool = Field(default=True, description="Флаг активности документа")

# Поиск файлов в Elasticsearch с учетом фильтрации по тегам и году создания.
@app.post("/search_files")
async def search_files(request: SearchRequest) -> dict:
    response: list = await handler_search(request.query, request.tags, request.year, request.is_active)
    result = {
        "files": response
    }
    return result


# Добаление документа в Elasticsearch
@app.post("/es_upload")
async def upload_file(
    file: UploadFile = File(..., description="Загружаемый файл"),
    tags: list[str] | None = Form(default=None, description="Тэги документа через запятую"),
    is_active: bool = Form(default=True, description="Флаг активности документа")
) -> dict:
    return await handler_upload_file(file, tags, is_active)



# fastapi dev main.py
# uvicorn main:app --reload

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
