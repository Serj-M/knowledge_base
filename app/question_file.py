import os
from typing_extensions import List, TypedDict
from langgraph.graph import START, StateGraph
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from fastapi import UploadFile
from langchain_community.document_loaders import DirectoryLoader

from app.rag import llm, vector_store, preamble



# Создаем сообщение для ИИ
async def handler_file(question: str, file: UploadFile) -> str:
    # Убедится что папка существует
    os.makedirs("upload_files", exist_ok=True)
    
    # Сохраняем файл
    save_file = os.path.join("upload_files", file.filename)
    with open(save_file, "wb") as buffer:
        buffer.write(await file.read())

    # Подучаем данные для обработки ИИ 
    loader = DirectoryLoader("upload_files", show_progress=True, use_multithreading=True)
    pages = loader.load()

    # Удаляем файл
    os.remove(save_file)

    # Делим текст на части
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(pages)

    # Добавляем данные в векторную БД
    _ = vector_store.add_documents(documents=all_splits)

    # Создаем граф с зданиями для обработки ИИ
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()

    result = await graph.ainvoke({"question": question})
    return result["answer"]


class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


# Получаем данные из векторной БД
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}

# Обрабатываем данные в локальной llm
def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = preamble.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}
