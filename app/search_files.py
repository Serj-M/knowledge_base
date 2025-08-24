import datetime
import os
import io
import uuid
import base64
from elasticsearch import AsyncElasticsearch
from dotenv import load_dotenv
from fastapi import UploadFile
from docx import Document


# Загружаем переменные из .env
load_dotenv()

# Получаем переменные окружения
DOCUMENTS_HOST=os.getenv("DOCUMENTS_HOST")
DOCUMENTS_LOGIN=os.getenv("DOCUMENTS_LOGIN")
DOCUMENTS_PASSWORD=os.getenv("DOCUMENTS_PASSWORD")
DOCUMENTS_CERT=os.getenv("DOCUMENTS_CERT")
index='hackathon'

# Initialize Elasticsearch connection if host is available
es = None
if DOCUMENTS_HOST:
    es = AsyncElasticsearch(
        hosts=DOCUMENTS_HOST,
        basic_auth=(DOCUMENTS_LOGIN, DOCUMENTS_PASSWORD),
        ca_certs=DOCUMENTS_CERT,
        request_timeout=300
    )


async def handler_search(query: str, tags: list[str] | None, year: str | None, is_active: bool) -> list:
    if not es:
        return {"results": [], "message": "Elasticsearch not configured"}
    
    clauses = []
    filters = []
    if query:
        clauses.append({
            "multi_match": {
                "query": query,
                "fields": ["filename", "content"]
            }
        })
    
    if tags:
        # clauses.append({
        #     "terms": {"tags": tags}  # Строгий поиск по тегам
        # })
        filters.append({
            "term": {"tags": tags[0]}
        })
    
    if year:
        # clauses.append({
        #     "term": {"created_at": year}
        # })
        filters.append({
            "term": {"created_at": year}
        })
    
    if is_active:
        filters.append({
            "term": {"is_active": is_active}
        })

    search_query = {"query": {"bool": {"must": clauses, "filter": filters}}}
    
    try:
        if not await es.indices.exists(index=index):
            await es.indices.create(index=index, body=mapping)

        response = await es.search(index=index, body=search_query)
        hits = response.get("hits", {}).get("hits", [])
        return {
            "results": [{
                "id": hit["_id"], 
                "filename": hit["_source"].get("filename", ""), 
                "tags": hit["_source"].get("tags", ""),
                "created_at": hit["_source"].get("created_at", ""),
                "is_active": hit["_source"].get("is_active", "")
            } for hit in hits]
        }
    except Exception as e:
        return {"results": [], "error": str(e)}
    

mapping = {
    "mappings": {
        "properties": {
            "filename": {
                "type": "text",
                "fields": {
                    "keyword": {
                        "type": "keyword"
                    }
                }
            },
            "content": {
                "type": "text"
            },
            "tags": {
                "type": "keyword"
            },
            "created_at": {
                "type": "keyword"
            },
            "is_active": {
                "type": "boolean"
            },
        }
    }
}


# Добавление документа в Elasticsearch
async def handler_upload_file(file: UploadFile, tags: list[str] | None, is_active: bool = True) -> dict:
    if not es:
        return {"message": "Elasticsearch not configured", "file_id": None}
    
    file_id = str(uuid.uuid4())  # Генерация ID

    content = await file.read()
    # получаем текст из docx файла
    doc = Document(io.BytesIO(content))
    text_doc = "\n".join([p.text for p in doc.paragraphs])

    file_name = file.filename

    document = {
        "filename": file_name,
        "content": text_doc,  # Текст файла
        "tags": tags if tags is not None else [],
        "created_at": str(datetime.datetime.now().year),
        "is_active": is_active
    }

    try:
        if not await es.indices.exists(index=index):
            await es.indices.create(index=index, body=mapping)

        await es.index(index=index, id=file_id, document=document)
        return {"message": "Файл загружен", "file_id": file_id}
    except Exception as e:
        return {"message": f"Error uploading file: {str(e)}", "file_id": None}
