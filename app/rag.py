import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain import hub
from langchain.prompts import PromptTemplate


# Загружаем переменные из .env
load_dotenv()

# Получаем переменные окружения
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")

# Устанавливаем переменные окружения
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = LANGSMITH_API_KEY
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_PROJECT"] = "pr-majestic-decency-69"

# подключаем модель
llm = ChatOllama(
    model="llama3.1",
    temperature=1
)

# подключаем эмбеддинги для индексации данных перед записью в векторную БД
embeddings = OllamaEmbeddings(model="llama3.1")

# подключаем векторную БД ChromaDB
vector_store = Chroma(embedding_function=embeddings)


# Получаем текущий промпт
prompt = hub.pull("rlm/rag-prompt")

template = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know. Keep the answer concise.
Gives answers in RUSSIAN LANGUAGE!

{context}

Question: {question}

Helpful Answer:"""

preamble = PromptTemplate.from_template(template)
