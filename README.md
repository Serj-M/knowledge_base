# Интеллектуальный Документоориетированный Ассистент (ИДА). 
Использование ИИ и полнотекстового поиска для работы с документацией. 

**Основной функционал:**

    - Чат с ИИ ассистентом, дообученного на внутренней документации.
    - Анализ загруженного документа, с использованием временного хранения в памяти, без сохранения в БД.
    - Полнотекстовый поиск по всей документации, с учётом фильтрации по тэгам, году создания.

**Стек:**

    - Frontend: JS, HTML
    - Backend: Python 3.12, FastApi, LLM, RAG, LangChain, LangSmith, LangGraph
    - DB: ElasticSearch, ChromaDB

**Для локального запуска:**

    pip install
    подключить нужную LLM, по умолчанию используется локальная Llama 3.1 8b (нужно перед запуском скачать в корень проекта, я это делал в корне своего виртуального окружения).