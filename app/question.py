from langchain_core.messages import AIMessage
from typing_extensions import List, TypedDict
from langgraph.graph import START, StateGraph
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from app.rag import llm, vector_store, preamble, embeddings



# Создаем сообщение для ИИ
async def handler(human_message: str) -> str:
    # file_path = ("docs/Хакатон 2_описание задания.pdf")
    # loader = PyPDFLoader(file_path)
    # pages = []
    # async for page in loader.alazy_load():
    #     pages.append(page)
    loader = DirectoryLoader("docs", show_progress=True, use_multithreading=True)
    pages = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(pages)
    _ = vector_store.add_documents(documents=all_splits)

    # vectorstore = Chroma.from_documents(documents=all_splits, embedding=embeddings)

    # template = """You are an assistant for question-answering tasks.
    # Use the following pieces of retrieved context to answer the question.
    # If you don't know the answer, just say that you don't know. Keep the answer concise.
    # Gives answers in RUSSIAN LANGUAGE!

    # <context>
    # {context}
    # </context>

    # Answer the following question:

    # {question}"""

    # prompt = ChatPromptTemplate.from_template(template)

    # chain = (
    #     RunnablePassthrough.assign(context=lambda input: format_docs(input["context"]))
    #     | prompt 
    #     | llm 
    #     | StrOutputParser()
    # )
    # docs = vectorstore.similarity_search(human_message)
    # result = chain.invoke({"context": docs, "question": human_message})

    # retriever = vectorstore.as_retriever()
    # qa_chain = (
    #     {"context": retriever | format_docs, "question": RunnablePassthrough()}
    #     | prompt
    #     | llm
    #     | StrOutputParser()
    # )
    # result = qa_chain.invoke(human_message)
    # return result

    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()

    result = await graph.ainvoke({"question": human_message})
    return result["answer"]


# Преобразовать загруженные документы в строки, объединяя их содержимое, игнорируя метаданные
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = preamble.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}
