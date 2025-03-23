from langchain_core.messages import AIMessage
from typing_extensions import List, TypedDict
from langgraph.graph import START, END, StateGraph, MessagesState
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode, tools_condition
from IPython.display import Image, display
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from app.rag import llm, vector_store, embeddings, State, get_template # preamble



# Создаем сообщение для ИИ
async def handler_with_memory(human_message: str) -> str:
    loader = DirectoryLoader("docs", show_progress=True, use_multithreading=True)
    pages = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(pages)
    _ = vector_store.add_documents(documents=all_splits)

    # graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder = StateGraph(MessagesState)
    # graph_builder.add_edge(START, "retrieve")
    graph_builder.add_node(query_or_respond)
    graph_builder.add_node(tools)
    graph_builder.add_node(generate)

    graph_builder.set_entry_point("query_or_respond")
    graph_builder.add_conditional_edges(
        "query_or_respond",
        tools_condition,
        {END: END, "tools": "tools"},
    )
    graph_builder.add_edge("tools", "generate")
    graph_builder.add_edge("generate", END)
    # graph = graph_builder.compile()
    memory = MemorySaver()
    graph = graph_builder.compile(checkpointer=memory)

    # Specify an ID for the thread
    config = {"configurable": {"thread_id": "abc123"}}

    display(Image(graph.get_graph().draw_mermaid_png()))

    for step in graph.stream(
        {"messages": [{"role": "user", "content": human_message}]},
        stream_mode="values",
        config=config,
    ):
        step["messages"][-1].pretty_print()

    # result = await graph.ainvoke({"question": human_message})
    # return result


# Преобразовать загруженные документы в строки, объединяя их содержимое, игнорируя метаданные
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_employee_position() -> str:
    # определение роли пользователя из учётной записи портала
    return "Программист"


# def retrieve(state: State):
#     retrieved_docs = vector_store.similarity_search(state["question"])
#     return {"context": retrieved_docs}

@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs


# def generate(state: State):
#     docs_content: str = "\n\n".join(doc.page_content for doc in state["context"])
#     employee_position: str = get_employee_position()
#     template: str = get_template(employee_position)
#     print(f"Prompt: {template}")
#     preamble: PromptTemplate = PromptTemplate.from_template(template)
#     messages = preamble.invoke(
#         {"question": state["question"], "context": docs_content, "employee_position": employee_position})
#     response = llm.invoke(messages)
#     return {"answer": response.content}

# Step 1: Generate an AIMessage that may include a tool-call to be sent.
def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond."""
    llm_with_tools = llm.bind_tools([retrieve])
    response = llm_with_tools.invoke(state["messages"])
    # MessagesState appends messages to state instead of overwriting
    return {"messages": [response]}


# Step 2: Execute the retrieval.
tools = ToolNode([retrieve])


# Step 3: Generate a response using the retrieved content.
def generate(state: MessagesState):
    """Generate answer."""
    # Get generated ToolMessages
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]

    # Format into prompt
    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    system_message_content = (
        "You are an assistant for question-answering tasks." 
        "Use the following pieces of retrieved context to answer the question."
        "If you don't know the answer, just say that you don't know. Keep the answer concise."
        f"In your answer, keep in mind that the person you are answering holds a position as a {get_employee_position()} at a research institute."
        "Gives answers in RUSSIAN LANGUAGE!"
        "\n\n"
        f"{docs_content}"
    )
    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]
    prompt = [SystemMessage(system_message_content)] + conversation_messages

    # Run
    response = llm.invoke(prompt)
    return {"messages": [response]}
