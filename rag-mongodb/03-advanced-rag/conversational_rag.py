"""
Conversational RAG with chat history.

Maintains a multi-turn conversation where each question is answered using
both the conversation history and retrieved MongoDB documents.

Architecture:
  1. Condense current question + history → standalone question
  2. Retrieve relevant documents for the standalone question
  3. Generate a grounded answer using context + history
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

from shared.mongodb_utils import get_collection

load_dotenv()

VECTOR_INDEX_NAME = "vector_index"

# Step 1: Condense question + history into a standalone question
CONDENSE_PROMPT = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
    ("human",
     "Given the conversation above, rephrase the follow-up question into a "
     "standalone question that can be understood without the history. "
     "If it's already standalone, return it unchanged."),
])

# Step 2: Answer using context + history
QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant. Answer the question using ONLY the context "
     "provided. If the answer is not in the context, say 'I don't know based "
     "on the available information.' Be concise.\n\nContext:\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])


def format_docs(docs) -> str:
    return "\n\n".join(
        f"[{doc.metadata.get('title', 'Untitled')}]\n{doc.page_content}"
        for doc in docs
    )


def build_conversational_rag_chain():
    """
    Build a conversational RAG chain that uses chat history.

    Returns a RunnableWithMessageHistory that accepts:
        - input: {"question": str}
        - config: {"configurable": {"session_id": str}}
    """
    collection = get_collection()

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.environ["OPENAI_API_KEY"],
    )

    vector_store = MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=embeddings,
        index_name=VECTOR_INDEX_NAME,
        embedding_key="embedding",
        text_key="text",
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=os.environ["OPENAI_API_KEY"],
    )

    # Condense step: rewrite follow-up into standalone question
    condense_chain = CONDENSE_PROMPT | llm | StrOutputParser()

    # Full chain
    def get_standalone_question(inputs: dict) -> str:
        if inputs.get("chat_history"):
            return condense_chain.invoke(inputs)
        return inputs["question"]

    rag_chain = (
        RunnablePassthrough.assign(
            standalone_question=RunnableLambda(get_standalone_question)
        )
        | RunnablePassthrough.assign(
            context=lambda x: format_docs(retriever.invoke(x["standalone_question"])),
            question=lambda x: x["standalone_question"],
        )
        | QA_PROMPT
        | llm
        | StrOutputParser()
    )

    # Session store (in-memory; swap for MongoDB-backed store in production)
    session_store: dict[str, ChatMessageHistory] = {}

    def get_session_history(session_id: str) -> ChatMessageHistory:
        if session_id not in session_store:
            session_store[session_id] = ChatMessageHistory()
        return session_store[session_id]

    chain_with_history = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )

    return chain_with_history


def demo_conversation():
    """Simulate a multi-turn conversation."""
    chain = build_conversational_rag_chain()
    session_id = "workshop-demo"

    turns = [
        "What is MongoDB Atlas Vector Search?",
        "What distance metrics does it support?",           # follow-up (needs history)
        "How does that compare to keyword search?",         # follow-up
        "Can I use it with LangChain?",                    # follow-up
    ]

    print("=== Conversational RAG Demo ===\n")
    for question in turns:
        print(f"User: {question}")
        answer = chain.invoke(
            {"question": question},
            config={"configurable": {"session_id": session_id}},
        )
        print(f"AI  : {answer}\n")


if __name__ == "__main__":
    demo_conversation()
