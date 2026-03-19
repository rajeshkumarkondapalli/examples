"""
Step 2 — Build the RAG chain: retrieve context, augment prompt, generate answer.
"""

import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel


# --- Prompt template -------------------------------------------------------

RAG_PROMPT = ChatPromptTemplate.from_template(
    """You are a helpful assistant. Answer the question using ONLY the context
provided below. If the answer is not in the context, say "I don't know based
on the available information."

Context:
{context}

Question: {question}

Answer:"""
)


# --- Helpers ----------------------------------------------------------------

def format_docs(docs) -> str:
    """Concatenate retrieved document contents into a single context string."""
    return "\n\n".join(
        f"[{doc.metadata.get('title', 'Untitled')}]\n{doc.page_content}"
        for doc in docs
    )


# --- Chain builder ----------------------------------------------------------

def build_rag_chain(
    collection,
    index_name: str = "vector_index",
    k: int = 3,
):
    """
    Build and return a LangChain RAG chain.

    Args:
        collection: pymongo Collection with documents and vector index.
        index_name: Name of the Atlas Vector Search index.
        k: Number of documents to retrieve per query.

    Returns:
        A runnable chain: question (str) → answer (str)
    """
    # 1. Embedding model (must match the one used during ingestion)
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.environ["OPENAI_API_KEY"],
    )

    # 2. Vector store retriever
    vector_store = MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=embeddings,
        index_name=index_name,
        embedding_key="embedding",
        text_key="text",
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": k})

    # 3. LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=os.environ["OPENAI_API_KEY"],
    )

    # 4. Assemble the chain using LangChain Expression Language (LCEL)
    #    retrieve → format → augment → generate
    chain = (
        RunnableParallel(
            context=retriever | format_docs,
            question=RunnablePassthrough(),
        )
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    return chain, retriever
