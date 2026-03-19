"""
Multi-query retrieval: automatically rephrase a query in multiple ways
to reduce the risk of missing relevant documents due to wording.

How it works:
1. LLM generates N alternative phrasings of the original query.
2. Each phrasing is used for a separate vector search.
3. Results are deduplicated and combined.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.prompts import PromptTemplate

from shared.mongodb_utils import get_collection

load_dotenv()

VECTOR_INDEX_NAME = "vector_index"

# Custom prompt asking the LLM to generate query variations
MULTI_QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI assistant helping improve document retrieval.
Generate 3 different phrasings of the following question. Each phrasing should
capture the same intent but use different words or angles.

Output only the 3 questions, one per line, with no numbering or extra text.

Original question: {question}

Alternative phrasings:""",
)


def build_multi_query_retriever(k: int = 4):
    """
    Build a MultiQueryRetriever that generates query variants automatically.

    Args:
        k: Number of documents to retrieve per query variant.
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

    base_retriever = vector_store.as_retriever(search_kwargs={"k": k})

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3,  # slight temperature for diverse phrasings
        openai_api_key=os.environ["OPENAI_API_KEY"],
    )

    multi_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm,
        prompt=MULTI_QUERY_PROMPT,
    )

    return multi_retriever


def demo_multi_query():
    """Compare single-query vs multi-query retrieval."""
    query = "How can I improve search relevance in my application?"

    print(f"Query: {query}\n")

    retriever = build_multi_query_retriever(k=3)

    # Multi-query retrieval (deduplicates automatically)
    docs = retriever.invoke(query)

    print(f"Multi-query retrieved {len(docs)} unique documents:\n")
    for doc in docs:
        print(f"  - [{doc.metadata.get('category')}] {doc.metadata.get('title')}")
        print(f"    {doc.page_content[:100]}...\n")


if __name__ == "__main__":
    demo_multi_query()
