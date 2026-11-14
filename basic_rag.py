"""
The RAG pipeline from basicRag.ipynb, shared by app.py and attack_basicrag.py.

  - chat model:      gpt-4o-mini (temperature 0)
  - embedding model: text-embedding-3-small, 512 dimensions
  - vector store:    InMemoryVectorStore, loaded from data/acme_vector_store.json
                     (built from fake_docs.py if the file doesn't exist yet)
"""

import os
from pathlib import Path

import truststore

# Use the Windows certificate store so HTTPS works behind antivirus/proxy TLS inspection
truststore.inject_into_ssl()

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from fake_docs import DOCUMENTS

load_dotenv()

CHAT_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 512
DEFAULT_TOP_K = 2
VECTOR_STORE_PATH = Path(__file__).parent / "data" / "acme_vector_store.json"

SYSTEM_PROMPT = (
    "You are the Acme Cloud help assistant. "
    "Answer the user's question using only the context below. "
    "If the answer is not in the context, say \"I don't know based on the company documents.\" "
    "Keep answers short."
)


def get_llm() -> ChatOpenAI:
    return ChatOpenAI(model=CHAT_MODEL, temperature=0)


def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        dimensions=EMBEDDING_DIMENSIONS,
        api_key=os.getenv("OPENAI_API_KEY"),
    )


def load_documents() -> list[Document]:
    return [
        Document(page_content=d["text"], metadata={"id": d["id"], "title": d["title"]})
        for d in DOCUMENTS
    ]


def load_vector_store(embeddings: OpenAIEmbeddings | None = None) -> InMemoryVectorStore:
    """Load the saved store, or build it from fake_docs.py and save it."""
    embeddings = embeddings or get_embeddings()
    if VECTOR_STORE_PATH.exists():
        return InMemoryVectorStore.load(str(VECTOR_STORE_PATH), embedding=embeddings)

    store = InMemoryVectorStore.from_documents(load_documents(), embedding=embeddings)
    VECTOR_STORE_PATH.parent.mkdir(exist_ok=True)
    store.dump(str(VECTOR_STORE_PATH))
    return store


def build_messages(question: str, docs: list[Document]) -> list[tuple[str, str]]:
    context = "\n\n".join(f"[{d.metadata['title']}]\n{d.page_content}" for d in docs)
    return [
        ("system", f"{SYSTEM_PROMPT}\n\nContext:\n{context}"),
        ("user", question),
    ]


def retrieve(store: InMemoryVectorStore, question: str, k: int = DEFAULT_TOP_K):
    """Return [(Document, score), ...] for the top-k matches."""
    return store.similarity_search_with_score(question, k=k)
