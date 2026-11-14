"""
Web interface for the RAG chatbot built in basicRag.ipynb.

Uses the same setup as the notebook:
  - chat model:      gpt-4o-mini (temperature 0)
  - embedding model: text-embedding-3-small, 512 dimensions
  - vector store:    InMemoryVectorStore, loaded from data/acme_vector_store.json
                     (built from fake_docs.py if the file doesn't exist yet)

Run with:
    streamlit run app.py
"""

import os
from pathlib import Path

import truststore

# Use the Windows certificate store so HTTPS works behind antivirus/proxy TLS inspection
truststore.inject_into_ssl()

import streamlit as st
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from fake_docs import DOCUMENTS

load_dotenv()

VECTOR_STORE_PATH = Path(__file__).parent / "data" / "acme_vector_store.json"
SYSTEM_PROMPT = (
    "You are the Acme Cloud help assistant. "
    "Answer the user's question using only the context below. "
    "If the answer is not in the context, say \"I don't know based on the company documents.\" "
    "Keep answers short."
)

st.set_page_config(page_title="ABC Cloud Assistant", page_icon=":material/support_agent:")


# --- Models and vector store (same as basicRag.ipynb), created once ------
@st.cache_resource
def get_llm():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)


@st.cache_resource(show_spinner="Loading vector store...")
def get_vector_store():
    embedding_model = OpenAIEmbeddings(
        model="text-embedding-3-small",
        dimensions=512,
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    if VECTOR_STORE_PATH.exists():
        return InMemoryVectorStore.load(str(VECTOR_STORE_PATH), embedding=embedding_model)

    documents = [
        Document(page_content=d["text"], metadata={"id": d["id"], "title": d["title"]})
        for d in DOCUMENTS
    ]
    store = InMemoryVectorStore.from_documents(documents, embedding=embedding_model)
    VECTOR_STORE_PATH.parent.mkdir(exist_ok=True)
    store.dump(str(VECTOR_STORE_PATH))
    return store


# --- RAG: retrieve -> build prompt -> stream answer -----------------------
def stream_answer(question, docs):
    context = "\n\n".join(f"[{d.metadata['title']}]\n{d.page_content}" for d in docs)
    messages = [
        ("system", f"{SYSTEM_PROMPT}\n\nContext:\n{context}"),
        ("user", question),
    ]
    for chunk in get_llm().stream(messages):
        if chunk.content:
            yield chunk.content


def show_sources(sources):
    with st.expander(f"Sources ({len(sources)})", type="compact"):
        for s in sources:
            st.markdown(f"**{s['title']}** · `{s['id']}` · score {s['score']:.3f}")
            st.caption(s["text"])


# --- Page -----------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Documents to retrieve", 1, 5, 2)
    if st.button("Clear chat", icon=":material/delete:", width="stretch"):
        st.session_state.messages = []

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("ABC Cloud Assistant")
st.caption("Ask a question about the company documents.")

if not os.getenv("OPENAI_API_KEY"):
    st.error("Add OPENAI_API_KEY to your .env file.", title="Missing API key")
    st.stop()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            show_sources(msg["sources"])

if prompt := st.chat_input("Type your question", submit_mode="disable"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            results = get_vector_store().similarity_search_with_score(prompt, k=top_k)
            docs = [doc for doc, _ in results]
            answer = st.write_stream(stream_answer(prompt, docs))
            sources = [
                {
                    "id": d.metadata["id"],
                    "title": d.metadata["title"],
                    "text": d.page_content,
                    "score": score,
                }
                for d, score in results
            ]
            show_sources(sources)
        except Exception as e:
            answer, sources = f"Something went wrong: {e}", []
            st.error(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
