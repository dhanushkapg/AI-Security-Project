"""
Web interface for the RAG chatbot built in basicRag.ipynb.

The pipeline itself (models, vector store, prompt) lives in basic_rag.py.

Run with:
    streamlit run app.py
"""

import os

import streamlit as st

import basic_rag

st.set_page_config(page_title="ABC Cloud Assistant", page_icon=":material/support_agent:")


# --- Models and vector store (same as basicRag.ipynb), created once ------
@st.cache_resource
def get_llm():
    return basic_rag.get_llm()


@st.cache_resource(show_spinner="Loading vector store...")
def get_vector_store():
    return basic_rag.load_vector_store()


# --- RAG: retrieve -> build prompt -> stream answer -----------------------
def stream_answer(question, docs):
    for chunk in get_llm().stream(basic_rag.build_messages(question, docs)):
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
    top_k = st.slider("Documents to retrieve", 1, 5, basic_rag.DEFAULT_TOP_K)
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
            results = basic_rag.retrieve(get_vector_store(), prompt, k=top_k)
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
