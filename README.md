# ABC Cloud Assistant: a basic RAG chatbot

A small **Retrieval-Augmented Generation (RAG)** chatbot that answers questions about a fictional cloud company's internal documents. The project has two parts:

| Part | What it is |
|------|------------|
| [`basicRag.ipynb`](basicRag.ipynb) | A step-by-step notebook that builds the RAG pipeline: models, documents, embeddings and vector search |
| [`app.py`](app.py) | A Streamlit web chat app that uses the same pipeline, so you can ask questions in the browser |

This is the starting app for the **AI Security** exercises, such as [`02_prompt_injection.ipynb`](02_prompt_injection.ipynb).

---

## How it works

```
 user question
      │
      ▼
 ┌──────────────┐   embed question    ┌────────────────────────┐
 │  Retrieve    │ ──────────────────► │  Vector store (6 docs) │
 │  top-k docs  │ ◄────────────────── │  512-dim embeddings    │
 └──────────────┘   most similar docs └────────────────────────┘
      │
      ▼
 ┌──────────────┐
 │  Generate    │  system prompt + retrieved docs + question → gpt-4o-mini
 └──────────────┘
      │
      ▼
   answer + sources
```

1. **Knowledge base:** six fake company documents in [`fake_docs.py`](fake_docs.py): leave policy, remote work policy, password policy, pricing plans, refund policy and customer support.
2. **Embeddings:** each document is turned into a 512-dimension vector with OpenAI `text-embedding-3-small`.
3. **Vector store:** the vectors are kept in a LangChain `InMemoryVectorStore` and saved to `data/acme_vector_store.json`.
4. **Retrieval:** the question is embedded and the most similar documents are found (2 by default).
5. **Generation:** `gpt-4o-mini` answers **using only the retrieved context**. If the answer isn't there, it replies *"I don't know based on the company documents."*

---

## Project structure

```
AI Security/
├── basicRag.ipynb            # Notebook: build the RAG pipeline step by step
├── app.py                    # Streamlit chat web app
├── fake_docs.py              # The fake company documents (knowledge base)
├── data/
│   └── acme_vector_store.json  # Saved embeddings (reused by the app)
├── rag_core.py               # LangGraph version of the pipeline (used by later notebooks)
├── 02_prompt_injection.ipynb  # Security exercise: prompt injection
├── requirements.txt
└── .env                      # Your OpenAI API key (not committed)
```

---

## Setup

**Requirements:** Python 3.12 and an [OpenAI API key](https://platform.openai.com/api-keys).

```powershell
# 1. Create and activate a virtual environment
python -m venv AISecurity
AISecurity\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

3. Create a `.env` file in the project folder:

```env
OPENAI_API_KEY=sk-...
```

> `.env` is listed in `.gitignore`. Never commit your API key.

---

## Part 1: the notebook (`basicRag.ipynb`)

Open the notebook in VS Code or Jupyter, select the `AISecurity` kernel and run the cells in order:

| Step | What happens |
|------|--------------|
| Load environment | Reads `OPENAI_API_KEY` from `.env` |
| Define chat model | `ChatOpenAI(model="gpt-4o-mini", temperature=0)` |
| Define embedding model | `OpenAIEmbeddings(model="text-embedding-3-small", dimensions=512)` |
| Create documents | Turns `fake_docs.py` into LangChain `Document` objects with `id` and `title` metadata |
| Build vector store | Embeds the documents into an `InMemoryVectorStore` (one small paid API call) |
| Test search | Runs a similarity search for *"How many days of leave do I get?"* and prints scores |

Example output of the test search:

```
0.515 | hr-001 | Leave policy
0.360 | hr-002 | Remote work policy
```

---

## Part 2: the web app (`app.py`)

```powershell
AISecurity\Scripts\activate
streamlit run app.py
```

Then open **http://localhost:8501**.

### Features

- **Chat interface:** type a question at the bottom and the answer is shown as it's written.
- **Sources:** each answer has a *Sources* dropdown with the retrieved documents and their similarity scores.
- **Settings sidebar:** choose how many documents to retrieve (1–5), or clear the chat.
- **Reuses saved embeddings:** loads `data/acme_vector_store.json`. If the file is missing, it rebuilds it from `fake_docs.py` and saves it.
- **Clear errors:** shows a message if the API key is missing or a request fails.

### Try asking

- *How many days of leave do I get?*
- *Can I work from home?*
- *What are the password requirements?*
- *How much does the Pro plan cost?*
- *What are the support hours?*
- *What is the capital of France?* (not in the documents, so it should say it doesn't know)

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Missing API key` in the app | Add `OPENAI_API_KEY` to `.env` and restart the app |
| `CERTIFICATE_VERIFY_FAILED` / `Connection error` | Antivirus or a proxy is inspecting HTTPS. The app calls `truststore.inject_into_ssl()` to use the Windows certificate store. Add the same line at the top of the notebook if it fails there too |
| Slow first answer | The first question loads the vector store and makes two API calls. Later questions are faster |
| Changed `fake_docs.py` but answers don't reflect it | Delete `data/acme_vector_store.json` so it gets rebuilt, then restart the app |

---

## Tech stack

- [LangChain](https://python.langchain.com/) (`langchain-core`, `langchain-openai`): documents, vector store, model wrappers
- [OpenAI](https://platform.openai.com/): `gpt-4o-mini` and `text-embedding-3-small`
- [Streamlit](https://streamlit.io/): web chat interface
- [python-dotenv](https://pypi.org/project/python-dotenv/): loads the API key from `.env`

---

## License

See [LICENSE](LICENSE).
