import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

OPEN_AI_KEY = os.getenv("OPENAI_API_KEY")

# define chat model
chat_model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=OPEN_AI_KEY,
)

# define embedding model
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    dimensions=512,  # shorter vectors; any value up to 1536 (the default)
    api_key=OPEN_AI_KEY,
)


if __name__ == "__main__":
    # quick check: embed a test string and show the vector length
    vector = embedding_model.embed_query("hello world")
    print("Embedding length:", len(vector))
