from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

from app.db.vector_store import get_vector_store

vector_store = get_vector_store()

# model = ChatOllama(
#     model="qwen3.5:4b",
# )

model = ChatGroq(
    model="openai/gpt-oss-120b",
)
