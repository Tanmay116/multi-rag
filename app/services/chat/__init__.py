from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from app.core.config import USE_RERANKER, RERANKER_TYPE
from app.db.vector_store import get_vector_store

vector_store = get_vector_store()

model = ChatOllama(
    model="qwen3.5:4b",
    # model="minimax-m2.7:cloud",
)

# model = ChatGroq(
#     model="openai/gpt-oss-120b",
# )
reranker_api = None
reranker_local = None
if USE_RERANKER:
    if RERANKER_TYPE == "local":
        from sentence_transformers import CrossEncoder
        import os

        # Prevent memory fragmentation (set BEFORE model load)
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        # Load the reranker model
        reranker_local = CrossEncoder(
            "jinaai/jina-reranker-v1-turbo-en",
            revision="b8c14f4e723d9e0aab4732a7b7b93741eeeb77c2",
            trust_remote_code=True,
            device="cuda",
        )

        # Convert to FP16 (cuts VRAM significantly)
        reranker_local.model.half()

    else:
        import cohere
        reranker_api = cohere.ClientV2()


