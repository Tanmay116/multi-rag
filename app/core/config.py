import os
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

# LOGGING
LOG_LEVEL = os.environ["LOG_LEVEL"]
LOG_FILE = os.getenv("LOG_FILE", "rag.log")

# LLM
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:4b")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", None)

# RERANKER
USE_RERANKER: bool = True
RERANKER_TYPE: Literal["local", "api"] = "api"
CO_API_KEY = os.getenv("CO_API_KEY")