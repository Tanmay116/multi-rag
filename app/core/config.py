import os

from dotenv import load_dotenv

load_dotenv()

# LOGGING
LOG_LEVEL = os.environ["LOG_LEVEL"]
LOG_FILE = os.getenv("LOG_FILE", "rag.log")

# LLM
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:4b")