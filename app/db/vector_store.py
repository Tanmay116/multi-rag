import os
from typing import List
from uuid import uuid4

import faiss
import portalocker
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from app.core.logger import get_logger

logger = get_logger("vector_store")

INDEX_DIRECTORY = "faiss_index"
LOCK_FILE = "faiss_index/.faiss_lock"
LOCK_TIMEOUT = 30


def update_faiss_index(documents: List[Document]) -> bool:
    if not documents:
        logger.warning("update_faiss_index_empty_list")
        return False

    os.makedirs(INDEX_DIRECTORY, exist_ok=True)

    try:
        # 1. Acquire the File Lock
        # timeout=10 means it will wait 10 seconds for the other process to finish before failing
        with portalocker.Lock(LOCK_FILE, timeout=LOCK_TIMEOUT, mode="w") as _:
            logger.info("acquired_index_lock", extra={"process_id": os.getpid()})

            embeddings = HuggingFaceEmbeddings(
                model_name="BAAI/bge-base-en-v1.5",
                encode_kwargs={"normalize_embeddings": True},
            )

            uuids = [str(uuid4()) for _ in range(len(documents))]

            # 2. Check for existing index
            if os.path.exists(os.path.join(INDEX_DIRECTORY, "index.faiss")):
                vector_store = FAISS.load_local(
                    INDEX_DIRECTORY, embeddings, allow_dangerous_deserialization=True
                )
                vector_store.add_documents(documents=documents, ids=uuids)
            else:
                # Initialize new
                sample_dim = len(embeddings.embed_query("test"))
                index = faiss.IndexFlatL2(sample_dim)
                vector_store = FAISS(
                    embedding_function=embeddings,
                    index=index,
                    docstore=InMemoryDocstore(),
                    index_to_docstore_id={},
                )
                vector_store.add_documents(documents=documents, ids=uuids)

            # 3. Save while still holding the lock
            vector_store.save_local(INDEX_DIRECTORY)

            logger.info(
                "faiss_index_saved_successfully", extra={"docs_added": len(documents)}
            )

        # Lock is automatically released here after the 'with' block
        return True

    except portalocker.exceptions.LockException:
        logger.error(
            "lock_timeout_error",
            extra={"detail": f"Another process is holding the index lock for more than {LOCK_TIMEOUT}s"},
        )
        return False
    except Exception as e:
        logger.exception("unexpected_error_in_vector_store", extra={"error": str(e)})
        return False
