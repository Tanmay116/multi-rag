from typing import List, Tuple
# from numpy import float32
from app.services.chat import reranker_api, reranker_local
from langchain_core.documents import Document as LCDocument


def rerank_local(
    query: str, retrieved_docs: List[Tuple[LCDocument, float]]
) -> List[Tuple[LCDocument, float]]:
    doc_contents = [doc[0].page_content for doc in retrieved_docs]
    reranked_docs = reranker_local.rank(
        query,
        doc_contents,
        return_documents=False,
        top_k=2,
        batch_size=2
    )
    reranked_docs_with_scores = [
        (
            retrieved_docs[result["corpus_id"]][0],  # type: ignore
            float(result["score"]),
        )
        for result in reranked_docs
    ]
    return reranked_docs_with_scores


def rerank_api(
    query: str, retrieved_docs: List[Tuple[LCDocument, float]]
) -> List[Tuple[LCDocument, float]]:
    doc_contents = [doc[0].page_content for doc in retrieved_docs]
    reranked_docs = reranker_api.rerank(
        model="rerank-v4.0-fast",
        query=query,
        documents=doc_contents,
        top_n=2,
    )
    reranked_docs_with_scores = [
        (retrieved_docs[result.index][0], float(result.relevance_score))
        for result in reranked_docs.results
    ]
    return reranked_docs_with_scores