import asyncio
import sys
from typing import Callable

import tiktoken
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from crawl4ai.models import CrawlResultContainer
from langchain_core.documents import Document as LCDocument
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from app.core.logger import get_logger
from app.db.vector_store import update_faiss_index

logger = get_logger("webpage_ingestion")

# Windows requires the Proactor event loop for Playwright subprocesses.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def ingest_webpage_flow(url: str, description: str, max_page: int = 5) -> dict:
    """Crawl a website and ingest all successfully scraped pages into the FAISS index.

    The function streams crawl results page-by-page.  Each successfully crawled
    page is immediately chunked and ingested so that memory usage stays bounded
    even when ``max_page`` is large.

    Args:
        url: The root URL to start crawling from.
        description: Human-readable description of the knowledge-base entry.
            Currently stored for audit purposes; passed through to callers.
        max_page: Maximum number of pages the BFS crawler is allowed to visit.
            Defaults to 5.

    Returns:
        A summary dict with keys ``"ingested"`` (count of pages successfully
        indexed) and ``"failed"`` (count of pages that errored or were skipped).

    Raises:
        RuntimeError: If the crawler itself fails to initialise or the async
            stream raises an unrecoverable error.
    """
    logger.info(
        "Starting web ingestion",
        extra={"url": url, "max_page": max_page},
    )

    browser_config = BrowserConfig(
        headless=False,
        user_agent_mode="random",
        verbose=True,
        browser_mode="dedicatedr",
        viewport_height=600,
        viewport_width=600,
    )

    crawler_config = CrawlerRunConfig(
        deep_crawl_strategy=BFSDeepCrawlStrategy(
            max_depth=0, max_pages=max_page, include_external=False
        ),
        mean_delay=2,  # delay between requests to avoid overwhelming the server in seconds
        max_range=3,  # max range for random delays in seconds
        semaphore_count=1,  # number of concurrent requests
        user_agent_mode="random",  # automatically rotate through realistic browser User-Agents for each request
        scan_full_page=True,  # this will scroll the page to load dynamic content
        override_navigator=True,  # this will override the navigator properties to avoid bot detection
        simulate_user=True,  # this will simulate mouse movements and clicks
        locale="en-US",
        wait_for_images=False,  # wait for images to load before extracting content
        delay_before_return_html=1,  # wait for 1 seconds before returning the HTML content
        timezone_id="America/New_York",
        excluded_tags=[
            "style",
            "dialog",
            "script",
            "header",
            "footer",
            "site-footer",
            "nav",
        ],
        # cache_mode=CacheMode.DISABLED,
        remove_forms=True,
        exclude_all_images=True,
        remove_overlay_elements=False,
        # word_count_threshold=50,
        # markdown_generator=clean_markdown_generator,
        # magic=True,
        score_links=False,
        # wait_for="networkidle",
        stream=True,
    )

    ingested_count = 0
    failed_count = 0

    try:
        async with AsyncWebCrawler(config=browser_config) as crawler:
            logger.debug(
                "Crawler initialised — streaming pages",
                extra={"url": url},
            )
            crawler_iterator = await crawler.arun(url, config=crawler_config)

            async for result in crawler_iterator:
                if not result.success:
                    logger.warning(
                        "Crawl failed for page",
                        extra={
                            "page_url": result.url,
                            "error": getattr(result, "error_message", "unknown"),
                        },
                    )
                    failed_count += 1
                    continue

                logger.info(
                    "Page crawled successfully",
                    extra={"page_url": result.url},
                )

                try:
                    chunks = process_url(result)  # type: ignore
                except Exception:
                    logger.error(
                        "Failed to process page into chunks",
                        extra={"page_url": result.url},
                        exc_info=True,
                    )
                    failed_count += 1
                    continue

                if not chunks:
                    logger.warning(
                        "Processing produced zero chunks — skipping index update",
                        extra={"page_url": result.url},
                    )
                    failed_count += 1
                    continue

                try:
                    update_faiss_index(chunks)
                    logger.info(
                        "FAISS index updated",
                        extra={"page_url": result.url, "chunk_count": len(chunks)},
                    )
                    ingested_count += 1
                except Exception:
                    logger.error(
                        "Failed to update FAISS index",
                        extra={"page_url": result.url},
                        exc_info=True,
                    )
                    failed_count += 1

    except Exception:
        logger.error(
            "Unrecoverable error during web crawl",
            extra={"url": url},
            exc_info=True,
        )
        raise RuntimeError(
            f"Web ingestion flow failed for url='{url}'. "
            "Check logs for details."
        )

    logger.info(
        "Web ingestion complete",
        extra={"url": url, "ingested": ingested_count, "failed": failed_count},
    )

    status = "success"
    if ingested_count == 0:
        status = "failed"
    elif failed_count > 0:
        status = "partial_success"

    return {
        "status": status,
        "result": {
            "ingested": ingested_count,
            "failed": failed_count,
        }
    }


# ---------------------------------------------------------------------------
# Internal chunking helpers
# ---------------------------------------------------------------------------


def split_into_markdown_headers(url: CrawlResultContainer) -> list[LCDocument]:
    """Split a crawled page's markdown into header-delimited chunks.

    Uses :class:`MarkdownHeaderTextSplitter` to split on ``#``, ``##``, and
    ``###`` headings.  Each resulting :class:`LCDocument` receives the source
    URL injected into its ``metadata["url"]`` field so that downstream steps
    can use it for context and deduplication.

    Args:
        url: A successfully crawled page result from Crawl4AI.  The
            ``url.markdown`` and ``url.url`` attributes are used.

    Returns:
        A list of :class:`LCDocument` objects, one per Markdown section.
        Returns an empty list if the page has no parseable content.

    Raises:
        ValueError: If ``url.markdown`` is ``None`` or empty.
    """
    markdown_document = url.markdown
    actual_url = url.url

    if not markdown_document:
        raise ValueError(
            f"Page at '{actual_url}' returned empty markdown — nothing to split."
        )

    logger.debug(
        "Splitting markdown by headers",
        extra={"url": actual_url, "content_length": len(markdown_document)},
    )

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
    md_header_splits = markdown_splitter.split_text(markdown_document)

    for doc in md_header_splits:
        doc.metadata = {"url": actual_url, **doc.metadata}

    logger.debug(
        "Header splitting complete",
        extra={"url": actual_url, "section_count": len(md_header_splits)},
    )
    return md_header_splits


def _split_oversized_docs(
    docs: list[LCDocument],
    splitter: RecursiveCharacterTextSplitter,
    length_function: Callable[[str], int],
) -> list[LCDocument]:
    """Split any document whose token count exceeds the splitter's chunk size.

    Documents already within the size budget are passed through unchanged.
    Token counting is delegated to ``length_function`` so that the check is
    consistent with the splitter's own measurement.

    Args:
        docs: Input list of documents (typically from
            :func:`split_into_markdown_headers`).
        splitter: A configured :class:`RecursiveCharacterTextSplitter` whose
            ``_chunk_size`` property defines the token budget.
        length_function: A callable that accepts a string and returns its token
            count.  Must be the same function passed to *splitter* so the
            gating condition and the split behaviour are consistent.

    Returns:
        A new list of documents where no document exceeds ``splitter._chunk_size``
        tokens.
    """
    split_docs: list[LCDocument] = []
    n_original = len(docs)

    for doc in docs:
        token_count = length_function(doc.page_content)
        if token_count > splitter._chunk_size:
            logger.debug(
                "Document exceeds chunk size - splitting",
                extra={
                    "token_count": token_count,
                    "chunk_size": splitter._chunk_size,
                    "url": doc.metadata.get("url", "unknown"),
                },
            )
            sub_chunks = splitter.split_documents([doc])
            split_docs.extend(sub_chunks)
        else:
            split_docs.append(doc)

    logger.debug(
        "Oversized doc splitting complete",
        extra={"input_docs": n_original, "output_docs": len(split_docs)},
    )
    return split_docs


def _merge_small_docs(
    docs: list[LCDocument],
    merge_threshold: int,
    length_function: Callable[[str], int],
) -> list[LCDocument]:
    """Merge consecutive under-threshold documents into larger, cohesive chunks.

    Merging is constrained to documents that share the same source URL so that
    content from different pages is never combined.  When multiple small
    documents are merged, their ``Header*`` metadata keys are unioned with
    later (more-specific) headers overriding earlier ones.

    Args:
        docs: List of documents to process.  Usually the output of
            :func:`_split_oversized_docs`.
        merge_threshold: Minimum token count a document should reach before it
            is considered self-sufficient.  Documents with fewer tokens are
            buffered and merged with their neighbours.
        length_function: A callable returning the token count of a string.

    Returns:
        A new list of documents where small chunks have been merged where
        possible.
    """
    merged_chunks: list[LCDocument] = []
    doc_buffer: list[LCDocument] = []

    def flush_buffer() -> None:
        """Flush the current buffer into a single merged document."""
        if not doc_buffer:
            return

        merged_content = "\n\n".join(d.page_content for d in doc_buffer)

        # Start with the first doc's metadata (carries the URL),
        # then update Header* keys from subsequent docs so the most
        # specific heading context is preserved.
        merged_metadata = doc_buffer[0].metadata.copy()
        for d in doc_buffer[1:]:
            for k, v in d.metadata.items():
                if k.startswith("Header"):
                    merged_metadata[k] = v

        new_doc = LCDocument(page_content=merged_content, metadata=merged_metadata)
        merged_chunks.append(new_doc)
        doc_buffer.clear()

    for doc in docs:
        # Flush the buffer when the source URL changes to prevent
        # cross-page contamination.
        if doc_buffer and doc.metadata.get("url") != doc_buffer[0].metadata.get("url"):
            logger.debug(
                "URL boundary encountered — flushing merge buffer",
                extra={
                    "previous_url": doc_buffer[0].metadata.get("url"),
                    "new_url": doc.metadata.get("url"),
                },
            )
            flush_buffer()

        token_count = length_function(doc.page_content)
        if token_count >= merge_threshold:
            # Doc is large enough to stand alone; flush any buffered smalls first.
            flush_buffer()
            merged_chunks.append(doc)
        else:
            doc_buffer.append(doc)

    flush_buffer()  # Drain whatever remains after the loop.

    logger.debug(
        "Small doc merging complete",
        extra={
            "input_docs": len(docs),
            "output_docs": len(merged_chunks),
            "merge_threshold": merge_threshold,
        },
    )
    return merged_chunks


def chunk_markdown_splits(
    md_header_splits: list[LCDocument],
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    merge_threshold: int = 256,
) -> list[LCDocument]:
    """Regulate document sizes via a split-then-merge pipeline.

    **Step 1 — Split**: Any document larger than ``chunk_size`` tokens is
    recursively split using :class:`RecursiveCharacterTextSplitter`.  The
    splitter automatically applies ``chunk_overlap`` between resulting
    sub-chunks, preserving continuity within a section.

    **Step 2 — Merge**: Consecutive documents smaller than ``merge_threshold``
    tokens (from the same source URL) are merged into a single document.  This
    avoids ingesting tiny, low-signal chunks that hurt retrieval quality.

    Note:
        Token counting uses OpenAI's ``cl100k_base`` tokenizer via
        :mod:`tiktoken` so that chunk sizes are meaningful for embedding models
        that share that vocabulary.

    Args:
        md_header_splits: Documents produced by :func:`split_into_markdown_headers`.
        chunk_size: Target maximum token count per chunk.  Defaults to 512.
        chunk_overlap: Number of tokens to overlap between consecutive
            sub-chunks when a large document is split.  Defaults to 50.
        merge_threshold: Chunks with fewer than this many tokens are candidates
            for merging.  Defaults to 256.

    Returns:
        Final list of size-regulated :class:`LCDocument` objects ready for
        context prepending and vector indexing.

    Raises:
        ValueError: If ``md_header_splits`` is empty.
    """
    if not md_header_splits:
        raise ValueError("chunk_markdown_splits received an empty document list.")

    logger.debug(
        "Chunk markdown splits started",
        extra={
            "input_docs": len(md_header_splits),
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "merge_threshold": merge_threshold,
        },
    )

    tokenizer = tiktoken.get_encoding("cl100k_base")

    def tiktoken_len(text: str) -> int:
        return len(tokenizer.encode(text))

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=tiktoken_len,
    )

    split_docs = _split_oversized_docs(md_header_splits, text_splitter, tiktoken_len)
    final_chunks = _merge_small_docs(split_docs, merge_threshold, tiktoken_len)

    logger.debug(
        "Chunk markdown splits complete",
        extra={"input_docs": len(md_header_splits), "output_chunks": len(final_chunks)},
    )
    return final_chunks


def prepend_context_to_docs(docs: list[LCDocument]) -> list[LCDocument]:
    """Enrich each chunk's content by prepending source URL and header breadcrumb.

    The prepended text takes the form::

        Link: https://example.com/page
        # Top-level heading
        ## Sub-heading

        <original chunk content>

    This makes the source and structural context self-contained within the
    chunk text itself, which improves retrieval grounding without relying on
    metadata look-ups at query time.

    Note:
        This function mutates ``page_content`` in-place.  Call it **after**
        all size-based splitting and merging is complete, as the prepended text
        is *not* counted against the token budget during chunking.

    Args:
        docs: List of documents to enrich.  Each document's metadata is
            expected to contain a ``"url"`` key and zero or more
            ``"Header N"`` keys populated by :func:`split_into_markdown_headers`.

    Returns:
        The same list with ``page_content`` updated on every document.

    Raises:
        ValueError: If ``docs`` is empty.
    """
    if not docs:
        raise ValueError("prepend_context_to_docs received an empty document list.")

    for doc in docs:
        header_md = "\n".join(
            f"{'#' * int(k.split()[-1])} {v}"
            for k, v in doc.metadata.items()
            if v and k.startswith("Header")
        )

        source_url = doc.metadata.get("url")
        if not source_url:
            logger.warning(
                "Document is missing 'url' metadata — context prefix will be incomplete",
                extra={"content_preview": doc.page_content[:80]},
            )
            source_url = "URL not available"

        doc.page_content = f"Link: {source_url}\n{header_md}\n\n{doc.page_content}"

    logger.debug(
        "Context prepended to documents",
        extra={"doc_count": len(docs)},
    )
    return docs


def process_url(url: CrawlResultContainer) -> list[LCDocument]:
    """Run the full chunking pipeline for a single crawled page.

    Orchestrates the three-step pipeline:

    1. :func:`split_into_markdown_headers` — structural split by headings.
    2. :func:`chunk_markdown_splits` — token-based size regulation.
    3. :func:`prepend_context_to_docs` — context enrichment.

    Args:
        url: A successfully crawled page result from Crawl4AI.

    Returns:
        A list of context-enriched :class:`LCDocument` objects ready for
        ingestion into the vector store.  Returns an empty list if the page
        yields no usable chunks.

    Raises:
        ValueError: Propagated from sub-functions if the page markdown is
            empty or the pipeline receives degenerate input.
    """
    logger.debug("process_url started", extra={"url": url.url})

    header_splits: list[LCDocument] = split_into_markdown_headers(url)

    if not header_splits:
        logger.warning(
            "No header splits produced — page may lack structured headings",
            extra={"url": url.url},
        )
        return []

    raw_chunks: list[LCDocument] = chunk_markdown_splits(header_splits)

    if not raw_chunks:
        logger.warning(
            "Chunking produced no output",
            extra={"url": url.url},
        )
        return []

    chunks_with_context: list[LCDocument] = prepend_context_to_docs(raw_chunks)

    logger.debug(
        "process_url complete",
        extra={"url": url.url, "final_chunks": len(chunks_with_context)},
    )
    return chunks_with_context
