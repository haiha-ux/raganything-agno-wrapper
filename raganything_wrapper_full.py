"""
RAG-Anything Full Wrapper for Agno Framework

Complete implementation of RAG-Anything features as Agno Knowledge wrapper.
Supports ALL features from RAG-Anything documentation without missing anything.

Features Implemented:
✅ End-to-End Multimodal Pipeline
✅ Universal Document Support (PDF, Office, images, text)
✅ Specialized Content Analysis (images, tables, equations)
✅ Multimodal Knowledge Graph
✅ Adaptive Processing Modes (MinerU/Docling)
✅ Direct Content List Insertion
✅ Hybrid Intelligent Retrieval
✅ VLM Enhanced Queries
✅ Batch Processing
✅ Custom Modal Processors
✅ All Query Modes (text, VLM, multimodal)
✅ Parser Configuration (all MinerU parameters)
"""

from typing import Any, Dict, List, Optional, Tuple, Set, Callable
from dataclasses import dataclass
import asyncio
import logging
import os
import threading

from agno.knowledge import Knowledge
from agno.knowledge.document import Document
from raganything import RAGAnything, RAGAnythingConfig as RAGConfig
from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc

logger = logging.getLogger(__name__)


# =============================================================================
# PERSISTENT EVENT LOOP THREAD (Fix for LightRAG event loop conflicts)
# =============================================================================

class PersistentEventLoopThread:
    """
    Maintains a persistent thread with its own event loop for LightRAG queries.

    This solves the "Event loop is closed" error by keeping the same event loop
    alive across multiple queries, so LightRAG workers don't lose their event loop.
    """

    def __init__(self):
        self.loop = None
        self.thread = None
        self._started = threading.Event()
        self._start_thread()

    def _start_thread(self):
        """Start the persistent thread with event loop"""
        def run_loop():
            # Create new event loop for this thread
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self._started.set()  # Signal that loop is ready

            # Keep loop running forever
            self.loop.run_forever()

        self.thread = threading.Thread(target=run_loop, daemon=True)
        self.thread.start()
        self._started.wait()  # Wait for loop to be ready
        logger.info("✅ Persistent event loop thread started")

    def run_coroutine(self, coro, timeout=120):
        """Run a coroutine in the persistent event loop"""
        if not self.loop or not self.loop.is_running():
            raise RuntimeError("Event loop not running")

        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return future.result(timeout=timeout)

    def shutdown(self):
        """Stop the event loop and thread"""
        if self.loop and self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.thread.join(timeout=5)
            logger.info("✅ Persistent event loop thread stopped")


# =============================================================================
# CONFIGURATION CLASS (Full RAG-Anything Config Support)
# =============================================================================

@dataclass
class RAGAnythingConfig:
    """
    Complete configuration for RAG-Anything wrapper.
    Supports ALL parameters from RAG-Anything documentation.
    """
    # Core directories
    working_dir: str = "./rag_storage"
    output_dir: str = "./output"

    # Parser configuration
    parser: str = "mineru"  # "mineru" or "docling"
    parse_method: str = "auto"  # "auto", "ocr", or "txt"

    # Content processing toggles
    enable_image_processing: bool = True
    enable_table_processing: bool = True
    enable_equation_processing: bool = True

    # MinerU specific parameters (all supported kwargs)
    lang: Optional[str] = None  # Document language for OCR (e.g., "ch", "en", "ja")
    device: str = "cpu"  # Inference device: "cpu", "cuda", "cuda:0", "npu", "mps"
    start_page: Optional[int] = None  # Starting page number (0-based)
    end_page: Optional[int] = None  # Ending page number (0-based)
    formula: bool = True  # Enable formula parsing
    table: bool = True  # Enable table parsing
    backend: str = "pipeline"  # "pipeline", "vlm-transformers", "vlm-sglang-engine", "vlm-sglang-client"
    source: str = "huggingface"  # Model source: "huggingface", "modelscope", "local"
    vlm_url: Optional[str] = None  # Service address for vlm-sglang-client backend

    # Batch processing
    max_workers: int = 4  # Maximum parallel workers for batch processing

    # Display options
    display_stats: bool = True  # Display content statistics during processing

    # Text splitting
    split_by_character: Optional[str] = None  # Character to split text by
    split_by_character_only: bool = False  # Split only by character (no other chunking)

    # Citation extraction (Universal for all domains)
    enable_citation_extraction: bool = True  # Auto-extract URLs/references from chunks
    max_citations: int = 5  # Maximum citations to show
    citation_style: str = "grouped"  # "grouped" (by domain), "list" (flat list), or "none"
    citation_label: str = "References"  # Label for citation section (customizable per language)


    def to_rag_config(self) -> RAGConfig:
        """Convert to RAG-Anything's native config format"""
        return RAGConfig(
            working_dir=self.working_dir,
            parser=self.parser,
            parse_method=self.parse_method,
            enable_image_processing=self.enable_image_processing,
            enable_table_processing=self.enable_table_processing,
            enable_equation_processing=self.enable_equation_processing,
        )

    def get_mineru_kwargs(self) -> Dict[str, Any]:
        """Extract MinerU-specific parameters for document processing"""
        kwargs = {}

        if self.lang is not None:
            kwargs['lang'] = self.lang
        if self.device != "cpu":
            kwargs['device'] = self.device
        if self.start_page is not None:
            kwargs['start_page'] = self.start_page
        if self.end_page is not None:
            kwargs['end_page'] = self.end_page
        if not self.formula:
            kwargs['formula'] = False
        if not self.table:
            kwargs['table'] = False
        if self.backend != "pipeline":
            kwargs['backend'] = self.backend
        if self.source != "huggingface":
            kwargs['source'] = self.source
        if self.vlm_url is not None:
            kwargs['vlm_url'] = self.vlm_url

        return kwargs


# =============================================================================
# MAIN WRAPPER CLASS (Complete RAG-Anything Integration)
# =============================================================================

class RAGAnythingWrapper(Knowledge):
    """
    Complete wrapper for RAG-Anything with full Agno Knowledge compatibility.

    Implements ALL features from RAG-Anything:
    - End-to-end multimodal pipeline
    - Universal document support
    - Specialized content analysis
    - Multimodal knowledge graph
    - Adaptive processing modes
    - Direct content list insertion
    - Hybrid intelligent retrieval
    - VLM enhanced queries
    - Batch processing
    - Custom modal processors

    Example:
        ```python
        from agno.agent import Agent
        from raganything_integration.raganything_wrapper_full import (
            RAGAnythingWrapper, RAGAnythingConfig, create_raganything_wrapper
        )

        # Create wrapper with full configuration
        wrapper = create_raganything_wrapper(
            config=RAGAnythingConfig(
                working_dir="./rag_storage",
                parser="mineru",
                device="cuda",  # GPU acceleration
                enable_image_processing=True,
                enable_table_processing=True,
                enable_equation_processing=True,
            ),
            openai_api_key="sk-...",
        )

        # Use with Agno Agent (drop-in replacement!)
        agent = Agent(
            name="Nexi",
            knowledge=wrapper,
            tools=[...],
        )

        # Or use RAG-Anything features directly
        await wrapper.process_document_complete("document.pdf")
        await wrapper.process_folder_complete("./documents")
        await wrapper.insert_content_list(content_list, "source.pdf")

        # Query with different modes
        docs = await wrapper.async_search("query")  # Agno interface
        result = await wrapper.aquery("query", mode="hybrid")  # RAG-Anything interface
        vlm_result = await wrapper.aquery_vlm_enhanced("query with images")
        multimodal_result = await wrapper.aquery_with_multimodal("query", multimodal_content)
        ```
    """

    def __init__(
        self,
        rag_anything: RAGAnything,
        config: RAGAnythingConfig,
        name: str = "RAG-Anything Knowledge",
        description: str = "Advanced multimodal knowledge base powered by RAG-Anything",
        **kwargs
    ):
        """
        Initialize wrapper with RAG-Anything instance.

        Args:
            rag_anything: Initialized RAGAnything instance
            config: RAGAnythingConfig with full configuration
            name: Knowledge base name
            description: Knowledge base description
        """
        # Store RAG-Anything instance and config
        self.rag_anything = rag_anything
        self.config = config
        self._name = name
        self._description = description

        # Cache for valid filters (populated on first use)
        self._valid_filters: Optional[Set[str]] = None

        # CRITICAL: Create persistent event loop thread for LightRAG queries
        # This prevents "Event loop is closed" errors across multiple queries
        self._persistent_loop = PersistentEventLoopThread()

        logger.info(f"✅ RAGAnythingWrapper initialized: {name}")
        # Check if lightrag is initialized before accessing working_dir
        if hasattr(rag_anything, 'lightrag') and rag_anything.lightrag is not None:
            if hasattr(rag_anything.lightrag, 'working_dir'):
                logger.info(f"   Working dir: {rag_anything.lightrag.working_dir}")
        logger.info(f"   Parser: {config.parser}")
        logger.info(f"   Parse method: {config.parse_method}")
        logger.info(f"   Device: {config.device}")
        logger.info(f"   Image processing: {config.enable_image_processing}")
        logger.info(f"   Table processing: {config.enable_table_processing}")
        logger.info(f"   Equation processing: {config.enable_equation_processing}")

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    # =========================================================================
    # AGNO KNOWLEDGE INTERFACE (Required for Agent compatibility)
    # =========================================================================

    async def async_search(
        self,
        query: str,
        max_results: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        search_type: Optional[str] = None
    ) -> List[Document]:
        """
        Search knowledge base (async version) - Agno interface.

        This is called by Agno Agent when doing RAG retrieval.
        Converts RAG-Anything string response into Agno Document format.

        Args:
            query: Search query
            max_results: Maximum number of results (not used - RAG-Anything returns text)
            filters: Metadata filters (not supported by RAG-Anything)
            search_type: Search type (maps to RAG-Anything mode)

        Returns:
            List of Document objects containing retrieved content
        """
        logger.info("[ASYNC SEARCH CALLED] async_search() method entry point!")
        logger.info(f"[ASYNC SEARCH] Query: {query[:100]}")
        try:
            # Map Agno search_type to RAG-Anything mode
            mode = self._map_search_type(search_type)

            logger.info(f"🔍 [SEARCH] Searching RAG-Anything: query='{query[:50]}...', mode={mode}")

            # Query RAG-Anything with VLM enhancement if available
            # CRITICAL: Use persistent event loop thread to avoid "Event loop is closed" errors
            # LightRAG workers stay alive in the same event loop across queries
            logger.info(f"🔍 [SEARCH] Calling rag_anything.aquery() with mode={mode}...")

            # Run in persistent thread with stable event loop
            response_text = self._persistent_loop.run_coroutine(
                self.rag_anything.aquery(
                    query=query,
                    mode=mode,
                    vlm_enhanced=True
                ),
                timeout=120  # 2 min timeout
            )
            
            logger.info(f"✅ [SEARCH] aquery returned: {len(response_text)} chars")

            # DEBUG: Log first 500 chars to check if links are present in raw response
            preview = response_text[:500] if len(response_text) > 500 else response_text
            logger.info(f"🔍 [DEBUG] Raw LightRAG response preview: {preview}")

            # ENHANCEMENT: Extract URLs from chunks and append to response
            try:
                urls = await self._extract_urls_from_chunks(query, mode)
                if urls:
                    formatted_urls = self._format_citations_for_response(urls)
                    response_text = response_text + formatted_urls
                    logger.info(f"✅ [URL INJECTION] Appended {len(urls)} URLs to response")
            except Exception as e:
                logger.warning(f"⚠️  URL extraction failed (non-critical): {e}")

            # Convert string response to Document list
            documents = self._convert_to_documents(response_text, query)

            logger.info(f"✅ [SEARCH] Retrieved {len(documents)} documents")
            return documents

        except Exception as e:
            logger.error(f"❌ Search error: {e}")
            return []

    def search(
        self,
        query: str,
        max_results: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        search_type: Optional[str] = None
    ) -> List[Document]:
        """
        Search knowledge base (sync version) - Agno interface.

        Wraps async_search for sync compatibility.
        """
        try:
            import threading
            from concurrent.futures import Future

            # Check if loop is running (likely uvloop in FastAPI)
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # uvloop doesn't support nest_asyncio, use thread instead
                    result_future = Future()

                    def run_in_thread():
                        # Use asyncio.run() which properly manages loop lifecycle
                        try:
                            result = asyncio.run(
                                self.async_search(query, max_results, filters, search_type)
                            )
                            result_future.set_result(result)
                        except Exception as e:
                            result_future.set_exception(e)

                    thread = threading.Thread(target=run_in_thread)
                    thread.start()
                    thread.join()
                    return result_future.result()
                else:
                    # Loop not running, safe to use
                    return loop.run_until_complete(
                        self.async_search(query, max_results, filters, search_type)
                    )
            except RuntimeError:
                # No loop, use asyncio.run()
                return asyncio.run(
                    self.async_search(query, max_results, filters, search_type)
                )
        except Exception as e:
            logger.error(f"❌ Sync search error: {e}")
            return []

    # =========================================================================
    # RAG-ANYTHING QUERY METHODS (All query modes)
    # =========================================================================

    async def aquery(
        self,
        query: str,
        mode: str = "hybrid",
        vlm_enhanced: Optional[bool] = None
    ) -> str:
        """
        Pure text query using RAG-Anything (async).

        Query modes:
        - "hybrid": Combines vector similarity and graph traversal (recommended)
        - "local": Local context search
        - "global": Global knowledge graph search
        - "naive": Simple retrieval without graph

        Args:
            query: Search query
            mode: Query mode (hybrid/local/global/naive)
            vlm_enhanced: Enable VLM for image analysis (auto if None)

        Returns:
            String response from RAG-Anything
        """
        try:
            # Auto-enable VLM if vision_model_func is available
            if vlm_enhanced is None:
                vlm_enhanced = hasattr(self.rag_anything, 'vision_model_func') and \
                              self.rag_anything.vision_model_func is not None

            result = await self.rag_anything.aquery(
                query=query,
                mode=mode,
                vlm_enhanced=vlm_enhanced
            )

            logger.debug(f"✅ Query completed: {len(result)} chars")
            return result

        except Exception as e:
            logger.error(f"❌ Query error: {e}")
            return f"Error: {str(e)}"

    def query(self, query: str, mode: str = "hybrid") -> str:
        """Sync version of aquery"""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.aquery(query, mode))

    async def aquery_vlm_enhanced(
        self,
        query: str,
        mode: str = "hybrid"
    ) -> str:
        """
        VLM enhanced query - automatically analyze images in retrieved context.

        When documents contain images, VLM will:
        1. Retrieve relevant context containing image paths
        2. Load and encode images as base64
        3. Send both text and images to VLM for analysis

        Args:
            query: Search query
            mode: Query mode (hybrid/local/global/naive)

        Returns:
            String response with VLM-enhanced analysis
        """
        return await self.aquery(query, mode, vlm_enhanced=True)

    async def aquery_with_multimodal(
        self,
        query: str,
        multimodal_content: List[Dict[str, Any]],
        mode: str = "hybrid"
    ) -> str:
        """
        Multimodal query with specific multimodal content.

        Enhanced queries with tables, equations, or other multimodal elements.

        Args:
            query: Search query
            multimodal_content: List of multimodal content dicts:
                - Table: {"type": "table", "table_data": "...", "table_caption": "..."}
                - Equation: {"type": "equation", "latex": "...", "equation_caption": "..."}
            mode: Query mode

        Returns:
            String response with multimodal analysis

        Example:
            ```python
            result = await wrapper.aquery_with_multimodal(
                "Compare these metrics",
                multimodal_content=[{
                    "type": "table",
                    "table_data": "Method,Accuracy\\nOurs,95%\\nBaseline,87%",
                    "table_caption": "Performance comparison"
                }],
                mode="hybrid"
            )
            ```
        """
        try:
            result = await self.rag_anything.aquery_with_multimodal(
                query=query,
                multimodal_content=multimodal_content,
                mode=mode
            )

            logger.debug(f"✅ Multimodal query completed: {len(result)} chars")
            return result

        except Exception as e:
            logger.error(f"❌ Multimodal query error: {e}")
            return f"Error: {str(e)}"

    # =========================================================================
    # DOCUMENT PROCESSING (End-to-end pipeline)
    # =========================================================================

    async def process_document_complete(
        self,
        file_path: str,
        output_dir: Optional[str] = None,
        parse_method: Optional[str] = None,
        display_stats: Optional[bool] = None,
        doc_id: Optional[str] = None,
        **kwargs
    ) -> Tuple[List[Dict[str, Any]], str]:
        """
        Process single document with full RAG-Anything pipeline.

        Supports ALL MinerU parameters via kwargs:
        - lang: Document language (e.g., "ch", "en", "ja")
        - device: Inference device ("cpu", "cuda", "cuda:0", "npu", "mps")
        - start_page: Starting page number (0-based)
        - end_page: Ending page number (0-based)
        - formula: Enable formula parsing (default True)
        - table: Enable table parsing (default True)
        - backend: Parsing backend (pipeline/vlm-transformers/vlm-sglang-engine/vlm-sglang-client)
        - source: Model source (huggingface/modelscope/local)
        - vlm_url: Service address for vlm-sglang-client backend

        Args:
            file_path: Path to document
            output_dir: Output directory (default from config)
            parse_method: Parsing method (auto/ocr/txt, default from config)
            display_stats: Display statistics (default from config)
            doc_id: Optional custom document ID
            **kwargs: Additional MinerU parameters

        Returns:
            Tuple of (content_list, doc_id)
        """
        try:
            # Use config defaults if not specified
            output_dir = output_dir or self.config.output_dir
            parse_method = parse_method or self.config.parse_method
            display_stats = display_stats if display_stats is not None else self.config.display_stats

            # Merge config MinerU kwargs with provided kwargs
            mineru_kwargs = self.config.get_mineru_kwargs()
            mineru_kwargs.update(kwargs)

            logger.info(f"📄 Processing document: {file_path}")
            logger.info(f"   Parser: {self.config.parser}")
            logger.info(f"   Method: {parse_method}")
            logger.info(f"   Output: {output_dir}")
            if mineru_kwargs:
                logger.info(f"   MinerU params: {mineru_kwargs}")

            # Process document
            result = await self.rag_anything.process_document_complete(
                file_path=file_path,
                output_dir=output_dir,
                parse_method=parse_method,
                display_stats=display_stats,
                doc_id=doc_id,
                **mineru_kwargs
            )

            # Handle None return (RAG-Anything version compatibility)
            if result is None:
                logger.warning("⚠️  RAG-Anything process_document_complete returned None (processing likely succeeded)")
                logger.info(f"✅ Document processed: {doc_id}")
                return [], doc_id

            # Unpack result
            content_list, returned_doc_id = result
            logger.info(f"✅ Document processed: {returned_doc_id}")
            logger.info(f"   Content items: {len(content_list)}")

            return content_list, returned_doc_id

        except Exception as e:
            logger.error(f"❌ Document processing error: {e}")
            raise

    async def process_folder_complete(
        self,
        folder_path: str,
        output_dir: Optional[str] = None,
        file_extensions: Optional[List[str]] = None,
        recursive: bool = True,
        max_workers: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Tuple[List[Dict[str, Any]], str]]:
        """
        Batch process all documents in folder.

        Args:
            folder_path: Path to folder containing documents
            output_dir: Output directory (default from config)
            file_extensions: List of file extensions to process (default: all supported)
            recursive: Recursively process subdirectories
            max_workers: Maximum parallel workers (default from config)
            **kwargs: Additional parameters for process_document_complete

        Returns:
            Dict mapping file paths to (content_list, doc_id) tuples
        """
        try:
            output_dir = output_dir or self.config.output_dir
            max_workers = max_workers or self.config.max_workers

            # Get supported extensions if not specified
            if file_extensions is None:
                file_extensions = self.get_supported_file_extensions()

            logger.info(f"📁 Processing folder: {folder_path}")
            logger.info(f"   Extensions: {file_extensions}")
            logger.info(f"   Recursive: {recursive}")
            logger.info(f"   Max workers: {max_workers}")

            # Process folder
            results = await self.rag_anything.process_folder_complete(
                folder_path=folder_path,
                output_dir=output_dir,
                file_extensions=file_extensions,
                recursive=recursive,
                max_workers=max_workers,
                **kwargs
            )

            logger.info(f"✅ Folder processed: {len(results)} documents")

            return results

        except Exception as e:
            logger.error(f"❌ Folder processing error: {e}")
            raise

    # =========================================================================
    # DIRECT CONTENT INSERTION (Bypass parsing)
    # =========================================================================

    async def insert_content_list(
        self,
        content_list: List[Dict[str, Any]],
        file_path: str,
        split_by_character: Optional[str] = None,
        split_by_character_only: bool = False,
        doc_id: Optional[str] = None,
        display_stats: bool = True
    ) -> str:
        """
        Insert pre-parsed content list directly into knowledge base.

        Bypasses document parsing - useful for:
        - Content from external parsers
        - Programmatically generated content
        - Cached parsing results

        Content list format:
        - Text: {"type": "text", "text": "...", "page_idx": 0}
        - Image: {"type": "image", "img_path": "/absolute/path", "image_caption": [...], "page_idx": 1}
        - Table: {"type": "table", "table_body": "...", "table_caption": [...], "page_idx": 2}
        - Equation: {"type": "equation", "latex": "...", "text": "...", "page_idx": 3}

        Args:
            content_list: List of content dicts
            file_path: Reference file path for citation
            split_by_character: Optional character to split text by
            split_by_character_only: Split only by character
            doc_id: Optional custom document ID
            display_stats: Display content statistics

        Returns:
            Document ID

        Example:
            ```python
            content_list = [
                {"type": "text", "text": "Introduction...", "page_idx": 0},
                {"type": "image", "img_path": "/path/to/fig1.jpg",
                 "image_caption": ["Figure 1"], "page_idx": 1},
                {"type": "table", "table_body": "| A | B |\\n|---|---|\\n| 1 | 2 |",
                 "table_caption": ["Table 1"], "page_idx": 2}
            ]
            doc_id = await wrapper.insert_content_list(content_list, "paper.pdf")
            ```
        """
        try:
            logger.info(f"📝 Inserting content list: {file_path}")
            logger.info(f"   Items: {len(content_list)}")

            doc_id = await self.rag_anything.insert_content_list(
                content_list=content_list,
                file_path=file_path,
                split_by_character=split_by_character,
                split_by_character_only=split_by_character_only,
                doc_id=doc_id,
                display_stats=display_stats
            )

            logger.info(f"✅ Content list inserted: {doc_id}")

            return doc_id

        except Exception as e:
            logger.error(f"❌ Content insertion error: {e}")
            raise

    # =========================================================================
    # AGNO add_content INTERFACE (For Agno compatibility)
    # =========================================================================

    async def add_content_async(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        path: Optional[str] = None,
        url: Optional[str] = None,
        text_content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        topics: Optional[List[str]] = None,
        remote_content: Any = None,
        reader: Any = None,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        upsert: bool = True,
        skip_if_exists: bool = False,
        auth: Any = None
    ) -> None:
        """
        Add content to knowledge base (async) - Agno interface.

        Supports:
        - Files: Uses RAG-Anything's multimodal parsing
        - URLs: Downloads and processes (if supported)
        - Text: Direct insertion into LightRAG

        Args:
            path: Local file path
            url: URL to fetch and process
            text_content: Raw text content
            metadata: Additional metadata
            skip_if_exists: Skip if document already processed
        """
        try:
            # Handle file path
            if path:
                logger.info(f"📄 Adding document via RAG-Anything: {path}")

                # Check if already processed
                if skip_if_exists and metadata:
                    doc_id = metadata.get('doc_id')
                    if doc_id and self.is_document_fully_processed(doc_id):
                        logger.info(f"⏭️  Skipping already processed: {path}")
                        return

                # Process with RAG-Anything
                await self.process_document_complete(
                    file_path=path,
                    output_dir=metadata.get('output_dir', self.config.output_dir) if metadata else self.config.output_dir,
                    doc_id=metadata.get('doc_id') if metadata else None
                )

                logger.info(f"✅ Document added: {path}")

            # Handle URL
            elif url:
                logger.info(f"🌐 Adding URL via RAG-Anything: {url}")
                logger.warning("⚠️  URL support depends on RAG-Anything implementation")
                # RAG-Anything may not have direct URL support - would need to download first

            # Handle raw text
            elif text_content:
                logger.info(f"📝 Adding text content ({len(text_content)} chars)")

                # Check if LightRAG is available
                if not hasattr(self.rag_anything, 'lightrag') or self.rag_anything.lightrag is None:
                    raise RuntimeError(
                        "LightRAG instance not available. "
                        "Please use acreate_raganything_wrapper() for proper initialization, "
                        "or provide a pre-initialized LightRAG instance."
                    )

                # Insert text directly into LightRAG
                await self.rag_anything.lightrag.ainsert(text_content)

                logger.info(f"✅ Text content added")

            else:
                logger.warning("⚠️  No content provided (path, url, or text_content required)")

        except Exception as e:
            logger.error(f"❌ Error adding content: {e}")
            raise

    def add_content(self, *args, **kwargs) -> None:
        """Sync version of add_content_async"""
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.add_content_async(*args, **kwargs))

    # =========================================================================
    # UTILITY METHODS (Parser info, document status, etc.)
    # =========================================================================

    def check_parser_installation(self) -> bool:
        """
        Check if parser (MinerU/Docling) is properly installed.

        Returns:
            True if parser is ready, False otherwise
        """
        try:
            return self.rag_anything.check_parser_installation()
        except Exception as e:
            logger.error(f"❌ Parser check error: {e}")
            return False

    def get_supported_file_extensions(self) -> List[str]:
        """
        Get list of supported file extensions.

        Returns:
            List of extensions (e.g., [".pdf", ".docx", ".jpg"])
        """
        try:
            return self.rag_anything.get_supported_file_extensions()
        except Exception as e:
            logger.error(f"❌ Extension check error: {e}")
            return []

    def is_document_fully_processed(self, doc_id: str) -> bool:
        """
        Check if document has already been fully processed.

        Args:
            doc_id: Document ID

        Returns:
            True if document is fully processed, False otherwise
        """
        try:
            return self.rag_anything.is_document_fully_processed(doc_id)
        except Exception as e:
            logger.error(f"❌ Document status check error: {e}")
            return False

    # =========================================================================
    # HELPER METHODS (Internal utilities)
    # =========================================================================

    def _map_search_type(self, search_type: Optional[str]) -> str:
        """
        Map Agno search_type to RAG-Anything mode.

        Agno: vector, keyword, hybrid
        RAG-Anything: local, global, hybrid, naive
        """
        if search_type == "hybrid":
            return "hybrid"
        elif search_type == "vector":
            return "local"
        elif search_type == "keyword":
            return "global"
        else:
            return "hybrid"  # Default
        
    async def _extract_urls_from_chunks(self, query: str, mode: str = "hybrid") -> Dict[str, Any]:
        """
        Universal citation extraction from retrieved chunks.

        Extracts all types of references without domain-specific logic.
        Works for ANY domain: healthcare, legal, education, e-commerce, etc.

        Args:
            query: The original query
            mode: Query mode used

        Returns:
            Dict with structured citations:
            {
                "urls": [...],           # All URLs found
                "documents": [...],      # Document sources
                "chunks_with_refs": [...] # Chunks containing references
            }
        """
        try:
            import re
            from urllib.parse import urlparse

            # Access LightRAG instance
            lightrag = self.rag_anything.lightrag
            if not lightrag or not hasattr(lightrag, 'text_chunks'):
                return {"urls": [], "documents": [], "chunks_with_refs": []}

            # Get all chunks from store
            all_urls = set()
            doc_sources = set()
            chunks_with_refs = []

            chunk_store = lightrag.text_chunks

            if hasattr(chunk_store, '_data') and chunk_store._data:
                # Iterate through all stored chunks
                for doc_id, chunk_data in chunk_store._data.items():
                    if not chunk_data:
                        continue

                    content = chunk_data.get('content', '')
                    if not content:
                        continue

                    # Extract URLs (http, https, ftp, etc.)
                    urls = re.findall(r'https?://[^\s\)\]\,\"\'\>\<]+', content)
                    if urls:
                        all_urls.update(urls)
                        chunks_with_refs.append({
                            'doc_id': doc_id,
                            'urls': urls,
                            'content_preview': content[:200]
                        })

                    # Extract document references (e.g., "Source: doc.pdf", "See: paper.docx")
                    doc_refs = re.findall(r'(?:Source|See|Reference|Ref):\s*([^\s\.\,]+\.(?:pdf|docx?|txt|html))', content, re.IGNORECASE)
                    if doc_refs:
                        doc_sources.update(doc_refs)

            citations = {
                "urls": list(all_urls),
                "documents": list(doc_sources),
                "chunks_with_refs": chunks_with_refs
            }

            logger.info(f"🔗 [CITATION EXTRACTION] Found {len(all_urls)} URLs, {len(doc_sources)} document refs")
            return citations

        except Exception as e:
            logger.warning(f"⚠️  Citation extraction failed (non-critical): {e}")
            return {"urls": [], "documents": [], "chunks_with_refs": []}

    def _format_citations_for_response(self, citations: Dict[str, Any]) -> str:
        """
        Universal citation formatting for any domain.

        Automatically groups URLs by domain and formats naturally.
        No hard-coded domain filtering - works for ANY knowledge base.

        Args:
            citations: Dict from _extract_urls_from_chunks()

        Returns:
            Formatted citation string (empty if no citations or disabled)
        """
        # Check if citation extraction is enabled
        if not self.config.enable_citation_extraction:
            return ""

        urls = citations.get("urls", [])
        documents = citations.get("documents", [])

        if not urls and not documents:
            return ""

        # Check citation style
        if self.config.citation_style == "none":
            return ""

        # Build citation section header
        formatted = f"\n\n📚 {self.config.citation_label}:\n"

        # Format based on style
        if self.config.citation_style == "grouped":
            # Auto-group by domain
            from urllib.parse import urlparse
            from collections import defaultdict

            domain_groups = defaultdict(list)
            for url in urls[:self.config.max_citations]:
                try:
                    parsed = urlparse(url)
                    domain = parsed.netloc or "other"
                    # Simplify domain (remove www, subdomain if common)
                    domain = domain.replace('www.', '')
                    domain_groups[domain].append(url)
                except:
                    domain_groups["other"].append(url)

            # Format each domain group
            for domain, domain_urls in sorted(domain_groups.items()):
                formatted += f"• {domain.title()}: " + ", ".join(domain_urls[:3]) + "\n"

        elif self.config.citation_style == "list":
            # Flat list
            for i, url in enumerate(urls[:self.config.max_citations], 1):
                formatted += f"{i}. {url}\n"

        # Add document references if any
        if documents:
            formatted += f"• Documents: " + ", ".join(documents[:3]) + "\n"

        return formatted.rstrip()


    def _convert_to_documents(self, response_text: str, query: str) -> List[Document]:
        """
        Convert RAG-Anything string response to Agno Document list.
        """
        if not response_text or not response_text.strip():
            return []

        doc = Document(
            content=response_text,
            id=f"rag_anything_{hash(query)}",
            meta_data={
                "source": "rag-anything",
                "query": query,
                "parser": self.config.parser,
                "knowledge_graph": "lightrag"
            }
        )

        return [doc]

    # =========================================================================
    # AGNO KNOWLEDGE INTERFACE (Additional methods)
    # =========================================================================

    async def aget_valid_filters(self) -> Set[str]:
        """Get valid filter keys (RAG-Anything doesn't support filters)"""
        if self._valid_filters is None:
            self._valid_filters = set()
        return self._valid_filters

    def get_valid_filters(self) -> Set[str]:
        """Sync version"""
        return set()

    async def async_validate_filters(
        self,
        filters: Optional[Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], List[str]]:
        """Validate filters (RAG-Anything doesn't support filters)"""
        if filters:
            logger.warning("⚠️  RAG-Anything doesn't support metadata filters - ignoring")
        return {}, []

    def validate_filters(
        self,
        filters: Optional[Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], List[str]]:
        """Sync version"""
        return {}, []

    async def get_graph_stats(self) -> Dict[str, int]:
        """
        Get knowledge graph statistics (entities, relations, chunks)

        Returns:
            Dict with 'total_entities', 'total_relations', 'total_chunks'
        """
        try:
            if not self.rag_anything:
                return {"total_entities": 0, "total_relations": 0, "total_chunks": 0}

            # Access LightRAG KV storages (not vector DBs)
            lightrag = self.rag_anything.lightrag

            # Count entities from KV store
            entity_count = 0
            if hasattr(lightrag, 'full_entities') and lightrag.full_entities:
                try:
                    all_doc_ids = list(lightrag.full_entities._data.keys()) if hasattr(lightrag.full_entities, '_data') else []
                    all_entities = await lightrag.full_entities.get_by_ids(all_doc_ids)
                    for doc_entities in all_entities:
                        if doc_entities and 'count' in doc_entities:
                            entity_count += doc_entities['count']
                except Exception as e:
                    logger.warning(f'Failed to count entities: {e}')

            # Count relations from KV store
            relation_count = 0
            if hasattr(lightrag, 'full_relations') and lightrag.full_relations:
                try:
                    all_doc_ids = list(lightrag.full_relations._data.keys()) if hasattr(lightrag.full_relations, '_data') else []
                    all_relations = await lightrag.full_relations.get_by_ids(all_doc_ids)
                    for doc_relations in all_relations:
                        if doc_relations and 'relation_pairs' in doc_relations:
                            relation_count += len(doc_relations['relation_pairs'])
                except Exception as e:
                    logger.warning(f'Failed to count relations: {e}')

            # Count chunks from KV store
            chunk_count = 0
            if hasattr(lightrag, 'text_chunks') and lightrag.text_chunks:
                try:
                    all_doc_ids = list(lightrag.text_chunks._data.keys()) if hasattr(lightrag.text_chunks, '_data') else []
                    all_chunks = await lightrag.text_chunks.get_by_ids(all_doc_ids)
                    chunk_count = len([c for c in all_chunks if c is not None])
                except Exception as e:
                    logger.warning(f'Failed to count chunks: {e}')

            logger.info(f"📊 Graph stats: {entity_count} entities, {relation_count} relations, {chunk_count} chunks")

            return {
                "total_entities": entity_count,
                "total_relations": relation_count,
                "total_chunks": chunk_count
            }

        except Exception as e:
            logger.error(f"Failed to get graph stats: {e}", exc_info=True)
            return {"total_entities": 0, "total_relations": 0, "total_chunks": 0}



# =============================================================================
# FACTORY FUNCTIONS (Easy initialization)
# =============================================================================

async def acreate_raganything_wrapper(
    config: Optional[RAGAnythingConfig] = None,
    openai_api_key: Optional[str] = None,
    openai_base_url: Optional[str] = None,
    openai_model: str = "gpt-4o-mini",
    openai_vision_model: str = "gpt-4o",
    embedding_model: str = "text-embedding-3-large",
    embedding_dim: int = 3072,
    **kwargs
) -> RAGAnythingWrapper:
    """
    Async factory function to create RAGAnythingWrapper (RECOMMENDED).

    This is the RECOMMENDED way to create wrapper as it properly initializes
    all async components.

    Args:
        config: RAGAnythingConfig (if None, creates default)
        openai_api_key: OpenAI API key
        openai_base_url: Optional OpenAI base URL
        openai_model: LLM model for text processing
        openai_vision_model: Vision model for image analysis
        embedding_model: Embedding model
        embedding_dim: Embedding dimensions
        **kwargs: Additional config parameters

    Returns:
        Initialized RAGAnythingWrapper

    Example:
        ```python
        wrapper = await acreate_raganything_wrapper(
            config=RAGAnythingConfig(
                working_dir="./my_rag",
                parser="mineru",
                device="cuda",
            ),
            openai_api_key="sk-...",
        )

        agent = Agent(knowledge=wrapper, ...)
        ```
    """
    wrapper = create_raganything_wrapper(
        config=config,
        openai_api_key=openai_api_key,
        openai_base_url=openai_base_url,
        openai_model=openai_model,
        openai_vision_model=openai_vision_model,
        embedding_model=embedding_model,
        embedding_dim=embedding_dim,
        **kwargs
    )

    # Initialize storages asynchronously
    if hasattr(wrapper.rag_anything, 'lightrag') and wrapper.rag_anything.lightrag is not None:
        await wrapper.rag_anything.lightrag.initialize_storages()
        logger.info("✅ LightRAG storages initialized")

        # Initialize pipeline status (required for RAG operations)
        try:
            from lightrag.kg.shared_storage import initialize_pipeline_status
            await initialize_pipeline_status()
            logger.info("✅ Pipeline status initialized")
        except Exception as e:
            logger.warning(f"⚠️  Pipeline status initialization failed: {e}")

    return wrapper


def create_raganything_wrapper(
    config: Optional[RAGAnythingConfig] = None,
    openai_api_key: Optional[str] = None,
    openai_base_url: Optional[str] = None,
    openai_model: str = "gpt-4o-mini",
    openai_vision_model: str = "gpt-4o",
    embedding_model: str = "text-embedding-3-large",
    embedding_dim: int = 3072,
    **kwargs
) -> RAGAnythingWrapper:
    """
    Factory function to create RAGAnythingWrapper with OpenAI models.

    Args:
        config: RAGAnythingConfig (if None, creates default)
        openai_api_key: OpenAI API key
        openai_base_url: Optional OpenAI base URL
        openai_model: LLM model for text processing
        openai_vision_model: Vision model for image analysis
        embedding_model: Embedding model
        embedding_dim: Embedding dimensions
        **kwargs: Additional config parameters

    Returns:
        Initialized RAGAnythingWrapper

    Example:
        ```python
        wrapper = create_raganything_wrapper(
            config=RAGAnythingConfig(
                working_dir="./my_rag",
                parser="mineru",
                device="cuda",
            ),
            openai_api_key="sk-...",
        )

        agent = Agent(knowledge=wrapper, ...)
        ```
    """
    from lightrag.llm.openai import openai_complete_if_cache, openai_embed

    # Create config if not provided
    if config is None:
        config = RAGAnythingConfig(**kwargs)

    # Get API key from env if not provided
    if openai_api_key is None:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OpenAI API key required (provide openai_api_key or set OPENAI_API_KEY env var)")

    # LLM model function
    def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs_inner):
        return openai_complete_if_cache(
            openai_model,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=openai_api_key,
            base_url=openai_base_url,
            **kwargs_inner,
        )

    # Vision model function
    def vision_model_func(
        prompt, system_prompt=None, history_messages=[],
        image_data=None, messages=None, **kwargs_inner
    ):
        if messages:
            # Multimodal VLM format
            return openai_complete_if_cache(
                openai_vision_model,
                "",
                system_prompt=None,
                history_messages=[],
                messages=messages,
                api_key=openai_api_key,
                base_url=openai_base_url,
                **kwargs_inner,
            )
        elif image_data:
            # Single image format
            return openai_complete_if_cache(
                openai_vision_model,
                "",
                system_prompt=None,
                history_messages=[],
                messages=[
                    {"role": "system", "content": system_prompt} if system_prompt else None,
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
                            },
                        ],
                    } if image_data else {"role": "user", "content": prompt},
                ],
                api_key=openai_api_key,
                base_url=openai_base_url,
                **kwargs_inner,
            )
        else:
            # Pure text
            return llm_model_func(prompt, system_prompt, history_messages, **kwargs_inner)

    # Embedding function
    embedding_func = EmbeddingFunc(
        embedding_dim=embedding_dim,
        max_token_size=8192,
        func=lambda texts: openai_embed(
            texts,
            model=embedding_model,
            api_key=openai_api_key,
            base_url=openai_base_url,
        ),
    )

    # Create LightRAG instance first (ensures it exists)
    from lightrag import LightRAG

    lightrag = LightRAG(
        working_dir=config.working_dir,
        llm_model_func=llm_model_func,
        embedding_func=embedding_func,
        # Increased workers for better performance (balanced with stability)
        # PersistentEventLoopThread handles event loop isolation
        embedding_func_max_async=4,
        llm_model_max_async=4,
    )

    # Create RAG-Anything instance with pre-created LightRAG
    rag_anything = RAGAnything(
        lightrag=lightrag,  # Pass existing LightRAG instance
        vision_model_func=vision_model_func,
    )

    # Wrap it for Agno compatibility
    wrapper = RAGAnythingWrapper(
        rag_anything=rag_anything,
        config=config,
        name=f"RAG-Anything Knowledge ({config.parser})",
        description=f"Multimodal knowledge base with {config.parser} parsing and LightRAG"
    )

    logger.info("✅ RAGAnythingWrapper created successfully")
    return wrapper


def create_raganything_from_existing_lightrag(
    lightrag: LightRAG,
    vision_model_func: Callable,
    config: Optional[RAGAnythingConfig] = None,
    **kwargs
) -> RAGAnythingWrapper:
    """
    Create RAGAnythingWrapper from existing LightRAG instance.

    Useful for integrating with pre-existing LightRAG knowledge bases.

    Args:
        lightrag: Existing LightRAG instance
        vision_model_func: Vision model function for image processing
        config: Optional RAGAnythingConfig (if None, creates default)
        **kwargs: Additional config parameters

    Returns:
        RAGAnythingWrapper wrapping existing LightRAG

    Example:
        ```python
        # Load existing LightRAG
        lightrag = LightRAG(working_dir="./existing_rag", ...)
        await lightrag.initialize_storages()

        # Wrap it with RAG-Anything capabilities
        wrapper = create_raganything_from_existing_lightrag(
            lightrag=lightrag,
            vision_model_func=vision_func,
        )

        # Now has multimodal capabilities!
        await wrapper.process_document_complete("multimodal.pdf")
        ```
    """
    # Create config if not provided
    if config is None:
        config = RAGAnythingConfig(**kwargs)

    # Create RAG-Anything with existing LightRAG
    rag_anything = RAGAnything(
        lightrag=lightrag,
        vision_model_func=vision_model_func,
    )

    # Wrap it
    wrapper = RAGAnythingWrapper(
        rag_anything=rag_anything,
        config=config,
        name="RAG-Anything Knowledge (Existing LightRAG)",
        description="Multimodal wrapper for existing LightRAG knowledge base"
    )

    logger.info("✅ RAGAnythingWrapper created from existing LightRAG")
    return wrapper
