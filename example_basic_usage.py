"""
Basic Usage Example for RAG-Anything Wrapper

Demonstrates:
1. Creating wrapper with configuration
2. Processing a document (if available)
3. Querying with different modes
4. Using with Agno Agent
"""

import asyncio
import sys
import io
import os

# UTF-8 encoding for Vietnamese
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from raganything_wrapper_full import (
    acreate_raganything_wrapper,
    RAGAnythingConfig
)


async def main():
    """Basic usage example"""

    print("\n" + "=" * 80)
    print("RAG-ANYTHING WRAPPER - BASIC USAGE EXAMPLE")
    print("=" * 80 + "\n")

    # Step 1: Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not set")
        print("   Please set your OpenAI API key:")
        print("   export OPENAI_API_KEY=sk-...")
        return

    print("✅ OpenAI API key found\n")

    # Step 2: Create wrapper with configuration
    print("[STEP 1] Creating RAG-Anything wrapper...")
    print("   Configuration:")
    print("   - Parser: MinerU (multimodal)")
    print("   - Device: CPU")
    print("   - Image processing: Enabled")
    print("   - Table processing: Enabled")
    print("   - Equation processing: Enabled")
    print()

    config = RAGAnythingConfig(
        working_dir="./example_rag_storage",
        parser="mineru",
        parse_method="auto",
        device="cpu",
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True,
    )

    wrapper = await acreate_raganything_wrapper(
        config=config,
        openai_api_key=api_key,
        openai_model="gpt-4o-mini",
        openai_vision_model="gpt-4o",
    )

    print(f"✅ Wrapper created: {wrapper.name}")
    print(f"   Working dir: {config.working_dir}")
    print()

    # Step 3: Check parser installation
    print("[STEP 2] Checking parser installation...")
    parser_ok = wrapper.check_parser_installation()
    print(f"   Parser status: {'✅ Ready' if parser_ok else '⚠️  Not installed'}")

    if not parser_ok:
        print("   To install MinerU:")
        print("   pip install magic-pdf[full]")
    print()

    # Step 4: Check supported file extensions
    print("[STEP 3] Supported file extensions:")
    extensions = wrapper.get_supported_file_extensions()
    print(f"   Found {len(extensions)} extensions:")
    print(f"   {', '.join(extensions[:15])}...")
    print()

    # Step 5: Demonstrate query methods
    print("[STEP 4] Query method demonstrations:")
    print()

    # NOTE: Querying without documents will return error or empty result
    # This is expected - just demonstrating the API

    print("   a) Pure text query (aquery):")
    print("      - Mode: hybrid (vector + graph)")
    print("      - Used for: Standard RAG retrieval")
    try:
        # This will likely fail without documents, but shows the API
        result = await wrapper.aquery(
            "What information is available?",
            mode="hybrid"
        )
        print(f"      Result: {result[:100]}...")
    except Exception as e:
        print(f"      ⚠️  Expected error (no documents): {str(e)[:80]}...")
    print()

    print("   b) VLM enhanced query (aquery_vlm_enhanced):")
    print("      - Automatically analyzes images in context")
    print("      - Used for: Visual content analysis")
    print("      (Requires documents with images)")
    print()

    print("   c) Multimodal query (aquery_with_multimodal):")
    print("      - Query with specific table/equation data")
    print("      - Example:")
    print("        multimodal_content=[{")
    print("          'type': 'table',")
    print("          'table_data': '...',")
    print("          'table_caption': 'Performance metrics'")
    print("        }]")
    print()

    print("   d) Agno interface (async_search):")
    print("      - Compatible with Agno Agent")
    print("      - Returns List[Document]")
    try:
        documents = await wrapper.async_search(
            "test query",
            search_type="hybrid"
        )
        print(f"      Result: {len(documents)} documents")
    except Exception as e:
        print(f"      ⚠️  Expected error (no documents): {str(e)[:80]}...")
    print()

    # Step 6: Show how to use with Agno Agent
    print("[STEP 5] Integration with Agno Agent:")
    print()
    print("   Example code:")
    print("   ```python")
    print("   from agno.agent import Agent")
    print()
    print("   agent = Agent(")
    print("       name='Nexi',")
    print("       knowledge=wrapper,  # RAG-Anything wrapper!")
    print("       model='gpt-4o',")
    print("       instructions='You are Nexi...',")
    print("   )")
    print()
    print("   response = await agent.arun('Your question')")
    print("   ```")
    print()

    # Step 7: Show document processing API
    print("[STEP 6] Document Processing API:")
    print()
    print("   a) Process single document:")
    print("      content_list, doc_id = await wrapper.process_document_complete(")
    print("          file_path='document.pdf',")
    print("          output_dir='./output',")
    print("          parse_method='auto',")
    print("          lang='en',  # For OCR")
    print("          device='cpu',  # or 'cuda' for GPU")
    print("      )")
    print()

    print("   b) Process folder:")
    print("      results = await wrapper.process_folder_complete(")
    print("          folder_path='./documents',")
    print("          file_extensions=['.pdf', '.docx'],")
    print("          recursive=True,")
    print("          max_workers=4,")
    print("      )")
    print()

    print("   c) Insert pre-parsed content:")
    print("      doc_id = await wrapper.insert_content_list(")
    print("          content_list=[{")
    print("              'type': 'text',")
    print("              'text': 'Content...',")
    print("              'page_idx': 0")
    print("          }],")
    print("          file_path='source.pdf',")
    print("      )")
    print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print("✅ Wrapper initialized successfully")
    print(f"✅ Parser status: {'Ready' if parser_ok else 'Not installed (optional)'}")
    print(f"✅ Supported formats: {len(extensions)} file types")
    print("✅ All query methods available:")
    print("   - aquery() - Pure text")
    print("   - aquery_vlm_enhanced() - With image analysis")
    print("   - aquery_with_multimodal() - With specific content")
    print("   - async_search() - Agno interface")
    print("✅ Document processing methods available:")
    print("   - process_document_complete()")
    print("   - process_folder_complete()")
    print("   - insert_content_list()")
    print()
    print("🎯 Next steps:")
    print("   1. Process some documents: wrapper.process_document_complete('doc.pdf')")
    print("   2. Query your knowledge base: wrapper.aquery('question')")
    print("   3. Use with Agno Agent: Agent(knowledge=wrapper, ...)")
    print()
    print("📖 See README.md for complete documentation and examples")
    print()


if __name__ == "__main__":
    asyncio.run(main())
