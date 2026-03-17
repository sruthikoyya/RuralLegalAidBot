import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("indexer")

from modules.config import DOCS_DIR, CHROMA_DIR, CHROMA_COLLECTION
from modules.qa_system import index_documents


def main():
    logger.info("=" * 60)
    logger.info("  RuralLegalAidBot v2 — Document Indexing Pipeline")
    logger.info("=" * 60)
    logger.info(f"  Documents dir : {DOCS_DIR}")
    logger.info(f"  ChromaDB dir  : {CHROMA_DIR}")
    logger.info(f"  Collection    : {CHROMA_COLLECTION}")
    logger.info("=" * 60)

    if not DOCS_DIR.exists():
        logger.error(f"Documents directory not found: {DOCS_DIR}")
        logger.info("Create it and add your PDF/DOCX/TXT files, then re-run.")
        sys.exit(1)

    doc_files = list(DOCS_DIR.glob("**/*"))
    doc_files = [f for f in doc_files if f.is_file()]
    logger.info(f"Found {len(doc_files)} file(s) in {DOCS_DIR}")

    t_start = time.time()
    chunks  = index_documents(str(DOCS_DIR))
    elapsed = time.time() - t_start

    if chunks == 0:
        logger.error(
            "No chunks indexed. Possible causes:\n"
            "  • No supported files (.pdf, .docx, .txt) in the directory\n"
            "  • All files were empty or unreadable\n"
            "  • Tesseract not installed for scanned PDFs"
        )
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("   Indexing complete!")
    logger.info(f"     Chunks indexed : {chunks}")
    logger.info(f"     Time taken     : {elapsed:.1f}s")
    logger.info(f"     Vector DB      : {CHROMA_DIR}")
    logger.info("")
    logger.info("  Next: start the server:")
    logger.info("     uvicorn main:app --host 0.0.0.0 --port 8000")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
