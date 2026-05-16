#!/usr/bin/env python3
"""
CLI ingestion script — ingest documents into the FAISS vector store
without launching the Streamlit UI.

Usage:
    python ingest.py path/to/file.pdf
    python ingest.py path/to/data.xlsx
    python ingest.py data/                # ingest entire directory
    python ingest.py --reset              # wipe the vector store
"""

import sys
import argparse
import logging
from pathlib import Path

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from document_processor import DocumentProcessor
from vector_store import VectorStoreManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

SUPPORTED_EXTS = {".pdf", ".xlsx", ".xls", ".csv"}


def collect_files(paths: list[str]) -> list[Path]:
    files = []
    for p in paths:
        path = Path(p)
        if path.is_file():
            if path.suffix.lower() in SUPPORTED_EXTS:
                files.append(path)
            else:
                logger.warning(f"Skipping unsupported file: {path}")
        elif path.is_dir():
            for ext in SUPPORTED_EXTS:
                files.extend(path.rglob(f"*{ext}"))
        else:
            logger.warning(f"Path not found: {path}")
    return sorted(set(files))


def main():
    parser = argparse.ArgumentParser(description="RAG Sales Assistant – Document Ingestion")
    parser.add_argument("paths", nargs="*", help="Files or directories to ingest")
    parser.add_argument("--reset", action="store_true", help="Reset the vector store before ingesting")
    parser.add_argument("--vectorstore-dir", default="vectorstore", help="Vector store directory")
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=150)
    args = parser.parse_args()

    vs = VectorStoreManager(vectorstore_dir=args.vectorstore_dir)
    processor = DocumentProcessor(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    if args.reset:
        logger.info("Resetting vector store…")
        vs.reset()

    if not args.paths:
        logger.info("No paths provided. Use --reset to clear or pass file/dir paths.")
        parser.print_help()
        return

    # Load existing index
    vs.load_existing()

    # Collect files
    files = collect_files(args.paths)
    if not files:
        logger.warning("No supported files found.")
        return

    logger.info(f"Found {len(files)} file(s) to process.")
    total_chunks = 0

    for file_path in files:
        logger.info(f"\n── Processing: {file_path.name}")
        if file_path.name in vs.ingested_files:
            logger.info(f"   Already indexed – skipping")
            continue

        try:
            docs = processor.load_document(str(file_path))
            n = vs.add_documents(docs, file_path.name)
            total_chunks += n
            logger.info(f"   ✓ Added {n} chunks")
        except Exception as e:
            logger.error(f"   ✗ Failed: {e}")

    stats = vs.get_stats()
    print(f"\n{'='*50}")
    print(f"Ingestion complete!")
    print(f"  Files indexed : {stats['num_files']}")
    print(f"  Total vectors : {stats.get('num_vectors', 0)}")
    print(f"  New chunks    : {total_chunks}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
