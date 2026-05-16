"""
Vector Store Manager
Handles embedding creation and FAISS vector store operations.
Uses sentence-transformers/all-MiniLM-L6-v2 (fully local, no API key).
"""

import os
import logging
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

# Default paths
DEFAULT_VECTORSTORE_DIR = Path("vectorstore")
DEFAULT_INDEX_NAME = "sales_assistant"


class VectorStoreManager:
    """
    Manages a FAISS vector store backed by local sentence-transformer embeddings.
    """

    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(
        self,
        vectorstore_dir: str = str(DEFAULT_VECTORSTORE_DIR),
        index_name: str = DEFAULT_INDEX_NAME,
    ):
        self.vectorstore_dir = Path(vectorstore_dir)
        self.vectorstore_dir.mkdir(parents=True, exist_ok=True)
        self.index_name = index_name
        self.index_path = self.vectorstore_dir / self.index_name

        self._embeddings: Optional[HuggingFaceEmbeddings] = None
        self._vectorstore: Optional[FAISS] = None

        # Track ingested files to avoid duplicates
        self._metadata_path = self.vectorstore_dir / f"{self.index_name}_meta.pkl"
        self._ingested_files: set = self._load_metadata()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def embeddings(self) -> HuggingFaceEmbeddings:
        if self._embeddings is None:
            logger.info(f"Loading embedding model: {self.EMBEDDING_MODEL}")
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.EMBEDDING_MODEL,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        return self._embeddings

    @property
    def vectorstore(self) -> Optional[FAISS]:
        return self._vectorstore

    @property
    def is_ready(self) -> bool:
        return self._vectorstore is not None

    @property
    def ingested_files(self) -> set:
        return self._ingested_files.copy()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_existing(self) -> bool:
        """
        Load an existing FAISS index from disk.
        Returns True if successfully loaded.
        """
        if not (self.index_path / "index.faiss").exists():
            return False

        try:
            logger.info(f"Loading existing vector store from {self.index_path}")
            self._vectorstore = FAISS.load_local(
                str(self.index_path),
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
            logger.info("Vector store loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")
            return False

    def add_documents(self, documents: List[Document], source_filename: str) -> int:
        """
        Add documents to the vector store.
        Returns the number of chunks added.
        """
        if not documents:
            logger.warning("No documents to add")
            return 0

        if source_filename in self._ingested_files:
            logger.info(f"'{source_filename}' already ingested – skipping")
            return 0

        logger.info(f"Embedding {len(documents)} chunks from '{source_filename}'…")

        if self._vectorstore is None:
            self._vectorstore = FAISS.from_documents(documents, self.embeddings)
        else:
            self._vectorstore.add_documents(documents)

        self._ingested_files.add(source_filename)
        self._save()
        self._save_metadata()

        logger.info(f"Added {len(documents)} chunks. Total files: {len(self._ingested_files)}")
        return len(documents)

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        score_threshold: float = 0.0,
    ) -> List[Tuple[Document, float]]:
        """
        Return top-k most relevant chunks with similarity scores.
        """
        if not self.is_ready:
            raise RuntimeError("Vector store is not initialised. Please upload documents first.")

        results = self._vectorstore.similarity_search_with_relevance_scores(query, k=k)

        if score_threshold > 0:
            results = [(doc, score) for doc, score in results if score >= score_threshold]

        return results

    def get_retriever(self, k: int = 5):
        """Return a LangChain retriever interface."""
        if not self.is_ready:
            raise RuntimeError("Vector store is not initialised.")
        return self._vectorstore.as_retriever(search_kwargs={"k": k})

    def reset(self):
        """Delete the vector store and reset state."""
        import shutil
        if self.index_path.exists():
            shutil.rmtree(self.index_path)
        if self._metadata_path.exists():
            self._metadata_path.unlink()
        self._vectorstore = None
        self._ingested_files = set()
        logger.info("Vector store reset")

    def get_stats(self) -> dict:
        """Return basic statistics about the vector store."""
        stats = {
            "is_ready": self.is_ready,
            "ingested_files": list(self._ingested_files),
            "num_files": len(self._ingested_files),
            "index_path": str(self.index_path),
        }
        if self._vectorstore is not None:
            try:
                stats["num_vectors"] = self._vectorstore.index.ntotal
            except Exception:
                pass
        return stats

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _save(self):
        """Persist the FAISS index to disk."""
        if self._vectorstore:
            self.index_path.mkdir(parents=True, exist_ok=True)
            self._vectorstore.save_local(str(self.index_path))
            logger.info(f"Vector store saved to {self.index_path}")

    def _save_metadata(self):
        with open(self._metadata_path, "wb") as f:
            pickle.dump(self._ingested_files, f)

    def _load_metadata(self) -> set:
        if self._metadata_path.exists():
            try:
                with open(self._metadata_path, "rb") as f:
                    return pickle.load(f)
            except Exception:
                pass
        return set()
