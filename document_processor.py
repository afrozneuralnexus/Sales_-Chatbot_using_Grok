"""
Document Processor Module
Handles loading, parsing, and chunking of PDF and Excel/CSV documents.
"""

import os
import io
import logging
from pathlib import Path
from typing import List, Optional
import pandas as pd

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Processes PDF, Excel, and CSV files into LangChain Document chunks
    ready for embedding and vector storage.
    """

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_document(self, file_path: str) -> List[Document]:
        """Route file to the correct loader and return chunked Documents."""
        path = Path(file_path)
        ext = path.suffix.lower()

        logger.info(f"Loading: {path.name} ({ext})")

        if ext == ".pdf":
            raw_docs = self._load_pdf(file_path)
        elif ext in (".xlsx", ".xls"):
            raw_docs = self._load_excel(file_path)
        elif ext == ".csv":
            raw_docs = self._load_csv(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        chunks = self.text_splitter.split_documents(raw_docs)

        # Tag every chunk with the source filename
        for chunk in chunks:
            chunk.metadata["source_file"] = path.name

        logger.info(f"  → {len(raw_docs)} page(s) / sheet(s), {len(chunks)} chunks")
        return chunks

    def load_from_bytes(
        self, file_bytes: bytes, filename: str
    ) -> List[Document]:
        """
        Load a document from raw bytes (e.g. from Streamlit's uploader).
        Writes a temp file so loaders can use the file path.
        """
        tmp_dir = Path("/tmp/rag_uploads")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = tmp_dir / filename

        tmp_path.write_bytes(file_bytes)
        docs = self.load_document(str(tmp_path))
        return docs

    # ------------------------------------------------------------------
    # Private loaders
    # ------------------------------------------------------------------

    def _load_pdf(self, file_path: str) -> List[Document]:
        """Load PDF using pdfplumber for better text extraction."""
        try:
            import pdfplumber

            documents = []
            with pdfplumber.open(file_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text() or ""
                    # Also grab any tables on the page
                    tables = page.extract_tables()
                    if tables:
                        for table in tables:
                            table_text = self._table_to_text(table)
                            text += f"\n\nTable:\n{table_text}"

                    if text.strip():
                        documents.append(
                            Document(
                                page_content=text.strip(),
                                metadata={
                                    "source": file_path,
                                    "page": i + 1,
                                    "type": "pdf",
                                },
                            )
                        )
            return documents

        except Exception as e:
            logger.warning(f"pdfplumber failed ({e}), falling back to pypdf")
            return self._load_pdf_pypdf(file_path)

    def _load_pdf_pypdf(self, file_path: str) -> List[Document]:
        """Fallback PDF loader using pypdf."""
        from pypdf import PdfReader

        reader = PdfReader(file_path)
        documents = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                documents.append(
                    Document(
                        page_content=text.strip(),
                        metadata={"source": file_path, "page": i + 1, "type": "pdf"},
                    )
                )
        return documents

    def _load_excel(self, file_path: str) -> List[Document]:
        """Load Excel file – one Document per sheet."""
        documents = []
        xl = pd.ExcelFile(file_path)

        for sheet_name in xl.sheet_names:
            df = xl.parse(sheet_name)
            text = self._dataframe_to_text(df, sheet_name)
            if text.strip():
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": file_path,
                            "sheet": sheet_name,
                            "type": "excel",
                            "rows": len(df),
                            "columns": list(df.columns),
                        },
                    )
                )
        return documents

    def _load_csv(self, file_path: str) -> List[Document]:
        """Load CSV file."""
        try:
            df = pd.read_csv(file_path)
        except Exception:
            df = pd.read_csv(file_path, encoding="latin-1")

        text = self._dataframe_to_text(df, Path(file_path).stem)
        return [
            Document(
                page_content=text,
                metadata={
                    "source": file_path,
                    "type": "csv",
                    "rows": len(df),
                    "columns": list(df.columns),
                },
            )
        ]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _dataframe_to_text(df: pd.DataFrame, label: str) -> str:
        """Convert a DataFrame to a readable text block."""
        df = df.fillna("").astype(str)
        lines = [f"Data from: {label}", f"Columns: {', '.join(df.columns)}", ""]

        for _, row in df.iterrows():
            row_parts = [f"{col}: {val}" for col, val in row.items() if val.strip()]
            lines.append(" | ".join(row_parts))

        return "\n".join(lines)

    @staticmethod
    def _table_to_text(table: list) -> str:
        """Convert a pdfplumber table (list of lists) to text."""
        rows = []
        for row in table:
            cells = [str(c).strip() if c else "" for c in row]
            rows.append(" | ".join(cells))
        return "\n".join(rows)
