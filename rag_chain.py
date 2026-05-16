"""
RAG Chain Module
Supports two LLM backends:
  • Groq  — free cloud API (llama3, mixtral, gemma). No credit card needed.
            Sign up free at https://console.groq.com  → get API key → add to
            Streamlit secrets as GROQ_API_KEY.
  • Ollama — local inference, runs on your own machine. No key needed.
             Cannot be used on Streamlit Cloud (no localhost process allowed).
"""

import logging
import os
from typing import Any, Dict, Iterator, List, Tuple

from langchain.schema import Document
from langchain_core.prompts import PromptTemplate

logger = logging.getLogger(__name__)

BACKEND_GROQ = "groq"
BACKEND_OLLAMA = "ollama"

GROQ_MODELS = {
    "llama3-8b": {
        "groq_id": "llama3-8b-8192",
        "display": "Llama 3 8B (Groq ☁️)",
        "description": "Fast & capable, 8K context — free on Groq",
        "backend": BACKEND_GROQ,
    },
    "llama3-70b": {
        "groq_id": "llama3-70b-8192",
        "display": "Llama 3 70B (Groq ☁️)",
        "description": "Most capable open model — free on Groq",
        "backend": BACKEND_GROQ,
    },
    "mixtral-8x7b": {
        "groq_id": "mixtral-8x7b-32768",
        "display": "Mixtral 8×7B (Groq ☁️)",
        "description": "32K context, excellent reasoning — free on Groq",
        "backend": BACKEND_GROQ,
    },
    "gemma2-9b": {
        "groq_id": "gemma2-9b-it",
        "display": "Gemma 2 9B (Groq ☁️)",
        "description": "Google's efficient open model — free on Groq",
        "backend": BACKEND_GROQ,
    },
}

OLLAMA_MODELS = {
    "mistral": {
        "ollama_name": "mistral",
        "display": "Mistral 7B (Ollama 🖥)",
        "description": "Local only — requires Ollama running on this machine",
        "backend": BACKEND_OLLAMA,
    },
    "llama3": {
        "ollama_name": "llama3",
        "display": "Llama 3 8B (Ollama 🖥)",
        "description": "Local only — requires Ollama running on this machine",
        "backend": BACKEND_OLLAMA,
    },
    "phi3": {
        "ollama_name": "phi3",
        "display": "Phi-3 Mini (Ollama 🖥)",
        "description": "Local only — requires Ollama running on this machine",
        "backend": BACKEND_OLLAMA,
    },
}

SUPPORTED_MODELS = {**GROQ_MODELS, **OLLAMA_MODELS}

RAG_PROMPT_TEMPLATE = """You are an expert Sales Knowledge Assistant for a business.
Answer questions accurately using ONLY the provided context documents.

Rules:
- Answer based strictly on the provided context.
- If the context does not contain enough information, say so clearly.
- Be concise, professional, and sales-oriented.
- When referencing data (prices, quantities, dates) be precise.
- Format lists and tables clearly when the data calls for it.

Context:
{context}

Question: {question}

Answer:"""


class RAGChain:
    def __init__(
        self,
        model_key: str = "llama3-8b",
        temperature: float = 0.1,
        max_tokens: int = 1024,
        groq_api_key: str = None,
    ):
        if model_key not in SUPPORTED_MODELS:
            raise ValueError(f"Unknown model '{model_key}'. Choose from: {list(SUPPORTED_MODELS)}")

        self.model_key = model_key
        self.model_info = SUPPORTED_MODELS[model_key]
        self.backend = self.model_info["backend"]
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._groq_api_key = groq_api_key or self._resolve_groq_key()
        self._llm = None
        self._prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=RAG_PROMPT_TEMPLATE,
        )

    @staticmethod
    def _resolve_groq_key():
        key = os.environ.get("GROQ_API_KEY", "")
        if key:
            return key
        try:
            import streamlit as st
            return st.secrets.get("GROQ_API_KEY", "")
        except Exception:
            return ""

    @property
    def llm(self):
        if self._llm is None:
            self._llm = self._build_llm()
        return self._llm

    def _build_llm(self):
        if self.backend == BACKEND_GROQ:
            return self._build_groq_llm()
        return self._build_ollama_llm()

    def _build_groq_llm(self):
        try:
            from langchain_groq import ChatGroq
        except ImportError as e:
            raise ImportError("langchain-groq not installed. Run: pip install langchain-groq") from e

        key = self._groq_api_key or self._resolve_groq_key()
        if not key:
            raise ValueError(
                "GROQ_API_KEY is not set.\n"
                "Get a free key at https://console.groq.com then add it to:\n"
                "  Streamlit Cloud → App Settings → Secrets:\n"
                "    GROQ_API_KEY = \"gsk_...\"\n"
                "  Local: set env var or add to .env"
            )

        logger.info(f"Building Groq LLM: {self.model_info['groq_id']}")
        return ChatGroq(
            model=self.model_info["groq_id"],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            groq_api_key=key,
        )

    def _build_ollama_llm(self):
        try:
            from langchain_ollama import OllamaLLM
            logger.info(f"Building Ollama LLM: {self.model_info['ollama_name']}")
            return OllamaLLM(model=self.model_info["ollama_name"], temperature=self.temperature)
        except ImportError:
            from langchain_community.llms import Ollama
            return Ollama(model=self.model_info["ollama_name"], temperature=self.temperature)

    def query(self, question: str, retrieved_docs: List[Tuple[Document, float]]) -> Dict[str, Any]:
        if not retrieved_docs:
            return {
                "answer": "I couldn't find relevant information in the uploaded documents.",
                "sources": [],
                "context_used": "",
            }

        context_parts, sources = [], []
        for doc, score in retrieved_docs:
            context_parts.append(doc.page_content)
            meta = doc.metadata.copy()
            meta["relevance_score"] = round(score, 4)
            sources.append(meta)

        context = "\n\n---\n\n".join(context_parts)
        prompt_text = self._prompt.format(context=context, question=question)

        logger.info(f"Querying {self.model_key} via {self.backend}")
        raw = self.llm.invoke(prompt_text)
        answer = raw.content if hasattr(raw, "content") else str(raw)

        return {"answer": answer.strip(), "sources": sources, "context_used": context}

    def stream_query(self, question: str, retrieved_docs: List[Tuple[Document, float]]) -> Iterator[str]:
        if not retrieved_docs:
            yield "I couldn't find relevant information in the uploaded documents."
            return

        context = "\n\n---\n\n".join(doc.page_content for doc, _ in retrieved_docs)
        prompt_text = self._prompt.format(context=context, question=question)

        for chunk in self.llm.stream(prompt_text):
            yield chunk.content if hasattr(chunk, "content") else str(chunk)

    def check_backend_available(self) -> Tuple[bool, str]:
        if self.backend == BACKEND_GROQ:
            return self._check_groq()
        return self._check_ollama()

    # backward compat
    def check_ollama_available(self) -> Tuple[bool, str]:
        return self.check_backend_available()

    def _check_groq(self) -> Tuple[bool, str]:
        key = self._groq_api_key or self._resolve_groq_key()
        if not key:
            return False, (
                "GROQ_API_KEY not set.\n"
                "1. Get a free key → https://console.groq.com\n"
                "2. Streamlit Cloud: App Settings → Secrets → add:\n"
                "   GROQ_API_KEY = \"gsk_...\"\n"
                "3. Local: export GROQ_API_KEY=gsk_..."
            )
        return True, f"✅ Groq API key found — {self.model_info['display']} ready"

    def _check_ollama(self) -> Tuple[bool, str]:
        try:
            import ollama
            models = ollama.list()
            available = [m.model for m in models.models]
            name = self.model_info["ollama_name"]
            found = any(m == name or m.startswith(name + ":") for m in available)
            if found:
                return True, f"✅ Ollama model '{name}' ready"
            return False, (
                f"Model '{name}' not found.\nRun: ollama pull {name}"
            )
        except Exception as e:
            return False, (
                f"Cannot connect to Ollama: {e}\n"
                "Ollama cannot run on Streamlit Cloud. Use a Groq model instead."
            )

    @staticmethod
    def list_models() -> Dict[str, Dict]:
        return SUPPORTED_MODELS

    @staticmethod
    def groq_models() -> Dict[str, Dict]:
        return GROQ_MODELS

    @staticmethod
    def ollama_models() -> Dict[str, Dict]:
        return OLLAMA_MODELS
