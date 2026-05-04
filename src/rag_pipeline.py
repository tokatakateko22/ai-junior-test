from __future__ import annotations

from pathlib import Path
from typing import Iterable

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import (
    DATA_DIR,
    EMBEDDING_MODEL_NAME,
    RETRIEVAL_TOP_K,
    VECTORSTORE_DIR,
)


class NovaBiteRAGStore:
    def __init__(self) -> None:
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"local_files_only": True},
        )
        self.vectorstore: FAISS | None = None

    def ingest_documents(self, source_dir: Path = DATA_DIR) -> FAISS:
        docs = self._load_documents(source_dir)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=80,
            separators=["\n## ", "\n### ", "\n- ", "\n", " ", ""],
        )
        chunks = splitter.split_documents(docs)

        for idx, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = idx

        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        return self.vectorstore

    def save_local(self, target_dir: Path = VECTORSTORE_DIR) -> None:
        if self.vectorstore is None:
            raise ValueError("Vector store is not initialized.")
        target_dir.mkdir(parents=True, exist_ok=True)
        self.vectorstore.save_local(str(target_dir))

    def load_or_create(self) -> FAISS:
        VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
        index_file = VECTORSTORE_DIR / "index.faiss"
        if index_file.exists():
            self.vectorstore = FAISS.load_local(
                str(VECTORSTORE_DIR),
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
            return self.vectorstore

        self.ingest_documents(DATA_DIR)
        self.save_local(VECTORSTORE_DIR)
        if self.vectorstore is None:
            raise ValueError("Failed to initialize vector store.")
        return self.vectorstore

    def retrieve(self, query: str) -> list[Document]:
        if self.vectorstore is None:
            self.load_or_create()

        assert self.vectorstore is not None
        scored_docs = self.vectorstore.similarity_search_with_score(
            query, k=max(RETRIEVAL_TOP_K * 2, 6)
        )
        query_terms = {w.lower() for w in query.split() if len(w) > 3}

        preferred: list[Document] = []
        fallback: list[Document] = []
        for doc, _score in scored_docs:
            content_words = set(doc.page_content.lower().split())
            if query_terms and query_terms.intersection(content_words):
                preferred.append(doc)
            else:
                fallback.append(doc)

        docs = preferred + fallback
        return docs[:RETRIEVAL_TOP_K]

    @staticmethod
    def _load_documents(source_dir: Path) -> list[Document]:
        files = sorted(source_dir.glob("*.md"))
        docs: list[Document] = []
        for file in files:
            loader = TextLoader(str(file), encoding="utf-8")
            loaded_docs = loader.load()
            for doc in loaded_docs:
                doc.metadata["source"] = file.name
            docs.extend(loaded_docs)
        return docs

    @staticmethod
    def format_context(docs: Iterable[Document]) -> str:
        parts = []
        for d in docs:
            source = d.metadata.get("source", "unknown")
            chunk_id = d.metadata.get("chunk_id", "n/a")
            parts.append(f"[{source}#chunk-{chunk_id}]\n{d.page_content}")
        return "\n\n".join(parts)
