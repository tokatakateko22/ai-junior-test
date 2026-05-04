from __future__ import annotations

from src.rag_pipeline import NovaBiteRAGStore


def build() -> None:
    store = NovaBiteRAGStore()
    store.ingest_documents()
    store.save_local()
    print("Vector index built at data/vectorstore/")


if __name__ == "__main__":
    build()
