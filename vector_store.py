"""
vector_store.py

Stores and retrieves analysis reports from ChromaDB using
sentence-transformers for embeddings (free, local, no API key needed).
Works alongside your existing Gemini + CrewAI setup.
"""

import os
import chromadb
from datetime import datetime
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# ── Where ChromaDB saves data on disk ──────────────────────────────
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")

# ── Free local embeddings — no API key needed ──────────────────────
# all-MiniLM-L6-v2 is small (80MB), fast, and good enough for
# semantic search over business reports
embedding_fn = SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)


def _get_collection():
    """Gets or creates the persistent ChromaDB collection."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_or_create_collection(
        name="analysis_reports",
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )


# ────────────────────────────────────────────────────────────────────
# STORE
# ────────────────────────────────────────────────────────────────────
def store_report(report_text: str, dataset_name: str, username: str, chunk_size: int = 500):
    """
    Chunks the report and stores each chunk as a vector in ChromaDB.
    Metadata is stored per chunk so you can filter by user/dataset later.
    """
    collection = _get_collection()
    chunks     = _chunk_text(report_text, chunk_size)
    timestamp  = datetime.utcnow().isoformat()

    ids, documents, metadatas = [], [], []

    for i, chunk in enumerate(chunks):
        ids.append(f"{username}__{dataset_name}__{timestamp}__chunk{i}")
        documents.append(chunk)
        metadatas.append({
            "dataset_name": dataset_name,
            "username":     username,
            "timestamp":    timestamp,
            "chunk_index":  i,
            "total_chunks": len(chunks),
        })

    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    return len(chunks)


def _chunk_text(text: str, chunk_size: int) -> list[str]:
    """Splits text into overlapping chunks so context isn't cut off at boundaries."""
    overlap = chunk_size // 10   # 10% overlap
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start : start + chunk_size])
        start += chunk_size - overlap
    return [c for c in chunks if c.strip()]


# ────────────────────────────────────────────────────────────────────
# RETRIEVE
# ────────────────────────────────────────────────────────────────────
def query_reports(query: str, username: str, n_results: int = 5, dataset_name: str = None) -> list[dict]:
    """
    Finds the most semantically similar chunks to the query.
    Always filters by username so users only see their own reports.
    """
    collection = _get_collection()

    where = {"username": username}
    if dataset_name:
        where = {"$and": [{"username": username}, {"dataset_name": dataset_name}]}

    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    output = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        output.append({
            "text":         doc,
            "dataset_name": meta.get("dataset_name"),
            "timestamp":    meta.get("timestamp"),
            "score":        round(1 - dist, 3),  # 1.0 = identical, 0.0 = unrelated
        })

    return output


# ────────────────────────────────────────────────────────────────────
# LIST & DELETE
# ────────────────────────────────────────────────────────────────────
def list_stored_reports(username: str) -> list[str]:
    """Returns unique dataset names this user has stored."""
    collection = _get_collection()
    results = collection.get(where={"username": username}, include=["metadatas"])
    names = {m["dataset_name"] for m in results["metadatas"]}
    return sorted(names)


def delete_report(username: str, dataset_name: str) -> int:
    """Deletes all chunks for a given user + dataset. Returns count deleted."""
    collection = _get_collection()
    results = collection.get(
        where={"$and": [{"username": username}, {"dataset_name": dataset_name}]},
        include=["metadatas"],
    )
    if results["ids"]:
        collection.delete(ids=results["ids"])
    return len(results["ids"])