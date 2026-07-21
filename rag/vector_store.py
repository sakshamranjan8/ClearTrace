"""ClearTrace Module 3 — FAISS Vector Store for RAG.

This module handles:
  1. Reading .txt knowledge documents from app/knowledge/
  2. Chunking them into ~500-character pieces (with 50-char overlap)
  3. Embedding chunks using sentence-transformers (all-MiniLM-L6-v2)
  4. Storing embeddings in a FAISS IndexFlatL2 for similarity search
  5. Persisting the index to disk so it doesn't rebuild on every restart

WHY FAISS + sentence-transformers?
  - Both run locally — no API costs, no rate limits
  - all-MiniLM-L6-v2 is only 80 MB and produces 384-dim embeddings
  - FAISS IndexFlatL2 is simple (brute-force L2 distance) — perfect for
    small knowledge bases (<1000 chunks). No training needed.

LLM Zoomcamp connection:
  This is the "retrieval" step of RAG (Retrieval-Augmented Generation).
  The chatbot retrieves relevant knowledge chunks, then passes them to
  the LLM as context so it can give informed, accurate answers.
"""

import json
from pathlib import Path

import numpy as np

from app.config import settings


# ===========================================================================
# Module-level state
# ===========================================================================
_faiss_index = None       # The FAISS index object
_chunks = []              # List of text chunks (parallel to index vectors)
_embedding_model = None   # The sentence-transformers model
_is_ready = False         # True after index is built/loaded


# ===========================================================================
# Public API
# ===========================================================================

def build_or_load_index() -> bool:
    """Build the FAISS index from knowledge docs, or load from disk cache.

    Call this once at server startup. If a cached index exists on disk,
    it loads in <1 second. Otherwise, it reads the .txt files, chunks
    them, embeds them, and saves the result.

    Returns:
        True if the index is ready, False if something went wrong.
    """
    global _is_ready

    # Try loading from disk first (fast path)
    if _load_from_disk():
        _is_ready = True
        return True

    # No cache — build from knowledge docs (slow path, ~10-30 seconds)
    if _build_from_docs():
        _save_to_disk()
        _is_ready = True
        return True

    print("[VECTOR_STORE] Failed to build or load index")
    _is_ready = False
    return False


def search(query: str, top_k: int = 5) -> list[str]:
    """Search the knowledge base for chunks relevant to the query.

    Args:
        query: The user's question (natural language).
        top_k: Number of results to return.

    Returns:
        List of text chunks, ordered by relevance (most relevant first).
        Returns empty list if the index isn't ready.
    """
    if not _is_ready or _faiss_index is None:
        print("[VECTOR_STORE] Index not ready — returning empty results")
        return []

    try:
        # Embed the query using the same model as the documents
        model = _get_embedding_model()
        query_vector = model.encode([query], normalize_embeddings=True)
        query_vector = np.array(query_vector, dtype="float32")

        # Search FAISS for the top_k nearest neighbours
        distances, indices = _faiss_index.search(query_vector, top_k)
 
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(_chunks) and idx >= 0:
                results.append(_chunks[idx])
                print(
                    f"[VECTOR_STORE] Match {i + 1}: distance={distances[0][i]:.3f}, "
                    f"chunk={_chunks[idx][:80]}..."
                )

        return results

    except Exception as e:
        print(f"[VECTOR_STORE] Search error: {e}")
        return []


def get_status() -> dict:
    """Return the current status of the vector store (for /health endpoint)."""
    return {
        "ready": _is_ready,
        "num_chunks": len(_chunks),
        "embedding_model": settings.EMBEDDING_MODEL,
    }


# ===========================================================================
# Internal: embedding model
# ===========================================================================

def _get_embedding_model():
    """Lazy-load the sentence-transformers model.

    The model is ~80 MB and takes 2-5 seconds to load on first use.
    After that, it's cached in memory.
    """
    global _embedding_model

    if _embedding_model is not None:
        return _embedding_model

    print(f"[VECTOR_STORE] Loading embedding model: {settings.EMBEDDING_MODEL}")
    try:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
        print(f"[VECTOR_STORE] Model loaded successfully")
    except Exception as e:
        print(f"[VECTOR_STORE] Failed to load embedding model: {e}")
        raise RuntimeError(f"Embedding model {settings.EMBEDDING_MODEL} could not be loaded") from e

    return _embedding_model


# ===========================================================================
# Internal: chunking
# ===========================================================================

def _chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks for embedding.

    Why chunking?
      - Embedding models have a max input length (~512 tokens for MiniLM).
      - Smaller chunks give more precise retrieval results.
      - Overlap ensures we don't lose context at chunk boundaries.

    Args:
        text: Full document text.
        chunk_size: Target size in characters for each chunk.
        overlap: How many characters to overlap between consecutive chunks.

    Returns:
        List of text chunks.
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        # Try to break at a sentence boundary (period, newline)
        if end < len(text):
            # Look for natural break points near the end of the chunk
            for sep in ["\n\n", "\n", ". ", "? ", "! ", ".", "?", "!"]:
                break_pos = text.rfind(sep, start + chunk_size // 2, end + 100)
                if break_pos > 0:
                    end = break_pos + len(sep)
                    break

        chunk = text[start:end].strip()
        if chunk:  # Skip empty chunks
            chunks.append(chunk)

        # Move forward, minus the overlap
        start = end - overlap

    return chunks


# ===========================================================================
# Internal: build index from docs
# ===========================================================================

def _build_from_docs() -> bool:
    """Read all .txt files from knowledge/, chunk, embed, and index them.

    Returns:
        True if successful, False otherwise.
    """
    global _faiss_index, _chunks

    knowledge_dir = settings.KNOWLEDGE_DIR

    if not knowledge_dir.exists():
        print(f"[VECTOR_STORE] Knowledge directory not found: {knowledge_dir}")
        return False

    # Read all .txt files
    txt_files = sorted(knowledge_dir.glob("*.txt"))
    if not txt_files:
        print(f"[VECTOR_STORE] No .txt files found in {knowledge_dir}")
        return False

    print(f"[VECTOR_STORE] Found {len(txt_files)} knowledge files:")
    all_chunks = []

    for txt_file in txt_files:
        print(f"  - {txt_file.name}")
        text = txt_file.read_text(encoding="utf-8")
        file_chunks = _chunk_text(text)
        # Prepend source filename to each chunk for provenance
        for chunk in file_chunks:
            all_chunks.append(f"[Source: {txt_file.stem}] {chunk}")

    print(f"[VECTOR_STORE] Created {len(all_chunks)} chunks total")

    if not all_chunks:
        return False

    _chunks = all_chunks

    # Embed all chunks
    print("[VECTOR_STORE] Embedding chunks (this takes 10-30 seconds)...")
    model = _get_embedding_model()

    try:
        embeddings = model.encode(all_chunks, normalize_embeddings=True, show_progress_bar=True)
    except TypeError:
    # Fallback for older versions
        embeddings = model.encode(all_chunks, show_progress_bar=settings.DEBUG_MODE)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    embeddings = np.array(embeddings, dtype="float32")

    print(f"[VECTOR_STORE] Embedding shape: {embeddings.shape}")

    # Build FAISS index
    import faiss
    dimension = embeddings.shape[1]  # 384 for MiniLM
    _faiss_index = faiss.IndexFlatL2(dimension)
    _faiss_index.add(embeddings)

    print(f"[VECTOR_STORE] FAISS index built: {_faiss_index.ntotal} vectors")

    return True


# ===========================================================================
# Internal: persistence (save/load index to/from disk)
# ===========================================================================

def _save_to_disk() -> bool:
    """Save the FAISS index and chunks to disk for fast reloading.

    Files saved:
      data/faiss_index/index.faiss  — the FAISS binary index
      data/faiss_index/chunks.json  — the text chunks (parallel to index)
    """
    if _faiss_index is None or not _chunks:
        return False

    try:
        import faiss

        index_dir = settings.FAISS_INDEX_DIR
        index_dir.mkdir(parents=True, exist_ok=True)

        index_path = index_dir / "index.faiss"
        chunks_path = index_dir / "chunks.json"

        faiss.write_index(_faiss_index, str(index_path))
        chunks_path.write_text(json.dumps(_chunks, ensure_ascii=False), encoding="utf-8")

        print(f"[VECTOR_STORE] Saved index to {index_path}")
        print(f"[VECTOR_STORE] Saved {len(_chunks)} chunks to {chunks_path}")
        return True

    except Exception as e:
        print(f"[VECTOR_STORE] Failed to save index: {e}")
        return False


def _load_from_disk() -> bool:
    """Load a previously saved FAISS index from disk.

    Returns:
        True if loaded successfully, False if no cache exists.
    """
    global _faiss_index, _chunks

    index_path = settings.FAISS_INDEX_DIR / "index.faiss"
    chunks_path = settings.FAISS_INDEX_DIR / "chunks.json"

    if not index_path.exists() or not chunks_path.exists():
        print("[VECTOR_STORE] No cached index found — will build from docs")
        return False

    try:
        import faiss

        _faiss_index = faiss.read_index(str(index_path))
        _chunks = json.loads(chunks_path.read_text(encoding="utf-8"))

        print(
            f"[VECTOR_STORE] Loaded cached index: "
            f"{_faiss_index.ntotal} vectors, {len(_chunks)} chunks"
        )

        # Verify consistency
        if _faiss_index.ntotal != len(_chunks):
            print("[VECTOR_STORE] Index/chunks mismatch — rebuilding")
            _faiss_index = None
            _chunks = []
            return False

        # Still need the embedding model for search queries
        _get_embedding_model()

        return True

    except Exception as e:
        print(f"[VECTOR_STORE] Failed to load cached index: {e}")
        return False
