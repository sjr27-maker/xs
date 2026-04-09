# NCERT FAISS index + prerequisite graph
# memory/knowledge_store.py
"""
NCERT content store with FAISS-based retrieval.
Loads PDF chunks, builds embedding index.
Called by live_session.py for RAG context in prompts.

Usage:
  from memory.knowledge_store import retrieve_context
  ctx = retrieve_context("what is factorization", "Mathematics", 10)
"""
import os
import json
import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger("SYRA.KnowledgeStore")

# Index paths
INDEX_PATH  = Path("data/faiss_index.pkl")
CHUNKS_DIR  = Path("data/ncert_chunks")

# Lazy-loaded globals
_index       = None
_chunks      = []
_embedder    = None


def _get_embedder():
    """Lazy-load Gemini embedding client."""
    global _embedder
    if _embedder is None:
        from google import genai
        from dotenv import load_dotenv
        load_dotenv()
        _embedder = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    return _embedder


def _embed_text(text: str) -> Optional[np.ndarray]:
    """Embed a single text string using Gemini."""
    try:
        client   = _get_embedder()
        response = client.models.embed_content(
            model="models/text-embedding-004",
            contents=text,
        )
        return np.array(response.embeddings[0].values, dtype=np.float32)
    except Exception as e:
        logger.debug(f"Embedding error: {e}")
        return None


def _load_index():
    """Load FAISS index and chunks from disk."""
    global _index, _chunks

    if _index is not None:
        return True   # already loaded

    if not INDEX_PATH.exists():
        logger.info("No FAISS index found — RAG disabled. "
                    "Run add_ncert_pdf() to build one.")
        return False

    try:
        import faiss
        with open(INDEX_PATH, "rb") as f:
            data    = pickle.load(f)
            _index  = data["index"]
            _chunks = data["chunks"]
        logger.info(f"FAISS index loaded: {len(_chunks)} chunks")
        return True
    except ImportError:
        logger.warning("faiss-cpu not installed — RAG disabled. "
                       "pip install faiss-cpu")
        return False
    except Exception as e:
        logger.warning(f"Index load failed: {e}")
        return False


def retrieve_context(
        query:   str,
        subject: str,
        grade:   int,
        top_k:   int = 3,
) -> str:
    """
    Retrieve relevant NCERT content for the given query.
    Returns empty string if no index or no relevant results.
    Called in the main session loop — must be fast.
    """
    if not query or not _load_index():
        return ""

    query_vec = _embed_text(f"{subject} Class {grade}: {query}")
    if query_vec is None:
        return ""

    try:
        import faiss
        distances, indices = _index.search(
            query_vec.reshape(1, -1), top_k
        )

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(_chunks):
                continue
            # Filter by relevance threshold
            if dist > 1.5:
                continue
            chunk = _chunks[idx]
            # Filter by subject/grade if metadata available
            meta = chunk.get("metadata", {})
            if meta.get("subject") and meta["subject"].lower() != subject.lower():
                continue
            if meta.get("grade") and str(meta["grade"]) != str(grade):
                continue
            results.append(chunk["text"])

        if not results:
            return ""

        return "\n\n".join(results[:2])   # top 2 most relevant

    except Exception as e:
        logger.debug(f"Retrieval error: {e}")
        return ""


def add_ncert_pdf(
        pdf_path: str,
        subject:  str,
        grade:    int,
        chunk_size: int = 400,
        overlap:    int = 80,
) -> int:
    """
    Process a NCERT PDF and add to the knowledge store.
    Returns number of chunks added.
    Run this offline before student sessions.

    Example:
      add_ncert_pdf("maths_class10.pdf", "Mathematics", 10)
      rebuild_index()
    """
    try:
        import pypdf
    except ImportError:
        logger.error("pypdf not installed — pip install pypdf")
        return 0

    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    reader     = pypdf.PdfReader(pdf_path)
    full_text  = " ".join(
        page.extract_text() or "" for page in reader.pages
    )
    # Clean whitespace
    import re
    full_text = re.sub(r'\s+', ' ', full_text).strip()

    words  = full_text.split()
    chunks_added = 0

    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        if len(chunk_words) < 50:
            continue

        chunk_text = " ".join(chunk_words)
        chunk_id   = f"{subject.lower()}_{grade}_{i}"
        chunk_path = CHUNKS_DIR / f"{chunk_id}.json"

        chunk_data = {
            "text":     chunk_text,
            "metadata": {
                "subject": subject,
                "grade":   grade,
                "source":  Path(pdf_path).name,
                "offset":  i,
            }
        }
        chunk_path.write_text(
            json.dumps(chunk_data, ensure_ascii=False),
            encoding="utf-8"
        )
        chunks_added += 1

    logger.info(f"Added {chunks_added} chunks from {pdf_path}")
    return chunks_added


def rebuild_index():
    """
    Rebuild FAISS index from all chunks in data/ncert_chunks/.
    Run after add_ncert_pdf(). Takes a few minutes for large datasets.
    """
    global _index, _chunks

    try:
        import faiss
    except ImportError:
        logger.error("faiss-cpu not installed — pip install faiss-cpu")
        return

    chunk_files = list(CHUNKS_DIR.glob("*.json"))
    if not chunk_files:
        logger.warning("No chunks found — run add_ncert_pdf() first")
        return

    logger.info(f"Building index from {len(chunk_files)} chunks...")

    all_chunks   = []
    all_vectors  = []
    failed       = 0

    for path in chunk_files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            vec  = _embed_text(data["text"][:500])
            if vec is not None:
                all_chunks.append(data)
                all_vectors.append(vec)
            else:
                failed += 1
        except Exception as e:
            logger.debug(f"Chunk error {path}: {e}")
            failed += 1

    if not all_vectors:
        logger.error("No vectors generated — check Gemini API key")
        return

    dim    = len(all_vectors[0])
    matrix = np.stack(all_vectors).astype(np.float32)

    index = faiss.IndexFlatL2(dim)
    index.add(matrix)

    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(INDEX_PATH, "wb") as f:
        pickle.dump({"index": index, "chunks": all_chunks}, f)

    _index  = index
    _chunks = all_chunks

    logger.info(
        f"Index built: {len(all_chunks)} chunks, "
        f"dim={dim}, failed={failed}"
    )
    print(f"  Index ready: {len(all_chunks)} NCERT chunks indexed.")