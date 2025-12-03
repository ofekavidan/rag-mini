import os
from typing import List, Dict, Any

import numpy as np
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

# --------- ENV & OpenAI client ---------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is missing in .env")

client = OpenAI(api_key=OPENAI_API_KEY)

# --------- In-memory vector store ---------
TEXTS: List[str] = []
IDS: List[str] = []
METAS: List[Dict[str, Any]] = []
VECS: np.ndarray | None = None  # shape (N, D)


def embed_text_batch(texts: List[str]) -> List[List[float]]:
    """Use OpenAI to embed a batch of texts."""
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    return [d.embedding for d in resp.data]


def add_embeddings(ids: List[str], texts: List[str], metas: List[Dict[str, Any]]):
    """Add new texts + embeddings into in-memory store."""
    global TEXTS, IDS, METAS, VECS
    embs = embed_text_batch(texts)  # list[list[float]]
    arr = np.array(embs, dtype="float32")  # (batch, D)

    if VECS is None:
        VECS = arr
    else:
        VECS = np.vstack([VECS, arr])

    TEXTS.extend(texts)
    IDS.extend(ids)
    METAS.extend(metas)


def cosine_sim(query_vec: np.ndarray, mat: np.ndarray) -> np.ndarray:
    """Cosine similarity between query (D,) and mat (N,D)."""
    if mat.size == 0:
        return np.array([])
    q = query_vec / (np.linalg.norm(query_vec) + 1e-8)
    m = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8)
    return m @ q


# --------- FastAPI app ---------
app = FastAPI(title="Mini RAG Retriever (Simple Vector Store)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------- Models ---------
class IngestRequest(BaseModel):
    records: List[Dict[str, Any]]


class QueryResponseChunk(BaseModel):
    id: str
    text: str
    score: float
    metadata: Dict[str, Any]


# --------- Routes ---------
@app.get("/health")
def health():
    return {"status": "ok", "docs": len(TEXTS)}


@app.get("/debug_peek")
def debug_peek(n: int = 3):
    return {
        "count": len(TEXTS),
        "ids": IDS[:n],
        "documents": TEXTS[:n],
        "metadatas": METAS[:n],
    }


@app.post("/ingest")
def ingest(req: IngestRequest):
    try:
        ids = [r["id"] for r in req.records]
        texts = [r["text"] for r in req.records]
        metas = [{"row": r.get("row", None)} for r in req.records]

        add_embeddings(ids, texts, metas)
        return {"inserted": len(ids)}
    except Exception as e:
        print("INGEST ERROR:", e, flush=True)
        raise HTTPException(status_code=400, detail=f"ingest_failed: {e}")


@app.get("/search", response_model=List[QueryResponseChunk])
def search(
    q: str = Query(..., min_length=1),
    top_k: int = 5,
    min_score: float = 0.0,
):
    global VECS
    if VECS is None or len(TEXTS) == 0:
        return []

    try:
        # Embed query
        q_vec = np.array(embed_text_batch([q])[0], dtype="float32")  # (D,)
        sims = cosine_sim(q_vec, VECS)  # (N,)

        # Sort by similarity, descending
        order = np.argsort(-sims)

        results: List[QueryResponseChunk] = []
        for idx in order[:top_k]:
            score = float(sims[idx])  # cosine similarity in [-1, 1]
            if min_score > 0 and score < min_score:
                continue
            results.append(QueryResponseChunk(
                id=IDS[idx],
                text=TEXTS[idx],
                score=score,
                metadata=METAS[idx] or {},
            ))

        return results
    except Exception as e:
        print("SEARCH ERROR:", e, flush=True)
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"search_failed: {e}")
