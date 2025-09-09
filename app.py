# app.py
import os
import json
import time
import sqlite3
from pathlib import Path
from typing import List, Dict

from flask import Flask, request, jsonify, abort
import numpy as np
import faiss
import requests

from sentence_transformers import SentenceTransformer, CrossEncoder

# ---------- CONFIG ----------
FAISS_INDEX_PATH = os.environ.get("FAISS_INDEX_PATH", "/content/faiss_index1.bin")
METADATA_PATH = os.environ.get("METADATA_PATH", "/content/faiss_metadata1 (1).json")
EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL_NAME", "all-mpnet-base-v2")
RERANKER_MODEL = os.environ.get("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")


# Mistral config (set env vars or edit below)
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "WY7aNsxuO96YRQlhzN5zwkzbEWQWzY33")
MISTRAL_MODEL_NAME = os.environ.get("MISTRAL_MODEL_NAME", "mistral-medium-2505")
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"
MISTRAL_HEADERS = {"Authorization": f"Bearer {MISTRAL_API_KEY}", "Content-Type": "application/json"}

TOP_K_FINAL = 2
FAISS_K = 10
VERIFIER_TOPK = 3
SUPPORT_THRESHOLD = float(os.environ.get("SUPPORT_THRESHOLD", 0.60))

# Memory DB path
MEM_DB = os.environ.get("MEM_DB", "memory.db")
MEMORY_WINDOW = int(os.environ.get("MEMORY_WINDOW", 3))  # how many recent Q/A to include

# ---------- APP ----------
app = Flask(__name__)

# ---------- Load FAISS index + metadata ----------
if not Path(METADATA_PATH).exists():
    raise RuntimeError(f"Metadata JSON not found at {METADATA_PATH}. Please point METADATA_PATH correctly.")
with open(METADATA_PATH, "r", encoding="utf-8") as f:
    meta = json.load(f)
ids = meta.get("ids", [])
chunks = meta.get("chunks", [])
print(f"Loaded metadata: {len(ids)} chunks.")

if Path(FAISS_INDEX_PATH).exists():
    index = faiss.read_index(FAISS_INDEX_PATH)
    print("FAISS index loaded from", FAISS_INDEX_PATH)
else:
    raise RuntimeError("FAISS index file not found. Build index first and set FAISS_INDEX_PATH.")

# ---------- Load models ----------
print("Loading embedder:", EMBED_MODEL_NAME)
embedder = SentenceTransformer(EMBED_MODEL_NAME)
EMBED_DIM = embedder.get_sentence_embedding_dimension()
print("Embedding dimension:", EMBED_DIM)

print("Loading reranker:", RERANKER_MODEL)
reranker = CrossEncoder(RERANKER_MODEL)

# ---------- Memory DB helpers ----------
def init_memory_db():
    conn = sqlite3.connect(MEM_DB)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            question TEXT,
            answer TEXT,
            ts INTEGER
        )
    """)
    conn.commit()
    conn.close()

def add_memory(user_id: str, question: str, answer: str):
    conn = sqlite3.connect(MEM_DB)
    cur = conn.cursor()
    cur.execute("INSERT INTO memory (user_id, question, answer, ts) VALUES (?, ?, ?, ?)",
                (user_id, question, answer, int(time.time())))
    conn.commit()
    conn.close()

def get_recent_memory(user_id: str, n: int = MEMORY_WINDOW):
    conn = sqlite3.connect(MEM_DB)
    cur = conn.cursor()
    cur.execute("SELECT question, answer, ts FROM memory WHERE user_id = ? ORDER BY ts DESC LIMIT ?", (user_id, n))
    rows = cur.fetchall()
    conn.close()
    # return in chronological order (oldest first)
    rows = list(reversed(rows))
    return [{"q": r[0], "a": r[1], "ts": r[2]} for r in rows]

def clear_memory(user_id: str):
    conn = sqlite3.connect(MEM_DB)
    cur = conn.cursor()
    cur.execute("DELETE FROM memory WHERE user_id = ?", (user_id,))
    conn.commit()
    conn.close()

# Initialize DB at startup
init_memory_db()

# ---------- Retrieval + rerank helpers ----------
def embed_query(q: str):
    vec = embedder.encode([q], convert_to_numpy=True)
    vec = vec.astype('float32')
    # normalize
    vec = vec / np.linalg.norm(vec, axis=1, keepdims=True)
    return vec

def retrieve_faiss(qvec: np.ndarray, k: int = FAISS_K):
    D, I = index.search(qvec.astype('float32'), k)
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(ids):
            continue
        results.append({"index": int(idx), "faiss_score": float(score), "meta": chunks[int(idx)]})
    return results

def rerank_candidates(query: str, candidates: List[Dict], final_k: int = TOP_K_FINAL):
    texts = [c["meta"]["text"] for c in candidates]
    if len(texts) == 0:
        return []
    # CrossEncoder expects pair input
    pairs = [[query, t] for t in texts]
    scores = reranker.predict(pairs)
    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)
    sorted_c = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    out = [{"id": ids[c["index"]], "score": c["rerank_score"], "meta": c["meta"]} for c in sorted_c[:final_k]]
    return out

# ---------- Prompt composition ----------
SYSTEM_INSTR = (
    "You are a legal research assistant for Indian law. Use ONLY the provided context chunks and the user's memory context."
    " If the answer is not fully supported by the context, say 'I don't know; consult a qualified lawyer'. "
    "For each claim, include a citation in the format: [source: <source>, heading: <heading>, pages: <page_start>-<page_end>]."
)

def compose_messages(question: str, retrieved: List[Dict], memory: List[Dict]):
    # Build context string from retrieved chunks
    ctx_parts = []
    for i, r in enumerate(retrieved, start=1):
        m = r["meta"]
        txt = m.get("text","")
        head = m.get("heading","")
        ctx = f"[[CHUNK {i}]] source: {m.get('source','?')} | heading: {head} | pages: {m.get('page_start','?')}-{m.get('page_end','?')}\n{txt}\n"
        ctx_parts.append(ctx)
    context = "\n\n".join(ctx_parts)

    # build memory part
    mem_parts = []
    for mem in memory:
        mem_parts.append(f"User asked: {mem['q']}\nAssistant answered: {mem['a']}\n")
    mem_context = "\n\n".join(mem_parts)

    system_msg = {"role": "system", "content": SYSTEM_INSTR}
    # user content: include memory then context then question
    user_content = ""
    if mem_context:
        user_content += "PREVIOUS CONVERSATIONS:\n" + mem_context + "\n\n"
    user_content += "CONTEXT:\n" + context + "\n\nQUESTION:\n" + question + "\n\nAnswer:"
    user_msg = {"role": "user", "content": user_content}
    return [system_msg, user_msg]

# ---------- Mistral API wrapper ----------
def call_mistral_api(messages, max_tokens=512, temperature=0.1):
    if not MISTRAL_API_KEY or "<YOUR" in MISTRAL_API_KEY:
        raise RuntimeError("Set MISTRAL_API_KEY environment variable.")
    payload = {
        "model": MISTRAL_MODEL_NAME,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    resp = requests.post(MISTRAL_API_URL, headers=MISTRAL_HEADERS, json=payload, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"Mistral API error {resp.status_code}: {resp.text}")
    out = resp.json()
    # parse typical response
    if isinstance(out, dict) and "choices" in out and len(out["choices"])>0:
        choice = out["choices"][0]
        if "message" in choice and "content" in choice["message"]:
            return choice["message"]["content"]
        if "text" in choice:
            return choice["text"]
    # fallback
    return json.dumps(out)

# ---------- Verifier (top-K any) ----------
def split_sentences(text: str):
    import re
    parts = re.split(r'(?<=[\.\?\;])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]

def verify_answer_support_any(answer_text: str, topk_for_claim:int = VERIFIER_TOPK, threshold: float = SUPPORT_THRESHOLD):
    sents = split_sentences(answer_text)
    if not sents:
        return []
    sent_embs = embedder.encode(sents, convert_to_numpy=True)
    if sent_embs.ndim==1:
        sent_embs = sent_embs.reshape(1, -1)
    if sent_embs.dtype != np.float32:
        sent_embs = sent_embs.astype('float32')
    # normalize
    sent_embs = sent_embs / np.linalg.norm(sent_embs, axis=1, keepdims=True)
    D, I = index.search(sent_embs, topk_for_claim)
    reports = []
    for i, sent in enumerate(sents):
        rowD = D[i]
        rowI = I[i]
        best_idx = int(rowI.argmax()) if len(rowI)>0 else -1
        best_score = float(rowD.max()) if len(rowD)>0 else 0.0
        supported = best_score >= threshold
        support_meta = chunks[int(rowI[rowD.argmax()])] if (len(rowI)>0 and int(rowI[rowD.argmax()]) < len(chunks)) else None
        reports.append({
            "sentence": sent,
            "supported": bool(supported),
            "best_score": best_score,
            "supporting_chunk_id": support_meta.get("id") if support_meta else None,
            "supporting_heading": support_meta.get("heading") if support_meta else None,
            "supporting_snippet": (support_meta.get("text","")[:300] if support_meta else None)
        })
    return reports

# ---------- API endpoints ----------
@app.route("/query", methods=["POST"])
def query():
    body = request.get_json(force=True)
    user_id = body.get("user_id") or "anonymous"
    question = body.get("question")
    if not question or not question.strip():
        return jsonify({"error":"question required"}), 400

    # 1) memory
    memory = get_recent_memory(user_id, MEMORY_WINDOW)

    # 2) retrieve + rerank
    qvec = embed_query(question)
    faiss_cands = retrieve_faiss(qvec, k=FAISS_K)
    reranked = rerank_candidates(question, faiss_cands, final_k=TOP_K_FINAL)

    # 3) compose prompt including memory
    messages = compose_messages(question, reranked, memory)

    # 4) call LLM
    try:
        answer = call_mistral_api(messages, max_tokens=512, temperature=0.1)
    except Exception as e:
        return jsonify({"error": f"LLM call failed: {e}"}), 500

    # 5) verifier
    verifier = verify_answer_support_any(answer, topk_for_claim=VERIFIER_TOPK, threshold=SUPPORT_THRESHOLD)

    # 6) store memory (append)
    try:
        add_memory(user_id, question, answer)
    except Exception as e:
        # non-fatal
        print("Warning: memory write failed:", e)

    # 7) prepare retrieved simplified output
    retrieved_out = []
    for r in reranked:
        m = r["meta"]
        retrieved_out.append({
            "id": r["id"],
            "score": r["score"],
            "heading": m.get("heading"),
            "source": m.get("source"),
            "page_start": m.get("page_start"),
            "page_end": m.get("page_end"),
            "snippet": (m.get("text","")[:500])
        })

    return jsonify({
        "question": question,
        "answer": answer,
        "retrieved": retrieved_out,
        "verifier": verifier,
        "memory_used": memory
    }), 200

@app.route("/memory/list", methods=["GET"])
def memory_list():
    user_id = request.args.get("user_id", "anonymous")
    mem = get_recent_memory(user_id, MEMORY_WINDOW)
    return jsonify({"user_id": user_id, "memory": mem})

@app.route("/memory/clear", methods=["POST"])
def memory_clear():
    body = request.get_json(force=True)
    user_id = body.get("user_id", "anonymous")
    clear_memory(user_id)
    return jsonify({"status":"cleared", "user_id": user_id})

# ---------- run ----------
if __name__ == "__main__":
    # dev mode
    app.run(host="0.0.0.0", port=8001, debug=False, use_reloader=False)
