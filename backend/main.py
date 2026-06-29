import os
import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from RAG import Indexer, Retriever
from LLM import LLM, rewrite_query
from sentence_transformers import SentenceTransformer
from Tickers import tickers

app = FastAPI(title="Financial RAG API")


class Message(BaseModel):
    role: str
    content: str


class QueryRequest(BaseModel):
    query: str
    history: List[Message] = []


model = None
retriever = None


@app.on_event("startup")
def startup_event():
    global model, retriever

    model = SentenceTransformer("intfloat/multilingual-e5-small")
    indexer = Indexer(storage_path="backend/storage")
    CACHE_DIR = "backend/storage"

    if os.path.exists(CACHE_DIR) and len(os.listdir(CACHE_DIR)) > 0:
        indexes, chunks = indexer.load()
    else:
        texts_by_ticker = {}
        for ticker in tickers:
            path = f"backend/processed-filings/{ticker}_clean.txt"
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    texts_by_ticker[ticker] = f.read()

        indexes, chunks = indexer.build(texts_by_ticker)

    retriever = Retriever(indexes, chunks, model)
    print("Backend initialized successfully.")


@app.post("/ask")
def ask_financial_bot(request: QueryRequest):
    try:
        history_dicts = [{"role": m.role, "content": m.content} for m in request.history]

        search_query = rewrite_query(request.query, history_dicts)

        results = retriever.retrieve(search_query)
        context = "\n\n".join([f"[{ticker}] {text}" for ticker, text, score in results])

        answer = LLM(query=request.query, context=context, history=history_dicts)

        return {"answer": answer, "context_used": context}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
