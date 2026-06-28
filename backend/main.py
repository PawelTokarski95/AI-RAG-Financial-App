# backend/main.py
import os
import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

# Dodanie ścieżki do sys.path, aby Docker widział moduły poprawnie
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from RAG import Indexer, Retriever
from LLM import LLM, rewrite_query
from sentence_transformers import SentenceTransformer
from Tickers import tickers

app = FastAPI(title="Financial RAG API")


# Modele Pydantic do obsługi żądań i odpowiedzi
class Message(BaseModel):
    role: str
    content: str


class QueryRequest(BaseModel):
    query: str
    history: List[Message] = []


# Globalne zmienne dla wyszukiwarki
model = None
retriever = None


@app.on_event("startup")
def startup_event():
    global model, retriever

    # Inicjalizacja modeli i indeksów przy starcie aplikacji
    model = SentenceTransformer("intfloat/multilingual-e5-small")
    indexer = Indexer(storage_path="backend/storage")
    CACHE_DIR = "backend/storage"

    if os.path.exists(CACHE_DIR) and len(os.listdir(CACHE_DIR)) > 0:
        indexes, chunks = indexer.load()
    else:
        # Zakładamy, że przetworzone pliki tekstowe są w backend/processed-filings
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
        # Konwersja historii z formatu Pydantic na czyste słowniki dla OpenAI
        history_dicts = [{"role": m.role, "content": m.content} for m in request.history]

        # 1. Przepisanie pytania
        search_query = rewrite_query(request.query, history_dicts)

        # 2. Retrieval
        results = retriever.retrieve(search_query)
        context = "\n\n".join([f"[{ticker}] {text}" for ticker, text, score in results])

        # 3. Odpowiedź z LLM
        answer = LLM(query=request.query, context=context, history=history_dicts)

        return {"answer": answer, "context_used": context}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))