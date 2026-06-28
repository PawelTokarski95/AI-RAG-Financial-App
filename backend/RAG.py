import os
import faiss
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np
import re
from sentence_transformers import SentenceTransformer
import pickle

class Indexer:
    def __init__(self, storage_path="backend/storage"):
        self.model = SentenceTransformer("intfloat/multilingual-e5-small")

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )

        self.storage_path = storage_path

        self.indexes = {}
        self.chunks = {}

    def build(self, texts_by_ticker: dict):

        for ticker, text in texts_by_ticker.items():

            ticker_path = os.path.join(self.storage_path, ticker)
            os.makedirs(ticker_path, exist_ok=True)

            text = re.sub(r"[^\w\s\.\,\%\$\-\(\)\:\n]", " ", text)
            text = re.sub(r"\s+", " ", text).strip()

            chunks = self.splitter.split_text(text)

            embeddings = self.model.encode(
                chunks,
                convert_to_numpy=True
            ).astype(np.float32)

            embeddings = embeddings / np.linalg.norm(
                embeddings, axis=1, keepdims=True
            )

            dim = embeddings.shape[1]
            index = faiss.IndexFlatIP(dim)
            index.add(embeddings)


            faiss.write_index(
                index,
                os.path.join(ticker_path, "faiss.index")
            )

            with open(os.path.join(ticker_path, "chunks.pkl"), "wb") as f:
                pickle.dump(chunks, f)

            self.indexes[ticker] = index
            self.chunks[ticker] = chunks

        return self.indexes, self.chunks

    def load(self):

        for ticker_folder in os.listdir(self.storage_path):

            ticker_path = os.path.join(self.storage_path, ticker_folder)

            if not os.path.isdir(ticker_path):
                continue

            index_path = os.path.join(ticker_path, "faiss.index")
            chunks_path = os.path.join(ticker_path, "chunks.pkl")

            if os.path.exists(index_path) and os.path.exists(chunks_path):

                index = faiss.read_index(index_path)

                with open(chunks_path, "rb") as f:
                    chunks = pickle.load(f)

                self.indexes[ticker_folder] = index
                self.chunks[ticker_folder] = chunks

        return self.indexes, self.chunks

class Retriever:
    def __init__(self, indexes, chunks, model):
        self.indexes = indexes
        self.chunks = chunks
        self.model = model

    def retrieve(self, query: str, k=5):

        query_emb = self.model.encode(
            query,
            convert_to_numpy=True
        ).astype(np.float32)

        query_emb = query_emb / np.linalg.norm(query_emb)
        query_emb = query_emb.reshape(1, -1)

        results = []

        for ticker in self.indexes:

            index = self.indexes[ticker]
            chunks = self.chunks[ticker]

            distances, indices = index.search(query_emb, k)

            for i, idx in enumerate(indices[0]):
                if 0 <= idx < len(chunks):
                    results.append(
                        (ticker, chunks[idx], float(distances[0][i]))
                    )

        results.sort(key=lambda x: x[2], reverse=True)

        return results[:k]