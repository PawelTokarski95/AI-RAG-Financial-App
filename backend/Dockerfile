FROM python:3.10-slim

WORKDIR /app

# Instalacja kompilatora
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Pobieramy plik wskazując jego dokładną ścieżkę w projekcie
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Kopiujemy zawartość folderu backend bezpośrednio do /app
COPY backend/ .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]