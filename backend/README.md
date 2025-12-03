# Mini RAG Retrieval

Simple implementation of the **retrieval** part of a RAG system.  
Loads a tiny text dataset, embeds it with OpenAI, stores vectors in memory, and exposes
an API + minimal UI for semantic search.  
No LLM generation – only retrieval.

--------------------------------------------------
1. Structure
--------------------------------------------------

rag-mini/
├─ backend/
│  ├─ app.py               # FastAPI + in-memory vector store
│  ├─ prepare_dataset.py   # build data/dataset.csv from Kaggle CSV
│  ├─ ingest_to_api.py     # send dataset to the backend (/ingest)
│  ├─ requirements.txt
│  └─ .env                 # contains OPENAI_API_KEY (not committed)
├─ data/
│  ├─ JEOPARDY_CSV.csv     # original Kaggle file (small subset)
│  └─ dataset.csv          # 200-row cleaned sample
└─ frontend/
   ├─ index.html
   ├─ style.css
   └─ app.js

--------------------------------------------------
2. Dataset
--------------------------------------------------

- Source: Kaggle – “200,000+ Jeopardy Questions”.
- I keep a sample of 200 rows in data/dataset.csv.
- Each row is one chunk: the Jeopardy question text.
- Typical queries: “Canadian city”, “Hamlet”, “Swiss city”, etc.

--------------------------------------------------
3. Embeddings & Vector Store
--------------------------------------------------

- Embedding model: text-embedding-3-small (OpenAI).
- During ingestion, all 200 questions are embedded once.
- Vectors are stored in memory in a numpy array (no external DB).
- Similarity: cosine similarity.

Cost for 200 short questions is far below 1 USD.

--------------------------------------------------
4. Backend – How to Run
--------------------------------------------------

From the backend/ folder:

    python -m venv .venv
    .\.venv\Scripts\activate
    pip install -r requirements.txt

Create .env file in backend/:

    OPENAI_API_KEY=sk-...

(Optional) rebuild dataset.csv from JEOPARDY_CSV.csv:

    python prepare_dataset.py

Ingest dataset into the in-memory vector store:

    python ingest_to_api.py

Run the API:

    uvicorn app:app --reload

Endpoints:

- GET /health            – status and number of docs.
- GET /debug_peek?n=3    – first n chunks (ids, texts, metadata).
- POST /ingest           – used by ingest_to_api.py.
- GET /search            – semantic search.

Search endpoint example:

    /search?q=Hamlet&top_k=5&min_score=0.0

The /search flow:

1. Embed query q with text-embedding-3-small.
2. Compute cosine similarity against all stored vectors.
3. Sort by similarity (descending).
4. Take top_k results (filter by min_score if > 0).
5. Return: id, text, score (similarity), metadata.

Example result item:

    {
      "id": "row195_ch0",
      "text": "Funny in 1600 Dept.: When the king ... Hamlet says, \"I am too much in\" this",
      "score": 0.48,
      "metadata": { "row": 195 }
    }

--------------------------------------------------
5. Frontend – How to Run
--------------------------------------------------

From the frontend/ folder:

    python -m http.server 5500

Then open in a browser:

    http://127.0.0.1:5500/index.html

The UI:

- input for the query text
- inputs for Top-K and Min score
- Search button

The page calls the backend /search endpoint and shows a list of retrieved chunks
(ID, score, text).  
The UI only displays retrieved context – it does not generate new text.

--------------------------------------------------
6. Design Choices (short)
--------------------------------------------------

- Small Jeopardy dataset: simple, fits size constraints, naturally question-based.
- One question per chunk: short texts, no need for complex chunking.
- OpenAI embeddings: strong semantic retrieval, very low cost at this scale.
- In-memory vector store: easiest and most transparent solution for 200 rows.
- No LLM: focus is strictly on the retrieval component, as required in the assignment.
