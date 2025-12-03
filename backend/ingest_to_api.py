import pandas as pd
import requests
from pathlib import Path

DATA = Path("../data/dataset.csv")
API = "http://127.0.0.1:8000/ingest"

def chunk_text(text, chunk_size=400, overlap=80):
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end == n: break
        start = end - overlap
        if start < 0: start = 0
    return chunks

def main():
    df = pd.read_csv(DATA)
    records = []
    for i, row in df.iterrows():
        txt = str(row["text"]).strip()
        if not txt:
            continue
        for j, ch in enumerate(chunk_text(txt)):
            records.append({"id": f"row{i}_ch{j}", "row": int(i), "text": ch})

    print(f"Prepared {len(records)} chunks. Sending to API...")
    r = requests.post(API, json={"records": records})
    print(r.status_code, r.text)

if __name__ == "__main__":
    main()
