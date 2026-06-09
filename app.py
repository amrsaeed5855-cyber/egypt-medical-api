from fastapi import FastAPI
from rag_logic import rag

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat")
def chat(req: dict):
    response = rag(req["query"], req.get("history", []))
    return {"response": response}
