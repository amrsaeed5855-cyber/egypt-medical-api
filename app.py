from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from rag_logic import rag

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
def index():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat")
def chat(req: dict):
    response = rag(req["query"], req.get("history", []))
    return {"response": response}
