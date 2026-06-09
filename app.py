from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional

from rag_logic import pharmacy_consult, ChatResponse, DelegationPayload, delegation_to_dict

app = FastAPI(
    title="Egyptian Pharmacy Subagent",
    description="ElevenLabs workflow subagent for medication guidance",
)


@app.get("/", response_class=HTMLResponse)
def index():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.get("/health")
def health():
    return {"status": "ok", "agent": "pharmacy_subagent"}


class ChatRequest(BaseModel):
    message: str
    history: list = []
    delegation: Optional[DelegationPayload] = None


@app.post("/chat", response_model=ChatResponse)
@app.post("/pharmacy/consult", response_model=ChatResponse)
def chat(req: ChatRequest):
    message = req.message or ""
    if not message.strip():
        raise HTTPException(status_code=400, detail="message must not be empty")
    result = pharmacy_consult(message, req.history, delegation_to_dict(req.delegation))
    return ChatResponse(**result.to_dict())
