import asyncio

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional

from rag_logic import (
    pharmacy_consult,
    ChatResponse,
    DelegationPayload,
    delegation_to_dict,
    _rag_lock,
    CHAT_TIMEOUT_SEC,
)

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
    from rag_logic import df, index, ENABLE_SEMANTIC_SEARCH, _embed_load_failed, retrieval_engine
    return {
        "status": "ok",
        "agent": "pharmacy_subagent",
        "drug_count": len(df) if not df.empty else 0,
        "index_loaded": index is not None,
        "retrieval_ready": retrieval_engine is not None,
        "semantic_search": ENABLE_SEMANTIC_SEARCH and not _embed_load_failed,
    }


class ChatRequest(BaseModel):
    message: str
    history: list = []
    delegation: Optional[DelegationPayload] = None


@app.post("/chat", response_model=ChatResponse)
@app.post("/pharmacy/consult", response_model=ChatResponse)
async def chat(req: ChatRequest):
    message = req.message or ""
    if not message.strip():
        raise HTTPException(status_code=400, detail="message must not be empty")

    delegation = delegation_to_dict(req.delegation)

    def _run():
        with _rag_lock:
            return pharmacy_consult(message, req.history, delegation)

    try:
        result = await asyncio.wait_for(asyncio.to_thread(_run), timeout=CHAT_TIMEOUT_SEC)
    except asyncio.TimeoutError:
        return ChatResponse(
            response="معلش، الطلب أخد وقت طويل — استنى شوية وحاول تاني.",
            task_status="needs_info",
        )
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return ChatResponse(
            response="مش قادر أكمّل الطلب دلوقتي — جرّب تاني أو اسأل عن دواء بالاسم التجاري.",
            task_status="needs_info",
        )

    return ChatResponse(**result.to_dict())
