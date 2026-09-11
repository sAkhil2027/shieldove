import uuid
import asyncio
import logging
from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import asyncio
from sanitizer import ConfigurableSanitizer

# Load environment variables from .env in project root
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

# Initialize logger (reuse uvicorn logger if available)
logger = logging.getLogger("uvicorn.error")
sanitizer = ConfigurableSanitizer()

# Import RAG components
from rag_retriever import RAGRetriever
from llm_service import generate_answer
from retrieval_engine import search_vulnerabilities, match_owasp_guidance
from prompt_builder import build_llm_prompt

app = FastAPI(title="OWASP Security Assistant API")

# Allow CORS for any origin (adjust as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiter middleware (protects against abuse)
from fastapi import Request
from rate_limiter import rate_limiter

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    await rate_limiter(request)
    response = await call_next(request)
    return response

class ChatRequest(BaseModel):
    session_id: str
    question: str
    top_k: int = 2
    # Optional: limit of candidate chunks for reranking (defaults to 10 in retriever)
    candidate_k: int = 10

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    """Handle a chat request, retrieve relevant chunks, and stream an LLM answer.
    The response is a streaming text payload compatible with the existing Streamlit client.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    request_id = str(uuid.uuid4())
    logger.info(f"[REQ {request_id}] Received question: {request.question}")

    # Retrieve relevant context chunks using the RAG pipeline
    retriever = RAGRetriever()
    try:
        chunks = retriever.retrieve(request.question, top_k=request.top_k, candidate_k=request.candidate_k)
    except Exception as e:
        logger.error(f"[REQ {request_id}] Retrieval error: {e}")
        raise HTTPException(status_code=500, detail="Error during retrieval")

    # Stream the answer from the LLM service
    async def answer_generator():
        async for token in generate_answer(request.question, chunks, request_id):
            yield token

    return StreamingResponse(answer_generator(), media_type="text/event-stream")

# ---------------------------------------------------------------------------
# Admin endpoint to reload sanitization rules at runtime (protected by token)
# ---------------------------------------------------------------------------
@app.post("/admin/reload_rules")
async def admin_reload_rules(x_admin_token: str = Header(..., alias="X-Admin-Token")):
    """Reload sanitization rules without restarting the service.
    The secret token is read from the environment variable ADMIN_RELOAD_TOKEN.
    This endpoint is only enabled when ENABLE_ADMIN_ENDPOINT=true.
    """
    if os.getenv("ENABLE_ADMIN_ENDPOINT", "false").lower() != "true":
        raise HTTPException(status_code=403, detail="Admin endpoint disabled in this environment")
    expected_token = os.getenv("ADMIN_RELOAD_TOKEN", "change_me")
    if x_admin_token != expected_token:
        raise HTTPException(status_code=403, detail="Invalid admin token")
    sanitizer.reload_rules()
    return {"status": "rules reloaded"}

@app.post("/explain_vulns")
async def explain_vulns(request: ChatRequest):
    """Return up to 2 vulnerability explanations with OWASP guidance, streamed via LLM.
    The response is a streaming text payload.
    """
    # Enforce max 2 vulnerabilities
    top_k = min(request.top_k, 2)
    # Retrieve vulnerabilities
    vulns = search_vulnerabilities(request.question, top_k=top_k)
    if not vulns:
        raise HTTPException(status_code=404, detail="No matching vulnerabilities")
    # Enrich each with guidance
    enriched = []
    for v in vulns:
        guidance = match_owasp_guidance(v)
        enriched.append({**v, "owasp_guidance": guidance})
    # Build prompt for LLM using helper
    full_prompt = build_llm_prompt(enriched, request.question)
    request_id = str(uuid.uuid4())
    logger.info(f"[REQ {request_id}] Streaming explain_vulns for query: {request.question}")
    async def answer_generator():
        async for token in generate_answer(full_prompt, [], request_id):
            yield token
    return StreamingResponse(answer_generator(), media_type="text/event-stream")
