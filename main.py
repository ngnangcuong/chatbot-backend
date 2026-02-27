import logging
import uuid
from typing import List
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from contextlib import asynccontextmanager

import models
from database import engine, get_db
from llm_service import process_and_store_document, generate_chat_response

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Create database tables
models.Base.metadata.create_all(bind=engine)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up Chatbot API...")
    yield
    logger.info("Shutting down Chatbot API...")

app = FastAPI(title="AI Chatbot API", lifespan=lifespan)

# Add CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For dev, update in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health")
def health_check():
    return {"status": "healthy"}

@app.post("/api/sessions")
def create_session(db: Session = Depends(get_db)):
    """Create a new chat session."""
    session_id = str(uuid.uuid4())
    db_session = models.Session(id=session_id)
    db.add(db_session)
    db.commit()
    return {"session_id": session_id}

@app.post("/api/upload")
def upload_document(
    session_id: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Uploads a document and processes it into the vector store."""
    # Verify session
    db_session = db.query(models.Session).filter(models.Session.id == session_id).first()
    if not db_session:
        raise HTTPException(status_code=404, detail="Session not found")
        
    try:
        # Save file record in DB
        db_file = models.UploadedFile(
            session_id=session_id,
            filename=file.filename,
            content_type=file.content_type
        )
        db.add(db_file)
        db.commit()
        
        # Process and store in ChromaDB
        process_and_store_document(session_id, file)
        
        return {"status": "success", "filename": file.filename}
    except Exception as e:
        logger.error(f"Error processing upload: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/history/{session_id}")
def get_chat_history(session_id: str, db: Session = Depends(get_db)):
    """Retrieve chat history for a given session."""
    db_session = db.query(models.Session).filter(models.Session.id == session_id).first()
    if not db_session:
        raise HTTPException(status_code=404, detail="Session not found")
        
    messages = db.query(models.Message).filter(
        models.Message.session_id == session_id
    ).order_by(models.Message.created_at.asc()).all()
    
    return [
        {"role": msg.role, "content": msg.content, "created_at": msg.created_at} 
        for msg in messages
    ]

from pydantic import BaseModel
class ChatRequest(BaseModel):
    session_id: str
    message: str

@app.post("/api/chat")
def chat(request: ChatRequest, db: Session = Depends(get_db)):
    """Handles chat messages and AI response."""
    # Verify session
    db_session = db.query(models.Session).filter(models.Session.id == request.session_id).first()
    if not db_session:
        raise HTTPException(status_code=404, detail="Session not found")
        
    # Generate response
    try:
        # Retrieve history for context
        history = db.query(models.Message).filter(
            models.Message.session_id == request.session_id
        ).order_by(models.Message.created_at.asc()).all()
        
        # Generate response using LLM service
        ai_response_content = generate_chat_response(
            session_id=request.session_id,
            query=request.message,
            chat_history=history
        )
        
        # Save user message
        user_msg = models.Message(session_id=request.session_id, role="user", content=request.message)
        db.add(user_msg)
        
        # Save AI message
        ai_msg = models.Message(session_id=request.session_id, role="ai", content=ai_response_content)
        db.add(ai_msg)
        
        db.commit()
        
        return {"response": ai_response_content}
        
    except Exception as e:
        logger.error(f"Error during chat handling: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
