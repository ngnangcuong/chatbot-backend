import os
import shutil
import tempfile
import logging
import pandas as pd
from PyPDF2 import PdfReader
from fastapi import UploadFile

from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

logger = logging.getLogger(__name__)

# Constants
VECTOR_DB_PATH = "./chroma_db"
OLLAMA_MODEL = "phi3"
EMBEDDING_MODEL = "nomic-embed-text"
COLLECTION_NAME = "chatbot_embeddings"

def extract_text_from_file(file_path: str, filename: str) -> str:
    """Extract text from PDF, CSV or TXT file."""
    ext = filename.split('.')[-1].lower()
    text = ""
    try:
        if ext == "pdf":
            reader = PdfReader(file_path)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        elif ext == "csv":
            df = pd.read_csv(file_path)
            # Simple summarization: just convert to string
            text = df.to_string(index=False)
        elif ext in ["txt", "md"]:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    except Exception as e:
        logger.error(f"Error extracting text from {filename}: {e}")
        raise
    return text

def process_and_store_document_background(session_id: str, filename: str, file_content: bytes):
    """Processes an uploaded document, chunks it, and stores in Chroma in the background."""
    
    # Save file temporarily to disk since we might need seek operations
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, filename)
    
    try:
        with open(temp_path, "wb") as buffer:
            buffer.write(file_content)
            
        text = extract_text_from_file(temp_path, filename)
        if not text.strip():
            logger.warning("No text could be extracted from the document.")
            return
            
        # Split text into manageable chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        chunks = text_splitter.split_text(text)
        
        # Prepare metadata to associate chunks with a specific session
        metadatas = [{"session_id": session_id, "filename": filename} for _ in chunks]
        
        # Setup vector store
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=VECTOR_DB_PATH
        )
        
        # Add chunks to vector store
        vector_store.add_texts(texts=chunks, metadatas=metadatas)
        logger.info(f"Successfully processed {len(chunks)} chunks from {filename}")
        
    except Exception as e:
        logger.error(f"Background processing error for {filename}: {e}")
        raise
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir)

def process_and_store_document(session_id: str, file: UploadFile):
    """(Deprecated) synchronous process."""
    pass

def get_session_retriever(session_id: str):
    """Gets a retriever scoped to a specific session ID."""
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=VECTOR_DB_PATH
    )
    
    # We filter by session_id metadata so cross-session context doesn't bleed over
    return vector_store.as_retriever(
        search_kwargs={"k": 4, "filter": {"session_id": session_id}}
    )

def generate_chat_response(session_id: str, query: str, chat_history: list):
    """Generates an AI response incorporating history and document context."""
    return ""

async def generate_chat_response_stream(session_id: str, query: str, chat_history: list):
    """Generates an AI response stream incorporating history and document context."""
    llm = Ollama(model=OLLAMA_MODEL)
    retriever = get_session_retriever(session_id)
    
    # Format chat history for LangChain
    formatted_history = []
    for msg in chat_history:
        if msg.role == 'user':
            formatted_history.append(HumanMessage(content=msg.content))
        else:
            formatted_history.append(AIMessage(content=msg.content))
            
    # System prompt for retrieving documents based on conversation history
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given a chat history and the latest user question "
         "which might reference context in the chat history, formulate a standalone question "
         "which can be understood without the chat history. Do NOT answer the question, "
         "just reformulate it if needed and otherwise return it as is."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    
    # System prompt for answering the question
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant. Use the following context to answer the user's question. "
         "If you don't know the answer or the context doesn't contain it, just say that you don't know. "
         "Use formatting like bolding, code blocks, or lists where appropriate to make your answer readable.\n\n"
         "Context: {context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    logger.info("Starting RAG chain stream...")
    try:
        async for chunk in rag_chain.astream({
            "input": query,
            "chat_history": formatted_history
        }):
            if "answer" in chunk:
                yield chunk["answer"]
    except Exception as e:
        logger.error(f"Error during rag_chain.astream: {e}")
        yield f"\n[Error: {e}]"
