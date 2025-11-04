from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any
from uuid import uuid4
import os
import tempfile
import logging
from dotenv import load_dotenv
import asyncio

load_dotenv()

# Reuse existing modules from src/
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_processing.chat_data_manager import ChatDataManager
from agents.data_query_agent import DataQueryAgent
from .firebase_service import FirebaseService


class ChatRequest(BaseModel):
    chat_id: str
    message: str
    user_id: str


def get_user_base_dir(user_id: str) -> str:
    # Writes to data/<user_id>/
    return str(Path("data") / user_id)


def get_processed_dir(user_id: str, chat_id: str) -> str:
    return str(Path(get_user_base_dir(user_id)) / chat_id / "processed")


app = FastAPI(title="AI AR Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...),
    user_id: str = Form(...),
) -> Dict[str, Any]:
    # Create per-user base dir and chat id
    chat_id = str(uuid4())
    base_data_dir = get_user_base_dir(user_id)

    manager = ChatDataManager(base_data_dir=base_data_dir)

    # Persist incoming file temporarily
    try:
        suffix = os.path.splitext(file.filename or "")[1] or ""
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            content = await file.read()
            tmp.write(content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read upload: {e}")

    try:
        # Ensure chat workspace exists
        manager.create_chat_workspace(chat_id)

        # Process file into per-chat processed/ and register in DuckDB
        result = await asyncio.to_thread(
            manager.process_uploaded_file,
            chat_id,
            tmp_path,
            file.filename or "uploaded_file",
        )

        # Build metadata from catalog tables
        tables = manager.list_chat_tables(chat_id)
        # Pick first table for the simple metadata fields expected by FE
        first = tables[0] if tables else {"columns": [], "row_count": 0, "column_count": 0}

        summary = f"Processed {len(tables)} table(s), total rows ~ {result.get('total_rows', 0)}"

        # Write chat metadata to Firestore under users/{userId}/chats/{chatId}
        try:
            await asyncio.to_thread(
                FirebaseService().create_chat_session,
                chat_id,
                user_id,
                (file.filename or "Uploaded file"),
                summary,
            )
        except Exception as fb_err:
            # Log clearly so we can diagnose
            logging.getLogger(__name__).error(f"Firebase write failed for chat {chat_id} (user {user_id}): {fb_err}")

        return {
            "chatId": chat_id,
            "summary": summary,
            "metadata": {
                "row_count": first.get("row_count", 0),
                "column_count": first.get("column_count", 0),
                "columns": first.get("columns", []),
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


@app.post("/api/chat")
async def chat(req: ChatRequest) -> Dict[str, Any]:
    base_data_dir = get_user_base_dir(req.user_id)
    manager = ChatDataManager(base_data_dir=base_data_dir)

    # Prepare catalog and agent
    try:
        # First, persist the user's message to Firestore before processing
        try:
            fb = FirebaseService()
            await asyncio.to_thread(
                fb.add_message,
                req.chat_id,
                req.user_id,
                'user',
                req.message,
            )
        except Exception as fb_err:
            logging.getLogger(__name__).error(f"Firebase user message write failed for chat {req.chat_id}: {fb_err}")

        catalog = manager.get_chat_catalog(req.chat_id)
        processed_dir = get_processed_dir(req.user_id, req.chat_id)
        openai_api_key = os.getenv("OPENAI_API_KEY", "")
        if not openai_api_key:
            raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")

        agent = DataQueryAgent(
            duckdb_catalog=catalog,
            openai_api_key=openai_api_key,
            schema_profiles_dir=processed_dir,
        )

        result = await asyncio.to_thread(agent.ask, req.message, [])

        # Persist assistant response to Firestore
        try:
            fb = FirebaseService()
            await asyncio.to_thread(
                fb.add_message,
                req.chat_id,
                req.user_id,
                'assistant',
                result.get("answer", ""),
            )
        except Exception as fb_err:
            logging.getLogger(__name__).error(f"Firebase message write failed for chat {req.chat_id}: {fb_err}")

        return {
            "response": result.get("answer", ""),
            "messageId": str(uuid4()),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/chat/delete/{chat_id}")
async def delete_chat(
    chat_id: str,
    user_id: str
) -> Dict[str, Any]:
    """
    Delete a chat session and all its data.
    
    Deletes:
    - Local data folder: data/{user_id}/{chat_id}/
    - Firebase document: chatSessions/{chat_id} and all messages subcollection
    """
    base_data_dir = get_user_base_dir(user_id)
    manager = ChatDataManager(base_data_dir=base_data_dir)
    
    errors = []
    
    # Delete local data folder
    try:
        success = await asyncio.to_thread(manager.delete_chat_data, chat_id)
        if not success:
            errors.append(f"Chat data folder not found or could not be deleted")
        else:
            logging.getLogger(__name__).info(f"Deleted local data for chat {chat_id}")
    except Exception as e:
        errors.append(f"Error deleting chat data: {str(e)}")
        logging.getLogger(__name__).error(f"Failed to delete chat data for {chat_id}: {e}")
    
    # Delete Firebase data
    try:
        await asyncio.to_thread(
            FirebaseService().delete_chat,
            chat_id,
            user_id
        )
        logging.getLogger(__name__).info(f"Deleted Firebase data for chat {chat_id}")
    except Exception as fb_err:
        errors.append(f"Error deleting Firebase data: {str(fb_err)}")
        logging.getLogger(__name__).error(f"Failed to delete Firebase chat {chat_id}: {fb_err}")
    
    if errors:
        raise HTTPException(
            status_code=500,
            detail=f"Some errors occurred during deletion: {'; '.join(errors)}"
        )
    
    return {
        "success": True,
        "message": f"Chat {chat_id} deleted successfully",
        "chat_id": chat_id
    }


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


