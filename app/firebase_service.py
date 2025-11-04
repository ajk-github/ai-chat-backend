import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path
import os
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class FirebaseService:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FirebaseService, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize Firebase Admin SDK"""
        logger = logging.getLogger(__name__)
        cred = None
        try:
            # Resolve credentials path from .env or use default (repo root)
            project_root = Path(__file__).resolve().parents[1]
            default_cred_path = project_root / "firebase_config.json"

            cred_path = os.getenv("FIREBASE_CREDENTIALS_PATH")
            if cred_path:
                # If a relative path is provided, resolve it relative to project root
                cred_path = str((project_root / cred_path).resolve()) if not os.path.isabs(cred_path) else cred_path
            else:
                cred_path = str(default_cred_path)

            if os.path.exists(cred_path):
                cred = credentials.Certificate(cred_path)
                logger.info(f"Initializing Firebase with service account at {cred_path}")
            else:
                raise FileNotFoundError(
                    f"Firebase credentials not found at {cred_path}. "
                    f"Set FIREBASE_CREDENTIALS_PATH to your service account JSON, or ensure {default_cred_path} exists."
                )

            if not firebase_admin._apps:
                firebase_admin.initialize_app(cred)

            self.db = firestore.client()
            logger.info("Firebase initialized successfully")
        except Exception as e:
            logger.error(f"Firebase initialization failed: {e}")
            raise
    
    def create_chat_session(
        self, 
        chat_id: str, 
        user_id: str, 
        title: str,
        summary: str
    ) -> None:
        """Create a new chat session in Firestore under chatSessions/{chatId}"""
        chat_ref = self.db.collection('chatSessions').document(chat_id)
        
        try:
            chat_ref.set({
            'chatId': chat_id,
            'userId': user_id,
            'title': title,
            'createdAt': firestore.SERVER_TIMESTAMP,
            'updatedAt': firestore.SERVER_TIMESTAMP,
        })

            # Add initial system message
            self.add_message(
                chat_id=chat_id,
                user_id=user_id,
                role='system',
                content=summary
            )
        except Exception as e:
            logging.getLogger(__name__).error(f"Failed to create chat session in Firestore: {e}")
            raise
    
    def add_message(
        self,
        chat_id: str,
        user_id: str,
        role: str,  # 'user', 'assistant', or 'system'
        content: str
    ) -> str:
        """Add a message to a chat's messages subcollection"""
        messages_ref = (
            self.db
            .collection('chatSessions')
            .document(chat_id)
            .collection('messages')
        )
        
        message_ref = messages_ref.document()
        message_id = message_ref.id
        
        try:
            message_ref.set({
            'messageId': message_id,
            'userId': user_id,
            'role': role,
            'content': content,
            'timestamp': firestore.SERVER_TIMESTAMP,
        })

            # Update parent chat's updatedAt
            chat_ref = self.db.collection('chatSessions').document(chat_id)
            chat_ref.update({
                'updatedAt': firestore.SERVER_TIMESTAMP,
            })
        except Exception as e:
            logging.getLogger(__name__).error(f"Failed to add message to Firestore: {e}")
            raise
        
        return message_id
    
    def get_chat(self, chat_id: str) -> Optional[Dict[str, Any]]:
        """Get a chat session"""
        chat_ref = self.db.collection('chatSessions').document(chat_id)
        doc = chat_ref.get()
        
        if doc.exists:
            return doc.to_dict()
        return None
    
    def delete_chat(self, chat_id: str, user_id: str) -> None:
        """Delete a chat and all its messages"""
        # Delete all messages in the subcollection (no userId filter needed - one chat = one user)
        messages_ref = self.db.collection('chatSessions').document(chat_id).collection('messages')
        messages = messages_ref.stream()
        
        for msg in messages:
            msg.reference.delete()
        
        # Delete chat document
        self.db.collection('chatSessions').document(chat_id).delete()
        
        logger = logging.getLogger(__name__)
        logger.info(f"Deleted chat {chat_id} and all messages from Firestore")