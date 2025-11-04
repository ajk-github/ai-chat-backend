"""
Logging Utilities
Structured logging for query provenance and application events.
"""

import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


class ProvenanceLogger:
    """Logs query provenance for auditing and debugging."""

    def __init__(self, log_path: str = "logs/query_provenance.jsonl"):
        """
        Initialize provenance logger.

        Args:
            log_path: Path to log file (JSONL format)
        """
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_query(
        self,
        chat_id: str,
        user_id: str,
        question: str,
        sql_query: Optional[str],
        success: bool,
        row_count: Optional[int] = None,
        execution_time: Optional[float] = None,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Log a query execution for provenance.

        Args:
            chat_id: Chat session ID
            user_id: User ID
            question: Natural language question
            sql_query: Generated SQL query
            success: Whether query succeeded
            row_count: Number of rows returned
            execution_time: Query execution time in seconds
            error: Error message if query failed
            metadata: Additional metadata
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "chat_id": chat_id,
            "user_id": user_id,
            "question": question,
            "sql_query": sql_query,
            "success": success,
            "row_count": row_count,
            "execution_time_seconds": execution_time,
            "error": error
        }

        if metadata:
            log_entry["metadata"] = metadata

        # Append to JSONL file
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry) + '\n')

    def get_recent_queries(self, limit: int = 100) -> list:
        """
        Get recent query logs.

        Args:
            limit: Maximum number of logs to return

        Returns:
            List of log entries
        """
        if not self.log_path.exists():
            return []

        logs = []

        with open(self.log_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    logs.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue

        # Return most recent first
        return logs[-limit:][::-1]


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """
    Configure application logging.

    Args:
        log_level: Logging level
        log_file: Optional file to log to
    """
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(log_level.upper())

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level.upper())

    # Format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level.upper())
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)

    return logger
