"""
Status Logger for Agent Thought Process
Logs detailed reasoning steps and provides streaming capabilities for UI updates.
"""
import logging
import time
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class StepStatus(str, Enum):
    """Status of an agent step"""
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class StatusLogger:
    """
    Logger for agent thought process and status updates.

    Features:
    - Logs detailed reasoning steps
    - Stores steps for API response
    - Supports real-time callbacks for streaming to UI
    - Thread-safe for concurrent requests
    """

    def __init__(self, callback: Optional[Callable[[Dict[str, Any]], None]] = None):
        """
        Initialize status logger.

        Args:
            callback: Optional callback function that receives status updates
                     Format: callback({"step": str, "status": str, "details": str, "timestamp": str})
        """
        self.steps: List[Dict[str, Any]] = []
        self.callback = callback
        self.start_time = time.time()

    def log_step(
        self,
        step_name: str,
        status: StepStatus,
        details: str = "",
        reasoning: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Log a single step in the agent's process.

        Args:
            step_name: Name of the step (e.g., "Loading Schema", "Generating SQL")
            status: Status of the step
            details: Human-readable details about what's happening
            reasoning: List of reasoning points (agent's thought process)
            metadata: Additional metadata (SQL query, table names, etc.)
        """
        timestamp = datetime.now().isoformat()
        elapsed = time.time() - self.start_time

        step_data = {
            "step": step_name,
            "status": status.value,
            "details": details,
            "timestamp": timestamp,
            "elapsed_seconds": round(elapsed, 2),
        }

        if reasoning:
            step_data["reasoning"] = reasoning

        if metadata:
            step_data["metadata"] = metadata

        # Store step
        self.steps.append(step_data)

        # Log to console with detailed formatting (always use INFO or above to ensure visibility)
        log_level = logging.WARNING if status == StepStatus.FAILED else logging.INFO

        # Print to console directly for better visibility
        print(f"\n{'='*70}")
        print(f"[{step_name}] - {status.value.upper()}")
        print(f"Time: {elapsed:.2f}s | {timestamp}")
        print(f"{'-'*70}")
        print(f"{details}")
        if reasoning:
            print(self._format_reasoning(reasoning).rstrip())
        if metadata:
            print(self._format_metadata(metadata).rstrip())
        print(f"{'='*70}\n")

        # Also log via logger
        logger.log(
            log_level,
            f"[{step_name}] {status.value.upper()} - {details}"
        )

        # Call callback for real-time streaming
        if self.callback:
            try:
                self.callback(step_data)
            except Exception as e:
                logger.error(f"Error in status callback: {e}")

    def log_thought(self, thought: str, context: Optional[Dict[str, Any]] = None):
        """
        Log an intermediate thought/reasoning without creating a full step.
        Useful for detailed agent thinking process.

        Args:
            thought: The agent's current thought
            context: Optional context dictionary
        """
        timestamp = datetime.now().isoformat()
        elapsed = time.time() - self.start_time

        thought_data = {
            "type": "thought",
            "content": thought,
            "timestamp": timestamp,
            "elapsed_seconds": round(elapsed, 2),
        }

        if context:
            thought_data["context"] = context

        self.steps.append(thought_data)

        # Print to console directly
        print(f"\n💭 THOUGHT [{elapsed:.2f}s]: {thought}{f' | Context: {context}' if context else ''}")

        # Also log via logger
        logger.info(f"THOUGHT: {thought}")

        # Call callback
        if self.callback:
            try:
                self.callback(thought_data)
            except Exception as e:
                logger.error(f"Error in thought callback: {e}")

    def log_decision(
        self,
        decision: str,
        options: List[str],
        chosen: str,
        reasoning: str
    ):
        """
        Log a decision point in the agent's workflow.

        Args:
            decision: What decision is being made
            options: Available options
            chosen: Which option was chosen
            reasoning: Why this option was chosen
        """
        timestamp = datetime.now().isoformat()
        elapsed = time.time() - self.start_time

        decision_data = {
            "type": "decision",
            "decision": decision,
            "options": options,
            "chosen": chosen,
            "reasoning": reasoning,
            "timestamp": timestamp,
            "elapsed_seconds": round(elapsed, 2),
        }

        self.steps.append(decision_data)

        # Print to console directly
        print(f"\n⚡ DECISION [{elapsed:.2f}s]: {decision}")
        print(f"   Options: {', '.join(options)}")
        print(f"   Chosen: {chosen}")
        print(f"   Reasoning: {reasoning}\n")

        # Also log via logger
        logger.info(f"DECISION: {decision} -> {chosen}")

        if self.callback:
            try:
                self.callback(decision_data)
            except Exception as e:
                logger.error(f"Error in decision callback: {e}")

    def get_steps(self) -> List[Dict[str, Any]]:
        """Get all logged steps."""
        return self.steps

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the entire process."""
        total_time = time.time() - self.start_time

        # Count steps by status
        completed = sum(1 for s in self.steps if s.get("status") == StepStatus.COMPLETED.value)
        failed = sum(1 for s in self.steps if s.get("status") == StepStatus.FAILED.value)
        running = sum(1 for s in self.steps if s.get("status") == StepStatus.RUNNING.value)

        return {
            "total_steps": len(self.steps),
            "completed": completed,
            "failed": failed,
            "running": running,
            "total_time_seconds": round(total_time, 2),
            "steps": self.steps
        }

    def _format_reasoning(self, reasoning: Optional[List[str]]) -> str:
        """Format reasoning points for console output."""
        if not reasoning:
            return ""

        formatted = "\nReasoning:\n"
        for i, point in enumerate(reasoning, 1):
            formatted += f"  {i}. {point}\n"

        return formatted

    def _format_metadata(self, metadata: Optional[Dict[str, Any]]) -> str:
        """Format metadata for console output."""
        if not metadata:
            return ""

        formatted = "\nMetadata:\n"
        for key, value in metadata.items():
            # Truncate long values
            str_value = str(value)
            if len(str_value) > 200:
                str_value = str_value[:200] + "..."
            formatted += f"  • {key}: {str_value}\n"

        return formatted


# Global registry for active status loggers (keyed by request ID or session ID)
_active_loggers: Dict[str, StatusLogger] = {}


def get_logger(request_id: str) -> Optional[StatusLogger]:
    """Get status logger for a specific request."""
    return _active_loggers.get(request_id)


def register_logger(request_id: str, logger: StatusLogger):
    """Register a status logger for a request."""
    _active_loggers[request_id] = logger


def unregister_logger(request_id: str):
    """Unregister a status logger after request completes."""
    _active_loggers.pop(request_id, None)
