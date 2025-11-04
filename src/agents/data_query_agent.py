"""
Data Query Agent
LangGraph-based agent for converting natural language queries to SQL and executing them.
"""

import logging
import json
from typing import Dict, List, Any, Optional, TypedDict, Annotated
from operator import add

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from data_processing.duckdb_catalog import DuckDBCatalog
from utils.sql_validator import SQLValidator

logger = logging.getLogger(__name__)


# ===== State Definition =====

class AgentState(TypedDict):
    """State for the data query agent."""
    # Input
    question: str
    chat_history: List[Dict[str, str]]

    # Schema context
    available_tables: List[str]
    schema_context: str

    # SQL generation
    sql_query: Optional[str]
    sql_valid: bool
    validation_error: Optional[str]
    validation_warnings: List[str]

    # Execution
    query_results: Optional[Dict[str, Any]]
    execution_success: bool
    execution_error: Optional[str]

    # Response
    answer: str
    metadata: Dict[str, Any]

    # Control
    retry_count: int
    max_retries: int


# ===== Agent Class =====

class DataQueryAgent:
    """LangGraph agent for natural language data queries."""

    def __init__(
        self,
        duckdb_catalog: DuckDBCatalog,
        openai_api_key: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
        max_retries: int = 2,
        schema_profiles_dir: Optional[str] = None
    ):
        """
        Initialize data query agent.

        Args:
            duckdb_catalog: DuckDB catalog instance
            openai_api_key: OpenAI API key
            model: OpenAI model name
            temperature: LLM temperature
            max_retries: Maximum SQL generation retries
            schema_profiles_dir: Directory containing schema profile JSONs
        """
        self.catalog = duckdb_catalog
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.schema_profiles_dir = Path(schema_profiles_dir) if schema_profiles_dir else None

        # Initialize LLM
        self.llm = ChatOpenAI(
            api_key=openai_api_key,
            model=model,
            temperature=temperature
        )

        # Initialize SQL validator
        self.validator = SQLValidator(
            max_rows=1000,
            max_joins=3,
            pii_columns=["ssn", "password", "credit_card"]
        )

        # Load schema profiles
        self.schema_profiles = self._load_schema_profiles()

        # Build graph
        self.graph = self._build_graph()

    def _load_schema_profiles(self) -> Dict[str, Any]:
        """Load schema profiles from JSON files."""
        profiles = {}

        if not self.schema_profiles_dir or not self.schema_profiles_dir.exists():
            logger.warning("Schema profiles directory not found")
            return profiles

        # Load catalog
        catalog_path = self.schema_profiles_dir / "schema_catalog.json"

        if catalog_path.exists():
            with open(catalog_path, 'r') as f:
                catalog_data = json.load(f)
                profiles = catalog_data.get("tables", {})

        logger.info(f"Loaded {len(profiles)} schema profiles")

        return profiles

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("load_schema", self.load_schema_node)
        workflow.add_node("generate_sql", self.generate_sql_node)
        workflow.add_node("validate_sql", self.validate_sql_node)
        workflow.add_node("execute_query", self.execute_query_node)
        workflow.add_node("format_response", self.format_response_node)
        workflow.add_node("handle_error", self.handle_error_node)

        # Define edges
        workflow.set_entry_point("load_schema")

        workflow.add_edge("load_schema", "generate_sql")

        workflow.add_conditional_edges(
            "generate_sql",
            self.check_sql_generated,
            {
                "validate": "validate_sql",
                "error": "handle_error"
            }
        )

        workflow.add_conditional_edges(
            "validate_sql",
            self.check_validation,
            {
                "execute": "execute_query",
                "retry": "generate_sql",
                "error": "handle_error"
            }
        )

        workflow.add_conditional_edges(
            "execute_query",
            self.check_execution,
            {
                "format": "format_response",
                "error": "handle_error"
            }
        )

        workflow.add_edge("format_response", END)
        workflow.add_edge("handle_error", END)

        return workflow.compile()

    # ===== Nodes =====

    def load_schema_node(self, state: AgentState) -> AgentState:
        """Load schema information for relevant tables."""
        logger.info("Loading schema context...")

        # Get available tables
        tables = self.catalog.list_tables()

        # Build schema context
        schema_parts = []

        for table_name in tables:
            # Get basic schema
            schema = self.catalog.get_table_schema(table_name)

            # Get profile if available
            profile = self.schema_profiles.get(table_name, {})

            schema_parts.append(f"Table: {table_name}")

            if profile.get("row_count"):
                schema_parts.append(f"  Rows: {profile['row_count']}")

            schema_parts.append("  Columns:")

            # Use profile columns if available, otherwise use schema
            if profile.get("columns"):
                for col in profile["columns"]:
                    # Handle both string and dict formats
                    if isinstance(col, str):
                        col_desc = f"    - {col}"
                    else:
                        col_name = col.get("name", "")
                        col_type = col.get("dtype", col.get("type", ""))
                        semantic = col.get("semantic_hints", [])

                        col_desc = f"    - {col_name} ({col_type})"

                        if semantic and semantic != ["unknown"]:
                            col_desc += f" - {', '.join(semantic)}"

                    schema_parts.append(col_desc)
            else:
                for col in schema:
                    schema_parts.append(f"    - {col['name']} ({col['type']})")

            schema_parts.append("")

        schema_context = "\n".join(schema_parts)

        state["available_tables"] = tables
        state["schema_context"] = schema_context

        logger.info(f"Loaded schema for {len(tables)} tables")

        return state

    def generate_sql_node(self, state: AgentState) -> AgentState:
        """Generate SQL query from natural language."""
        logger.info("Generating SQL query...")

        # Build system prompt
        system_prompt = f"""You are a SQL expert. Convert the natural language question into a DuckDB SQL query.

Available tables and schemas:
{state['schema_context']}

Rules:
- Generate ONLY SELECT queries (read-only)
- Use proper table and column names exactly as shown
- Always include appropriate WHERE clauses to filter data
- Use aggregations (COUNT, SUM, AVG, etc.) when appropriate
- Return ONLY the SQL query without any explanation, markdown, or formatting
- Do not include markdown code blocks or backticks
- The query should be executable as-is

Previous conversation:
{self._format_chat_history(state.get('chat_history', []))}

If validation failed previously, fix this error: {state.get('validation_error', 'N/A')}
"""

        # Generate SQL
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=state["question"])
            ]

            response = self.llm.invoke(messages)

            sql_query = response.content.strip()

            # Clean up SQL (remove markdown formatting if present)
            sql_query = sql_query.replace("```sql", "").replace("```", "").strip()

            state["sql_query"] = sql_query

            logger.info(f"Generated SQL: {sql_query}")

        except Exception as e:
            logger.error(f"Error generating SQL: {e}")
            state["sql_query"] = None
            state["execution_error"] = f"Failed to generate SQL: {str(e)}"

        return state

    def validate_sql_node(self, state: AgentState) -> AgentState:
        """Validate generated SQL query."""
        logger.info("Validating SQL query...")

        sql_query = state.get("sql_query")

        if not sql_query:
            state["sql_valid"] = False
            state["validation_error"] = "No SQL query generated"
            return state

        # Validate
        validation_result = self.validator.validate(sql_query)

        state["sql_valid"] = validation_result["valid"]

        if validation_result["valid"]:
            state["sql_query"] = validation_result["sanitized_sql"]
            state["validation_warnings"] = validation_result.get("warnings", [])
            logger.info("SQL validation passed")
        else:
            state["validation_error"] = validation_result["error"]
            logger.warning(f"SQL validation failed: {validation_result['error']}")

        return state

    def execute_query_node(self, state: AgentState) -> AgentState:
        """Execute validated SQL query."""
        logger.info("Executing SQL query...")

        sql_query = state.get("sql_query")

        if not sql_query:
            state["execution_success"] = False
            state["execution_error"] = "No SQL query to execute"
            return state

        try:
            # Execute query
            result = self.catalog.execute_query(sql_query, timeout=10, max_rows=1000)

            state["query_results"] = result
            state["execution_success"] = result["success"]

            if not result["success"]:
                state["execution_error"] = result.get("error", "Unknown execution error")
                logger.error(f"Query execution failed: {state['execution_error']}")
            else:
                logger.info(
                    f"Query executed successfully: {result['row_count']} rows "
                    f"in {result['execution_time_seconds']:.3f}s"
                )

        except Exception as e:
            state["execution_success"] = False
            state["execution_error"] = str(e)
            logger.error(f"Error executing query: {e}")

        return state

    def format_response_node(self, state: AgentState) -> AgentState:
        """Format query results into natural language answer."""
        logger.info("Formatting response...")

        results = state.get("query_results", {})

        if not results or not results.get("success"):
            state["answer"] = "I couldn't retrieve the data. Please try rephrasing your question."
            return state

        rows = results.get("rows", [])
        row_count = results.get("row_count", 0)

        # Build natural language response
        try:
            # Create summary prompt
            summary_prompt = f"""Based on the SQL query results below, provide a natural language answer to the user's question.

User's question: {state['question']}

SQL query used: {state.get('sql_query', 'N/A')}

Results ({row_count} rows):
{json.dumps(rows[:10], indent=2)}

Provide a clear, concise answer that:
1. Directly answers the question
2. Highlights key insights from the data
3. Mentions the row count if relevant
4. Is conversational and easy to understand
"""

            messages = [
                SystemMessage(content="You are a helpful data analyst assistant."),
                HumanMessage(content=summary_prompt)
            ]

            response = self.llm.invoke(messages)

            state["answer"] = response.content.strip()

            # Add metadata
            state["metadata"] = {
                "sql_query": state.get("sql_query"),
                "row_count": row_count,
                "execution_time_seconds": results.get("execution_time_seconds"),
                "sample_rows": rows[:5] if rows else []
            }

            logger.info("Response formatted successfully")

        except Exception as e:
            logger.error(f"Error formatting response: {e}")

            # Fallback response
            state["answer"] = (
                f"I found {row_count} result(s) but had trouble formatting the answer. "
                f"Here's a sample: {json.dumps(rows[:3])}"
            )

        return state

    def handle_error_node(self, state: AgentState) -> AgentState:
        """Handle errors gracefully."""
        logger.info("Handling error...")

        error_msg = (
            state.get("execution_error") or
            state.get("validation_error") or
            "An unknown error occurred"
        )

        state["answer"] = (
            f"I encountered an issue processing your question: {error_msg}. "
            "Please try rephrasing your question or asking something else."
        )

        state["metadata"] = {
            "error": error_msg,
            "sql_query": state.get("sql_query")
        }

        return state

    # ===== Conditional edges =====

    def check_sql_generated(self, state: AgentState) -> str:
        """Check if SQL was generated successfully."""
        if state.get("sql_query"):
            return "validate"
        return "error"

    def check_validation(self, state: AgentState) -> str:
        """Check SQL validation result."""
        if state.get("sql_valid"):
            return "execute"

        # Retry if under max retries
        if state.get("retry_count", 0) < state.get("max_retries", self.max_retries):
            state["retry_count"] = state.get("retry_count", 0) + 1
            logger.info(f"Retrying SQL generation (attempt {state['retry_count']})")
            return "retry"

        return "error"

    def check_execution(self, state: AgentState) -> str:
        """Check query execution result."""
        if state.get("execution_success"):
            return "format"
        return "error"

    # ===== Helper methods =====

    def _format_chat_history(self, chat_history: List[Dict[str, str]]) -> str:
        """Format chat history for context."""
        if not chat_history:
            return "No previous conversation"

        # Take last 5 exchanges (10 messages)
        recent = chat_history[-10:]

        formatted = []

        for msg in recent:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "user":
                formatted.append(f"User: {content}")
            elif role == "assistant":
                formatted.append(f"Assistant: {content}")

        return "\n".join(formatted) if formatted else "No previous conversation"

    # ===== Public interface =====

    def ask(
        self,
        question: str,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Ask a natural language question.

        Args:
            question: Natural language question
            chat_history: Previous chat messages

        Returns:
            Dictionary with answer and metadata
        """
        # Initialize state
        initial_state = {
            "question": question,
            "chat_history": chat_history or [],
            "available_tables": [],
            "schema_context": "",
            "sql_query": None,
            "sql_valid": False,
            "validation_error": None,
            "validation_warnings": [],
            "query_results": None,
            "execution_success": False,
            "execution_error": None,
            "answer": "",
            "metadata": {},
            "retry_count": 0,
            "max_retries": self.max_retries
        }

        # Run graph
        final_state = self.graph.invoke(initial_state)

        return {
            "answer": final_state.get("answer", "I couldn't process your question."),
            "metadata": final_state.get("metadata", {}),
            "sql_query": final_state.get("sql_query"),
            "success": final_state.get("execution_success", False)
        }


# Example CLI entry point removed; this module is used by the FastAPI app.
