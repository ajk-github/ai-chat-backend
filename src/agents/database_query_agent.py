"""
Database Query Agent (MySQL)
LangGraph-based agent for converting natural language queries to MySQL SQL and executing them.
"""
#src/agents/database_query_agent.py
import logging
import json
from typing import Dict, List, Any, Optional, TypedDict
from operator import add

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from data_processing.mysql_catalog import MySQLCatalog
from utils.sql_validator import SQLValidator

logger = logging.getLogger(__name__)


# ===== State Definition =====

class AgentState(TypedDict):
    """State for the database query agent."""
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

class DatabaseQueryAgent:
    """LangGraph agent for natural language database queries (MySQL)."""

    def __init__(
        self,
        mysql_catalog: MySQLCatalog,
        openai_api_key: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
        max_retries: int = 2,
    ):
        """
        Initialize database query agent.

        Args:
            mysql_catalog: MySQL catalog instance
            openai_api_key: OpenAI API key
            model: OpenAI model name
            temperature: LLM temperature
            max_retries: Maximum SQL generation retries
        """
        self.catalog = mysql_catalog
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries

        # Initialize LLM
        self.llm = ChatOpenAI(
            api_key=openai_api_key,
            model=model,
            temperature=temperature
        )

        # Initialize SQL validator (SELECT only)
        self.validator = SQLValidator(
            max_rows=1000,
            max_joins=5,  # MySQL can handle more joins
            pii_columns=["ssn", "password", "credit_card"]
        )

        # Build graph
        self.graph = self._build_graph()

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
                "retry": "generate_sql",
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

        # Ensure schema cache is loaded (async, but we'll handle it in execute_query)
        # For now, get tables from cache
        tables = self.catalog.list_tables()

        # Build schema context
        schema_parts = []

        for table_name in tables:
            # Get schema from cache
            schema = self.catalog.get_table_schema(table_name)

            schema_parts.append(f"Table: {table_name}")
            schema_parts.append("  Columns:")

            for col in schema:
                col_name = col.get("name", "")
                col_type = col.get("full_type", col.get("type", ""))
                nullable = col.get("nullable", True)
                key = col.get("key", "")

                col_desc = f"    - {col_name} ({col_type})"
                
                if key == "PRI":
                    col_desc += " - PRIMARY KEY"
                elif key == "MUL":
                    col_desc += " - INDEXED"
                elif key == "UNI":
                    col_desc += " - UNIQUE"
                
                if not nullable:
                    col_desc += " - NOT NULL"

                schema_parts.append(col_desc)

            schema_parts.append("")

        schema_context = "\n".join(schema_parts)

        state["available_tables"] = tables
        state["schema_context"] = schema_context

        logger.info(f"Loaded schema for {len(tables)} tables")

        return state

    def generate_sql_node(self, state: AgentState) -> AgentState:
        """Generate SQL query from natural language."""
        logger.info("Generating SQL query...")
        
        # Get error message for prompt BEFORE clearing it (for retry detection)
        error_msg = state.get('validation_error') or state.get('execution_error') or 'N/A'
        
        # Detect if this is a retry (has validation_error or execution_error from previous attempt)
        is_retry = bool(state.get("validation_error") or state.get("execution_error"))
        
        if is_retry:
            # Increment retry count
            current_retry = state.get("retry_count", 0)
            state["retry_count"] = current_retry + 1
            
            # Check if max retries exceeded
            max_retries = state.get("max_retries", self.max_retries)
            if state["retry_count"] > max_retries:
                logger.error(f"Max retries ({max_retries}) exceeded. Stopping.")
                state["execution_error"] = f"Max retries ({max_retries}) exceeded"
                return state
            
            # Clear execution-related state for fresh retry
            state["execution_error"] = None
            state["execution_success"] = False
            state["query_results"] = None
            state["sql_valid"] = False
            
            logger.info(f"Retrying SQL generation (attempt {state['retry_count']}/{max_retries})")
        else:
            # First attempt - initialize retry count
            state["retry_count"] = 0
            state["validation_error"] = None
            state["execution_error"] = None

        # Build error guidance for MySQL-specific issues
        error_guidance = ""
        if is_retry:
            error_guidance = """

═══════════════════════════════════════════════════════════════════
CRITICAL SQL ERROR - READ THIS CAREFULLY BEFORE GENERATING SQL
═══════════════════════════════════════════════════════════════════

The previous query failed. Common MySQL issues to avoid:

1. TABLE/COLUMN NOT FOUND:
   - Use exact table and column names from the schema (case-sensitive in some MySQL configs)
   - Use backticks for table/column names if they contain special characters: `table_name`
   - Check spelling carefully

2. GROUP BY ERRORS:
   - All non-aggregated columns in SELECT must appear in GROUP BY
   - MySQL allows aliases in GROUP BY (unlike DuckDB), but be careful
   - Example: SELECT EXTRACT(YEAR FROM date_col) AS year, COUNT(*) FROM table GROUP BY year

3. WHERE CLAUSE:
   - Can use column aliases in HAVING, but not in WHERE (use full expression)
   - Example: WHERE EXTRACT(YEAR FROM date_col) = 2025 (correct)
   - Example: WHERE year = 2025 (WRONG if 'year' is an alias)

4. DATE FUNCTIONS:
   - Use DATE_FORMAT(date_col, '%Y-%m-%d') for formatting
   - Use YEAR(date_col), MONTH(date_col), DAY(date_col) for extraction
   - Use DATE(date_col) to extract date part from datetime

5. JOIN SYNTAX:
   - Use explicit JOIN syntax: FROM table1 JOIN table2 ON table1.id = table2.id
   - Use LEFT JOIN, RIGHT JOIN, INNER JOIN as needed
   - Always specify join conditions

6. AGGREGATION:
   - Use GROUP BY when using aggregate functions (COUNT, SUM, AVG, etc.)
   - All non-aggregated columns must be in GROUP BY

═══════════════════════════════════════════════════════════════════
"""

        # Build system prompt
        system_prompt = f"""You are a MySQL SQL expert. Convert the natural language question into a MySQL SQL query.

Available tables and schemas:
{state['schema_context']}

Rules:
- Generate ONLY SELECT queries (read-only)
- Use proper table and column names exactly as shown in the schema
- Use backticks for table/column names if needed: `table_name`, `column_name`
- Always include appropriate WHERE clauses to filter data
- Use aggregations (COUNT, SUM, AVG, MAX, MIN, etc.) when appropriate
- MySQL-specific syntax:
  * Use DATE_FORMAT(date_col, '%Y-%m-%d') for date formatting
  * Use YEAR(date_col), MONTH(date_col), DAY(date_col) for date extraction
  * Use DATE(date_col) to extract date from datetime
  * Use CONCAT() for string concatenation
  * Use IFNULL() or COALESCE() for null handling
- WHERE clause rules:
  * You CANNOT use column aliases from SELECT in WHERE clause
  * Use the full expression: WHERE YEAR(date_col) = 2025, NOT WHERE year = 2025
  * Aliases can only be used in ORDER BY, HAVING, or subqueries
- GROUP BY rules:
  * MySQL allows aliases in GROUP BY, but all non-aggregated columns must be included
  * When grouping by date parts, include the full expression or alias in GROUP BY
  * Example: SELECT YEAR(date_col) AS year, COUNT(*) FROM table WHERE YEAR(date_col) = 2025 GROUP BY year
- JOIN syntax:
  * Use explicit JOIN syntax with ON clause
  * Specify join conditions clearly
  * Use appropriate join types (INNER, LEFT, RIGHT)
- Return ONLY the SQL query without any explanation, markdown, or formatting
- Do not include markdown code blocks or backticks around the SQL
- The query should be executable as-is

Previous conversation:
{self._format_chat_history(state.get('chat_history', []))}

If validation or execution failed previously, fix this error: {error_msg}{error_guidance}
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
            # Execute query (async, but we're in a sync node)
            # Create a new connection in the thread's event loop instead of using the pool
            # This avoids "attached to a different loop" errors
            import asyncio
            import concurrent.futures
            from datetime import datetime
            
            def run_async_query():
                """Run async query in a new event loop with a new connection."""
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                start_time = datetime.now()
                try:
                    async def execute():
                        # Create a new connection in this loop (not using the pool from FastAPI's loop)
                        try:
                            import aiomysql
                            conn = await aiomysql.connect(
                                host=self.catalog.host,
                                port=self.catalog.port,
                                user=self.catalog.user,
                                password=self.catalog.password,
                                db=self.catalog.database,
                                charset='utf8mb4',
                                cursorclass=aiomysql.DictCursor,
                            )
                            try:
                                async with conn.cursor() as cursor:
                                    # Add LIMIT if not present
                                    query_upper = sql_query.strip().upper()
                                    if "LIMIT" not in query_upper:
                                        final_query = sql_query.rstrip(';') + f" LIMIT 1000"
                                    else:
                                        final_query = sql_query
                                    
                                    await cursor.execute(final_query)
                                    rows = await cursor.fetchall()
                                    
                                    # Convert to list of dicts
                                    result_rows = []
                                    for row in rows:
                                        if isinstance(row, dict):
                                            row_dict = {}
                                            for key, value in row.items():
                                                if hasattr(value, 'isoformat'):
                                                    row_dict[key] = value.isoformat()
                                                else:
                                                    row_dict[key] = value
                                            result_rows.append(row_dict)
                                        else:
                                            result_rows.append(row)
                                    
                                    column_names = list(result_rows[0].keys()) if result_rows else []
                                    execution_time = (datetime.now() - start_time).total_seconds()
                                    
                                    return {
                                        "success": True,
                                        "rows": result_rows,
                                        "row_count": len(result_rows),
                                        "column_names": column_names,
                                        "execution_time_seconds": execution_time,
                                        "query": final_query
                                    }
                            finally:
                                conn.close()
                        except Exception as e:
                            execution_time = (datetime.now() - start_time).total_seconds()
                            logger.error(f"Query execution failed: {e}")
                            return {
                                "success": False,
                                "error": str(e),
                                "execution_time_seconds": execution_time,
                                "query": sql_query
                            }
                    
                    return new_loop.run_until_complete(execute())
                finally:
                    new_loop.close()
            
            # Execute in thread pool to avoid blocking and event loop conflicts
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(run_async_query)
                result = future.result(timeout=35)  # Slightly longer than query timeout

            state["query_results"] = result
            state["execution_success"] = result["success"]

            if not result["success"]:
                state["execution_error"] = result.get("error", "Unknown execution error")
                logger.error(f"Query execution failed: {state['execution_error']}")
            else:
                logger.info(
                    f"Query executed successfully: {result['row_count']} rows "
                    f"in {result.get('execution_time_seconds', 0):.3f}s"
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

        # Serialize rows for JSON compatibility
        serialized_rows = self._serialize_for_json(rows[:10])

        # Build natural language response
        try:
            # Create summary prompt
            summary_prompt = f"""Based on the SQL query results below, provide a natural language answer to the user's question.

User's question: {state['question']}

SQL query used: {state.get('sql_query', 'N/A')}

Results ({row_count} rows):
{json.dumps(serialized_rows, indent=2)}

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
            serialized_sample = self._serialize_for_json(rows[:3])
            state["answer"] = (
                f"I found {row_count} result(s) but had trouble formatting the answer. "
                f"Here's a sample: {json.dumps(serialized_sample)}"
            )

        return state

    def handle_error_node(self, state: AgentState) -> AgentState:
        """Handle errors gracefully."""
        logger.info("Handling error...")

        # Log technical error for debugging (but don't show to user)
        error_msg = (
            state.get("execution_error") or
            state.get("validation_error") or
            "An unknown error occurred"
        )
        
        # Log the technical error for debugging
        logger.warning(f"Query failed with error: {error_msg}")
        if state.get("sql_query"):
            logger.warning(f"Failed SQL query: {state.get('sql_query')}")

        # Return user-friendly message instead of technical error
        state["answer"] = (
            "I don't understand your question. Please try to be more specific."
        )

        state["metadata"] = {
            "error": error_msg,  # Keep technical error in metadata for debugging
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

        # Check if we can retry
        current_retry_count = state.get("retry_count", 0)
        max_retries = state.get("max_retries", self.max_retries)
        
        if current_retry_count < max_retries:
            logger.info(f"Validation error detected - will retry (current: {current_retry_count}/{max_retries})")
            return "retry"
        else:
            logger.error(f"Max retries ({max_retries}) reached for validation error.")
        
        return "error"

    def check_execution(self, state: AgentState) -> str:
        """Check query execution result."""
        if state.get("execution_success"):
            return "format"
        
        # Check if this is a fixable SQL error that should trigger retry
        execution_error = state.get("execution_error", "")
        if execution_error and any(keyword in execution_error.upper() for keyword in [
            "GROUP BY", "MUST APPEAR", "AGGREGATE", "UNKNOWN COLUMN", "TABLE", "SYNTAX"
        ]):
            current_retry_count = state.get("retry_count", 0)
            max_retries = state.get("max_retries", self.max_retries)
            
            # Move execution error to validation error so retry can see it
            state["validation_error"] = execution_error
            
            if current_retry_count < max_retries:
                logger.info(f"Execution error detected - will retry (current: {current_retry_count}/{max_retries})")
                return "retry"
            else:
                logger.error(f"Max retries ({max_retries}) reached for execution error.")
        
        return "error"

    # ===== Helper methods =====

    def _serialize_for_json(self, obj: Any) -> Any:
        """Convert non-serializable types to JSON-compatible formats."""
        if isinstance(obj, dict):
            return {k: self._serialize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._serialize_for_json(item) for item in obj]
        elif hasattr(obj, 'isoformat'):  # datetime, date objects
            return obj.isoformat()
        else:
            return obj

    def _format_chat_history(self, chat_history: List[Dict[str, str]]) -> str:
        """Format chat history for prompt."""
        if not chat_history:
            return "No previous conversation."

        formatted = []
        for msg in chat_history[-5:]:  # Last 5 messages
            role = msg.get("role", "user")
            content = msg.get("content", "")
            formatted.append(f"{role.capitalize()}: {content}")

        return "\n".join(formatted)

    # ===== Public API =====

    async def ask(
        self,
        question: str,
        chat_history: List[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Process a natural language question and return an answer.

        Args:
            question: Natural language question
            chat_history: Previous conversation messages

        Returns:
            Dictionary with answer and metadata
        """
        if chat_history is None:
            chat_history = []

        # Ensure schema is loaded (async)
        await self.catalog._load_schema_cache()

        # Initialize state
        initial_state: AgentState = {
            "question": question,
            "chat_history": chat_history,
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
            "max_retries": self.max_retries,
        }

        try:
            # Run graph (synchronous, but execute_query_node handles async internally)
            result_state = self.graph.invoke(initial_state)
            
            return {
                "answer": result_state.get("answer", ""),
                "metadata": result_state.get("metadata", {}),
                "sql_query": result_state.get("sql_query"),
                "success": result_state.get("execution_success", False),
            }

        except Exception as e:
            # Handle recursion limit or other errors
            if "recursion" in str(e).lower() or "limit" in str(e).lower():
                logger.error(f"Graph execution limit reached: {e}")
                return {
                    "answer": "I encountered an issue processing your question due to repeated query errors. Please try rephrasing your question.",
                    "metadata": {"error": str(e)},
                    "sql_query": None,
                    "success": False,
                }
            
            logger.error(f"Error in ask(): {e}", exc_info=True)
            return {
                "answer": "I encountered an issue processing your question. Please try again.",
                "metadata": {"error": str(e)},
                "sql_query": None,
                "success": False,
            }

