"""
Data Query Agent
LangGraph-based agent for converting natural language queries to SQL and executing them.
"""
#src/agents/data_query_agent.py
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

import pandas as pd

from data_processing.duckdb_catalog import DuckDBCatalog
from utils.sql_validator import SQLValidator
from agents.telos_weekly_report_tool import TelosWeeklyReportTool

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

    # Weekly report
    is_weekly_report: bool
    company_name: Optional[str]


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

        # Initialize SQL validator (no row limit - analyze entire dataset)
        self.validator = SQLValidator(
            max_rows=999999999,
            max_joins=3,
            pii_columns=["ssn", "password", "credit_card"]
        )

        # Initialize Telos weekly report tool
        self.weekly_report_tool = TelosWeeklyReportTool(duckdb_catalog)

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
        workflow.add_node("detect_intent", self.detect_intent_node)
        workflow.add_node("handle_weekly_report", self.handle_weekly_report_node)
        workflow.add_node("load_schema", self.load_schema_node)
        workflow.add_node("generate_sql", self.generate_sql_node)
        workflow.add_node("validate_sql", self.validate_sql_node)
        workflow.add_node("execute_query", self.execute_query_node)
        workflow.add_node("format_response", self.format_response_node)
        workflow.add_node("handle_error", self.handle_error_node)

        # Define edges
        workflow.set_entry_point("detect_intent")

        # Route based on intent detection
        workflow.add_conditional_edges(
            "detect_intent",
            self.route_by_intent,
            {
                "weekly_report": "handle_weekly_report",
                "normal_query": "load_schema"
            }
        )

        workflow.add_edge("handle_weekly_report", END)
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

    def detect_intent_node(self, state: AgentState) -> AgentState:
        """Detect if the question is asking for a weekly report."""
        logger.info("Detecting intent...")

        question = state["question"].lower()

        # Check for weekly report patterns
        weekly_patterns = [
            "/weeklyreport",
            "weekly report",
            "weekly kpi",
            "kpi report",
            "weekly metrics",
            "weekly summary",
            "performance report"
        ]

        is_weekly_report = any(pattern in question for pattern in weekly_patterns)
        state["is_weekly_report"] = is_weekly_report

        # Extract company name if present
        company_name = "Company"  # Default

        # Check for --company_name pattern
        if "--" in question:
            parts = question.split("--")
            if len(parts) > 1:
                company_name = parts[1].strip().title()

        # Check for common company name patterns
        company_keywords = ["for", "company", "client"]
        for keyword in company_keywords:
            if keyword in question:
                words = question.split()
                try:
                    idx = words.index(keyword)
                    if idx + 1 < len(words):
                        potential_company = words[idx + 1].strip(",.!?").title()
                        if potential_company and not potential_company in ["the", "a", "an"]:
                            company_name = potential_company
                except:
                    pass

        state["company_name"] = company_name

        logger.info(f"Intent detection: is_weekly_report={is_weekly_report}, company={company_name}")

        return state

    def handle_weekly_report_node(self, state: AgentState) -> AgentState:
        """Handle weekly report generation."""
        logger.info(f"Generating weekly report for {state.get('company_name', 'Company')}...")

        try:
            # Generate report
            report = self.weekly_report_tool.generate_report(
                company_name=state.get("company_name", "Company")
            )

            state["answer"] = report
            state["execution_success"] = True
            state["metadata"] = {
                "report_type": "weekly_kpi",
                "company": state.get("company_name", "Company")
            }

        except Exception as e:
            logger.error(f"Error generating weekly report: {e}")
            state["answer"] = f"❌ Error generating weekly report: {str(e)}"
            state["execution_success"] = False

        return state

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
        
        # Get error message for prompt BEFORE clearing it (for retry detection)
        error_msg = state.get('validation_error') or state.get('execution_error') or 'N/A'
        
        # Detect if this is a retry (has validation_error or execution_error from previous attempt)
        is_retry = bool(state.get("validation_error") or state.get("execution_error"))
        
        if is_retry:
            # We're retrying - increment retry count and prepare state
            current_retry_count = state.get("retry_count", 0)
            max_retries = state.get("max_retries", self.max_retries)
            
            # Check if we've exceeded max retries
            if current_retry_count >= max_retries:
                logger.error(f"Max retries ({max_retries}) already reached. Stopping retry.")
                # Don't generate SQL, just set error state
                state["sql_query"] = None
                state["execution_error"] = "Maximum retry attempts exceeded"
                return state
            
            # Increment retry count
            state["retry_count"] = current_retry_count + 1
            logger.warning(f"Retrying SQL generation (attempt {state['retry_count']}/{max_retries})")
            
            # Ensure validation_error is set from execution_error if needed (preserve error_msg)
            if state.get("execution_error") and not state.get("validation_error"):
                state["validation_error"] = state.get("execution_error")
            
            # Clear execution state for fresh attempt (but keep validation_error for prompt)
            state["execution_error"] = None
            state["execution_success"] = False
            state["query_results"] = None
            # Reset validation state since we're generating new SQL
            state["sql_valid"] = False
        
        # Build enhanced error guidance for GROUP BY and WHERE clause issues
        error_guidance = ""
        if error_msg and any(keyword in error_msg.upper() for keyword in ["GROUP BY", "MUST APPEAR", "AGGREGATE", "WHERE", "BINDER ERROR"]):
            error_guidance = """

═══════════════════════════════════════════════════════════════════
CRITICAL SQL ERROR - READ THIS CAREFULLY BEFORE GENERATING SQL
═══════════════════════════════════════════════════════════════════

The previous query failed with a GROUP BY or WHERE clause error.

═══════════════════════════════════════════════════════════════════
ERROR 1: WHERE CLAUSE CANNOT USE COLUMN ALIASES
═══════════════════════════════════════════════════════════════════

CRITICAL RULE: You CANNOT use column aliases from SELECT in WHERE clause!
- Aliases are only available in ORDER BY, HAVING, or subqueries
- In WHERE clause, you MUST use the full expression

WRONG (using alias in WHERE and GROUP BY):
SELECT 
    EXTRACT(YEAR FROM visit_date) AS year,
    EXTRACT(MONTH FROM visit_date) AS month,
    SUM(total_payment) AS total_payments
FROM ar_analysis
WHERE year = 2025  ← WRONG! Can't use alias 'year' in WHERE
GROUP BY year, month  ← WRONG! Can't use aliases 'year', 'month' in GROUP BY (DuckDB requirement)

CORRECT (using full expression in WHERE and GROUP BY):
SELECT 
    EXTRACT(YEAR FROM visit_date) AS year,
    EXTRACT(MONTH FROM visit_date) AS month,
    SUM(total_payment) AS total_payments
FROM ar_analysis
WHERE EXTRACT(YEAR FROM visit_date) = 2025  ← CORRECT! Use full expression in WHERE
GROUP BY EXTRACT(YEAR FROM visit_date), EXTRACT(MONTH FROM visit_date)  ← CORRECT! Use full expressions in GROUP BY too!
ORDER BY month

═══════════════════════════════════════════════════════════════════
ERROR 2: GROUP BY WITH DATE EXTRACTION (CRITICAL FOR DUCKDB)
═══════════════════════════════════════════════════════════════════

CRITICAL RULE: DuckDB does NOT allow using column aliases in GROUP BY clause!
- You MUST use the FULL EXPRESSION in GROUP BY, not the alias
- WRONG: GROUP BY year, month (using aliases)
- CORRECT: GROUP BY EXTRACT(YEAR FROM visit_date), EXTRACT(MONTH FROM visit_date) (using full expressions)

RULE: When you use EXTRACT() on a date column, you CANNOT select the original date column in SELECT unless you:
  1. Include it in GROUP BY (which defeats the purpose of grouping by month/year), OR
  2. Wrap it with ANY_VALUE(date_col) or MIN(date_col) or MAX(date_col)

CORRECT EXAMPLE (break down payments by month for year 2025):
SELECT 
    EXTRACT(YEAR FROM visit_date) AS year,
    EXTRACT(MONTH FROM visit_date) AS month,
    SUM(total_payment) AS total_payments
FROM ar_analysis
WHERE EXTRACT(YEAR FROM visit_date) = 2025  ← Use full expression, not alias!
GROUP BY EXTRACT(YEAR FROM visit_date), EXTRACT(MONTH FROM visit_date)  ← DUCKDB: Use full expressions in GROUP BY!
ORDER BY month

WRONG EXAMPLE 1 (using aliases in GROUP BY - DuckDB doesn't allow this):
SELECT 
    EXTRACT(YEAR FROM visit_date) AS year,
    EXTRACT(MONTH FROM visit_date) AS month,
    SUM(total_payment) AS total_payments
FROM ar_analysis
WHERE EXTRACT(YEAR FROM visit_date) = 2025
GROUP BY year, month  ← WRONG! DuckDB requires full expressions in GROUP BY
ORDER BY month

WRONG EXAMPLE 2 (selecting original date column):
SELECT 
    EXTRACT(YEAR FROM visit_date) AS year,
    EXTRACT(MONTH FROM visit_date) AS month,
    visit_date,  ← THIS IS THE PROBLEM! Don't select visit_date here
    SUM(total_payment) AS total_payments
FROM ar_analysis
WHERE EXTRACT(YEAR FROM visit_date) = 2025
GROUP BY EXTRACT(YEAR FROM visit_date), EXTRACT(MONTH FROM visit_date)

If you need the original date value, use:
SELECT 
    EXTRACT(YEAR FROM visit_date) AS year,
    EXTRACT(MONTH FROM visit_date) AS month,
    ANY_VALUE(visit_date) AS sample_date,  ← Use ANY_VALUE() wrapper
    SUM(total_payment) AS total_payments
FROM ar_analysis
WHERE EXTRACT(YEAR FROM visit_date) = 2025
GROUP BY EXTRACT(YEAR FROM visit_date), EXTRACT(MONTH FROM visit_date)  ← Use full expressions!

KEY POINT: For "break down by month", you only need year, month, and aggregated values. You do NOT need the original date column.

═══════════════════════════════════════════════════════════════════

"""

        # Build system prompt
        system_prompt = f"""You are a SQL expert. Convert the natural language question into a DuckDB SQL query.

Available tables and schemas:
{state['schema_context']}

Rules:
- Generate ONLY SELECT queries (read-only)
- Use proper table and column names exactly as shown
- Always include appropriate WHERE clauses to filter data
- Use aggregations (COUNT, SUM, AVG, etc.) when appropriate
- DO NOT add LIMIT clauses - the system will handle row limits automatically
- Analyze the ENTIRE dataset unless the user specifically asks for a sample or limit
- WHERE clause rules (CRITICAL):
  * You CANNOT use column aliases from SELECT in WHERE clause
  * Use the full expression: WHERE EXTRACT(YEAR FROM date_col) = 2025, NOT WHERE year = 2025
  * Aliases can only be used in ORDER BY, HAVING, or subqueries
  * Example: SELECT EXTRACT(YEAR FROM visit_date) AS year ... WHERE EXTRACT(YEAR FROM visit_date) = 2025 (correct)
  * Example: SELECT EXTRACT(YEAR FROM visit_date) AS year ... WHERE year = 2025 (WRONG!)
- GROUP BY rules (CRITICAL for DuckDB):
  * DuckDB does NOT allow using column aliases in GROUP BY clause - you MUST use the full expression
  * When using GROUP BY, all non-aggregated columns in SELECT must appear in GROUP BY using their FULL EXPRESSION, not aliases
  * When grouping by date parts (EXTRACT(YEAR FROM date_col), EXTRACT(MONTH FROM date_col)), use: GROUP BY EXTRACT(YEAR FROM date_col), EXTRACT(MONTH FROM date_col) - NOT GROUP BY year, month
  * Only select the extracted values, not the original date column (unless wrapped in ANY_VALUE())
  * Example: SELECT EXTRACT(YEAR FROM visit_date) AS year, EXTRACT(MONTH FROM visit_date) AS month, COUNT(*) FROM table WHERE EXTRACT(YEAR FROM visit_date) = 2025 GROUP BY EXTRACT(YEAR FROM visit_date), EXTRACT(MONTH FROM visit_date) ORDER BY month
  * If you need the original date value, use ANY_VALUE(date_col) or MIN(date_col)
- Return ONLY the SQL query without any explanation, markdown, or formatting
- Do not include markdown code blocks or backticks
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
            # Execute query (no row limit - analyze entire dataset)
            result = self.catalog.execute_query(sql_query, timeout=60, max_rows=999999999)

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

        # Serialize rows for JSON compatibility (safety net for any remaining Timestamps)
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

            # Fallback response - serialize sample data for safety
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

    def route_by_intent(self, state: AgentState) -> str:
        """Route based on detected intent."""
        if state.get("is_weekly_report", False):
            return "weekly_report"
        return "normal_query"

    def check_sql_generated(self, state: AgentState) -> str:
        """Check if SQL was generated successfully."""
        if state.get("sql_query"):
            return "validate"
        return "error"

    def check_validation(self, state: AgentState) -> str:
        """Check SQL validation result."""
        if state.get("sql_valid"):
            return "execute"

        # Check if we can retry (retry_count will be incremented in generate_sql_node)
        current_retry_count = state.get("retry_count", 0)
        max_retries = state.get("max_retries", self.max_retries)
        
        if current_retry_count < max_retries:
            logger.info(f"Validation error detected - will retry (current: {current_retry_count}/{max_retries})")
            return "retry"
        else:
            logger.error(f"Max retries ({max_retries}) reached for validation error. Stopping retry loop.")

        return "error"

    def check_execution(self, state: AgentState) -> str:
        """Check query execution result."""
        if state.get("execution_success"):
            return "format"
        
        # Check if this is a fixable SQL error (like GROUP BY issues) that should trigger retry
        execution_error = state.get("execution_error", "")
        if execution_error and any(keyword in execution_error.upper() for keyword in ["GROUP BY", "MUST APPEAR", "AGGREGATE"]):
            # Get current retry count and max retries
            current_retry_count = state.get("retry_count", 0)
            max_retries = state.get("max_retries", self.max_retries)
            
            # Build enhanced error message with the failed SQL query
            failed_sql = state.get("sql_query", "")
            enhanced_error = execution_error
            if failed_sql:
                enhanced_error = f"{execution_error}\n\nFailed SQL query:\n{failed_sql}\n\nThis query is WRONG. Fix it by removing the original date column from SELECT or using ANY_VALUE()."
            
            # Move execution error to validation error so retry can see it
            # This will be detected by generate_sql_node as a retry condition
            state["validation_error"] = enhanced_error
            
            # Check if we can still retry (retry_count will be incremented in generate_sql_node)
            if current_retry_count < max_retries:
                logger.info(f"Execution error detected - will retry (current: {current_retry_count}/{max_retries})")
                return "retry"
            else:
                logger.error(f"Max retries ({max_retries}) reached for execution error. Stopping retry loop.")
        
        return "error"

    # ===== Helper methods =====

    def _serialize_for_json(self, obj: Any) -> Any:
        """
        Convert pandas Timestamps and other non-serializable types to JSON-compatible formats.
        
        This is a safety net in case any Timestamps slip through from DuckDB results.
        """
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif obj is None:
            return None
        elif isinstance(obj, dict):
            return {k: self._serialize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._serialize_for_json(item) for item in obj]
        else:
            # Safely check for NaN/NaT (only for scalars)
            try:
                if pd.isna(obj):
                    return None
            except (ValueError, TypeError):
                # If pd.isna() fails (e.g., on arrays), keep the value as-is
                pass
        return obj

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
            "max_retries": self.max_retries,
            "is_weekly_report": False,
            "company_name": None
        }

        # Run graph with recursion limit to prevent infinite loops
        try:
            final_state = self.graph.invoke(initial_state)
        except Exception as e:
            if "recursion" in str(e).lower() or "limit" in str(e).lower():
                logger.error(f"Graph recursion limit reached. This usually indicates an infinite retry loop.")
                logger.error(f"Final state - retry_count: {initial_state.get('retry_count', 0)}, "
                           f"validation_error: {initial_state.get('validation_error')}, "
                           f"execution_error: {initial_state.get('execution_error')}")
                # Return user-friendly error response
                return {
                    "answer": "I don't understand your question. Please try to be more specific.",
                    "metadata": {
                        "error": "Recursion limit reached - query generation failed after multiple retries",
                        "sql_query": initial_state.get("sql_query"),
                        "retry_count": initial_state.get("retry_count", 0)
                    },
                    "sql_query": initial_state.get("sql_query"),
                    "success": False
                }
            raise

        # Return user-friendly message if query failed
        # (handle_error_node should have already set this, but ensure it's always user-friendly)
        answer = final_state.get("answer", "")
        if not final_state.get("execution_success", False):
            # Query failed - always show user-friendly message
            answer = "I don't understand your question. Please try to be more specific."
        elif not answer or not answer.strip():
            # No answer generated but query succeeded (shouldn't happen, but safety check)
            answer = "I don't understand your question. Please try to be more specific."
        
        return {
            "answer": answer,
            "metadata": final_state.get("metadata", {}),
            "sql_query": final_state.get("sql_query"),
            "success": final_state.get("execution_success", False)
        }


# Example CLI entry point removed; this module is used by the FastAPI app.
