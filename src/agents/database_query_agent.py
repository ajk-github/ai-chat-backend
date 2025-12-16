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

    # Clarification (NEW for better UX)
    needs_clarification: bool
    clarification_question: Optional[str]
    ambiguity_type: Optional[str]

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

    # Status logging
    status_logger: Optional[Any]  # StatusLogger instance


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

        # Load few-shot examples for better SQL generation
        self.query_examples = self._load_query_examples()

        # Load table relationships for multi-table queries
        self.relationships = self._load_relationships()

        # Load business rules and domain knowledge
        self.business_rules = self._load_business_rules()

        # Build graph
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("load_schema", self.load_schema_node)
        workflow.add_node("check_clarification", self.check_clarification_node)
        workflow.add_node("generate_sql", self.generate_sql_node)
        workflow.add_node("validate_sql", self.validate_sql_node)
        workflow.add_node("execute_query", self.execute_query_node)
        workflow.add_node("format_response", self.format_response_node)
        workflow.add_node("handle_error", self.handle_error_node)

        # Define edges
        workflow.set_entry_point("load_schema")

        # After loading schema, check if clarification is needed
        workflow.add_edge("load_schema", "check_clarification")

        # Route based on clarification check
        workflow.add_conditional_edges(
            "check_clarification",
            self.route_after_clarification,
            {
                "clarify": END,  # Return clarification question to user
                "proceed": "generate_sql"  # Continue with SQL generation
            }
        )

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

    # ===== Enhancement Loading Methods =====

    def _load_query_examples(self) -> List[Dict[str, Any]]:
        """Load few-shot query examples from JSON file."""
        examples = []

        # Look for MySQL examples file in schemas directory
        examples_path = Path(__file__).parent.parent.parent / "schemas" / "query_examples_mysql.json"

        # Fallback to DuckDB examples if MySQL-specific file doesn't exist
        if not examples_path.exists():
            examples_path = Path(__file__).parent.parent.parent / "schemas" / "query_examples_duckdb.json"
            logger.warning("MySQL examples not found, using DuckDB examples as fallback")

        if not examples_path.exists():
            logger.warning(f"Query examples file not found at {examples_path}")
            return examples

        try:
            with open(examples_path, 'r') as f:
                data = json.load(f)
                examples = data.get("examples", [])
                logger.info(f"Loaded {len(examples)} query examples for MySQL agent")
        except Exception as e:
            logger.error(f"Error loading query examples: {e}")

        return examples

    def _get_relevant_examples(self, question: str, n: int = 3) -> List[Dict[str, Any]]:
        """
        Find relevant query examples based on keyword matching.

        Args:
            question: User's natural language question
            n: Number of examples to return

        Returns:
            List of relevant examples
        """
        if not self.query_examples:
            return []

        question_lower = question.lower()

        # Score each example
        scored_examples = []
        for ex in self.query_examples:
            score = 0

            # Check if question words appear in example question or tags
            for word in question_lower.split():
                # Skip common words
                if word in ['the', 'a', 'an', 'is', 'are', 'was', 'were', 'what', 'how', 'show', 'me']:
                    continue

                if word in ex["question"].lower():
                    score += 2
                if word in " ".join(ex.get("tags", [])):
                    score += 1

            # Boost CRITICAL examples if they're relevant
            if "CRITICAL" in ex.get("tags", []) and score > 0:
                score += 3

            if score > 0:
                scored_examples.append((score, ex))

        # Return top N examples
        scored_examples.sort(reverse=True, key=lambda x: x[0])
        return [ex for score, ex in scored_examples[:n]]

    def _load_relationships(self) -> Dict[str, Any]:
        """Load table relationships from JSON file."""
        relationships = {}

        # Look for relationships file in schemas directory
        rel_path = Path(__file__).parent.parent.parent / "schemas" / "relationships.json"

        if not rel_path.exists():
            logger.warning(f"Relationships file not found at {rel_path}")
            return relationships

        try:
            with open(rel_path, 'r') as f:
                data = json.load(f)
                relationships = data
                logger.info(f"Loaded {len(data.get('tables', {}))} table relationships for MySQL agent")
        except Exception as e:
            logger.error(f"Error loading relationships: {e}")

        return relationships

    def _get_relationship_context(self, tables: List[str]) -> str:
        """
        Get relationship context for tables mentioned in the question.

        Args:
            tables: List of table names potentially involved in the query

        Returns:
            Formatted relationship context string
        """
        if not self.relationships or not tables:
            return ""

        context_parts = []
        table_data = self.relationships.get("tables", {})
        join_paths = self.relationships.get("common_join_paths", [])

        # Add relationship info for each table
        for table in tables:
            if table in table_data:
                t_info = table_data[table]

                # Show foreign keys (how this table joins to others)
                if t_info.get("foreign_keys"):
                    fks = []
                    for col, fk_info in t_info["foreign_keys"].items():
                        fks.append(f"{col} → {fk_info['references']}")
                    if fks:
                        context_parts.append(f"{table} joins to: {', '.join(fks)}")

        # Add relevant common join paths
        relevant_joins = [jp for jp in join_paths if any(t in jp.get("tables", []) for t in tables)]
        if relevant_joins:
            context_parts.append("\nCommon Join Patterns:")
            for jp in relevant_joins[:3]:  # Show top 3 relevant joins
                context_parts.append(f"  - {jp['description']}")
                context_parts.append(f"    SQL: {jp['sql_template']}")

        return "\n".join(context_parts) if context_parts else ""

    def _load_business_rules(self) -> Dict[str, Any]:
        """Load business rules and domain knowledge from JSON file."""
        rules = {}

        # Look for business rules file in schemas directory
        rules_path = Path(__file__).parent.parent.parent / "schemas" / "business_rules.json"

        if not rules_path.exists():
            logger.warning(f"Business rules file not found at {rules_path}")
            return rules

        try:
            with open(rules_path, 'r') as f:
                data = json.load(f)
                rules = data
                logger.info(f"Loaded business rules with {len(data.get('kpi_definitions', {}))} KPI definitions for MySQL agent")
        except Exception as e:
            logger.error(f"Error loading business rules: {e}")

        return rules

    def _get_business_context(self, question: str) -> str:
        """
        Get relevant business rules context based on the question.

        Args:
            question: User's natural language question

        Returns:
            Formatted business rules context string
        """
        if not self.business_rules:
            return ""

        context_parts = []
        question_lower = question.lower()

        # Check if question involves KPIs
        kpi_defs = self.business_rules.get("kpi_definitions", {})
        for kpi_name, kpi_info in kpi_defs.items():
            if any(word in question_lower for word in kpi_name.lower().split()):
                context_parts.append(f"KPI: {kpi_name}")
                context_parts.append(f"  Formula: {kpi_info['formula']}")
                # Adapt SQL template for MySQL if needed
                sql_template = kpi_info['sql_template']
                # MySQL uses same syntax for most aggregates
                context_parts.append(f"  SQL: {sql_template}")

        # Check if question involves date filtering
        if any(word in question_lower for word in ['this month', 'last month', 'this year', 'current', 'ytd', 'last week']):
            common_filters = self.business_rules.get("common_filters", {})
            context_parts.append("\nCommon Date Filters:")
            for filter_name, filter_info in list(common_filters.items())[:3]:
                if isinstance(filter_info, dict):
                    # Convert DuckDB syntax to MySQL if needed
                    sql = filter_info.get('sql', '')
                    # Basic conversion: DATE_TRUNC → DATE_FORMAT, INTERVAL syntax similar
                    context_parts.append(f"  {filter_name}: {sql}")

        return "\n".join(context_parts) if context_parts else ""

    def _detect_ambiguity(self, question: str, schema_context: str) -> Dict[str, Any]:
        """
        Detect if the question is ambiguous and needs clarification.

        Returns:
            Dict with 'needs_clarification', 'question', and 'type' keys
        """
        question_lower = question.lower()

        # Pattern 1: Missing date range for aggregate queries
        has_aggregate = any(word in question_lower for word in [
            'total', 'sum', 'count', 'average', 'avg', 'how many', 'revenue', 'charges'
        ])
        has_date_filter = any(word in question_lower for word in [
            '2024', '2025', 'this month', 'last month', 'this year', 'last year',
            'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august',
            'september', 'october', 'november', 'december', 'week', 'quarter', 'ytd'
        ])

        if has_aggregate and not has_date_filter:
            return {
                "needs_clarification": True,
                "question": "For what time period would you like this analysis?\n\nOptions:\n• This month\n• Last month\n• This year (2025)\n• Last year (2024)\n• All time\n• Custom date range",
                "type": "missing_date_range"
            }

        # Pattern 2: Multiple date columns ambiguity
        if any(word in question_lower for word in ['date', 'when', 'by month', 'by week', 'by year']):
            # Check if schema has both visit_date and transaction_date
            if 'visit_date' in schema_context.lower() and 'transaction_date' in schema_context.lower():
                # Only ask if the question doesn't specify which date
                if not any(word in question_lower for word in ['visit date', 'service date', 'transaction date', 'payment date', 'posting date']):
                    return {
                        "needs_clarification": True,
                        "question": "Which date would you like to use?\n\nOptions:\n• **Visit Date** (when service was performed) - for operational analysis\n• **Transaction Date** (when payment posted) - for financial/cash flow analysis",
                        "type": "date_column_ambiguity"
                    }

        # Pattern 3: Vague "show me" or "get me" without specifics
        vague_requests = ['show me', 'get me', 'give me', 'list', 'display']
        if any(vague in question_lower for vague in vague_requests):
            # Check if it's too vague (just "show me X" without any filters or conditions)
            words = question_lower.split()
            if len(words) <= 4:  # Very short query
                return {
                    "needs_clarification": True,
                    "question": "I'd be happy to help! Could you provide more details?\n\nFor example:\n• What specific information do you need?\n• Any filters (date range, status, provider, etc.)?\n• How would you like the data grouped or sorted?",
                    "type": "vague_request"
                }

        # Pattern 4: Ambiguous "revenue" - could be gross charges vs collections
        if 'revenue' in question_lower and 'payment' not in question_lower and 'collection' not in question_lower:
            if 'charge' not in question_lower:
                return {
                    "needs_clarification": True,
                    "question": "What type of revenue are you looking for?\n\nOptions:\n• **Gross Charges** (total billed amount)\n• **Collections** (actual payments received)\n• **Net Revenue** (collections minus adjustments)",
                    "type": "revenue_ambiguity"
                }

        # No ambiguity detected
        return {
            "needs_clarification": False,
            "question": None,
            "type": None
        }

    # ===== Nodes =====

    def load_schema_node(self, state: AgentState) -> AgentState:
        """Load schema information for relevant tables."""
        from utils.status_logger import StepStatus

        logger.info("Loading schema context...")

        # Log status
        if state.get("status_logger"):
            state["status_logger"].log_step(
                step_name="Load Schema",
                status=StepStatus.RUNNING,
                details=f"Loading database schema and table information",
                reasoning=[
                    "Connecting to MySQL database",
                    "Retrieving list of available tables",
                    "Extracting column information for SQL generation"
                ]
            )

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

        # Complete status
        if state.get("status_logger"):
            table_list = ", ".join(tables[:5])
            if len(tables) > 5:
                table_list += f", ... ({len(tables)-5} more)"

            state["status_logger"].log_step(
                step_name="Load Schema",
                status=StepStatus.COMPLETED,
                details=f"Loaded schema for {len(tables)} tables",
                reasoning=[
                    f"Found {len(tables)} tables in database",
                    f"Tables: {table_list}",
                    "Schema context ready for SQL generation"
                ],
                metadata={"table_count": len(tables), "tables": tables}
            )

        return state

    def check_clarification_node(self, state: AgentState) -> AgentState:
        """
        Check if the question needs clarification before generating SQL.

        This provides a ChatGPT-level UX by asking follow-up questions when the query is ambiguous.
        """
        logger.info("Checking if clarification is needed...")

        # Detect ambiguity using the question and schema context
        ambiguity = self._detect_ambiguity(
            state["question"],
            state.get("schema_context", "")
        )

        # Update state with clarification info
        state["needs_clarification"] = ambiguity["needs_clarification"]
        state["clarification_question"] = ambiguity["question"]
        state["ambiguity_type"] = ambiguity["type"]

        if ambiguity["needs_clarification"]:
            # Set the clarification question as the answer
            state["answer"] = ambiguity["question"]
            logger.info(f"Clarification needed: {ambiguity['type']}")
        else:
            logger.info("No clarification needed, proceeding with SQL generation")

        return state

    def generate_sql_node(self, state: AgentState) -> AgentState:
        """Generate SQL query from natural language."""
        from utils.status_logger import StepStatus

        logger.info("Generating SQL query...")

        # Log status
        if state.get("status_logger"):
            state["status_logger"].log_step(
                step_name="Generate SQL",
                status=StepStatus.RUNNING,
                details="Converting natural language question to SQL query",
                reasoning=[
                    f"Question: {state['question']}",
                    "Using schema context and business rules",
                    "Applying query examples for better SQL generation"
                ]
            )
        
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

        # Get relevant examples for this question
        relevant_examples = self._get_relevant_examples(state["question"], n=3)

        # Get relationship context if multiple tables might be involved
        available_tables = state.get("available_tables", [])
        relationship_context = self._get_relationship_context(available_tables)

        # Get business rules context
        business_context = self._get_business_context(state["question"])

        # Format relationship context
        relationship_text = ""
        if relationship_context:
            relationship_text = "\n\n" + "="*60 + "\n"
            relationship_text += "TABLE RELATIONSHIPS (Use these for multi-table queries):\n"
            relationship_text += "="*60 + "\n"
            relationship_text += relationship_context + "\n"
            relationship_text += "="*60 + "\n"

        # Format business rules context
        business_text = ""
        if business_context:
            business_text = "\n\n" + "="*60 + "\n"
            business_text += "BUSINESS RULES & KPIs:\n"
            business_text += "="*60 + "\n"
            business_text += business_context + "\n"
            business_text += "="*60 + "\n"

        # Format examples for prompt
        examples_text = ""
        if relevant_examples:
            examples_text = "\n\n" + "="*60 + "\n"
            examples_text += "EXAMPLE QUERIES (Learn from these patterns):\n"
            examples_text += "="*60 + "\n\n"

            for i, ex in enumerate(relevant_examples, 1):
                examples_text += f"Example {i}: {ex['question']}\n"
                examples_text += f"SQL: {ex['sql']}\n"
                if ex.get("explanation"):
                    examples_text += f"Note: {ex['explanation']}\n"
                examples_text += "\n"

            examples_text += "="*60 + "\n"

        # Build system prompt
        system_prompt = f"""You are a MySQL SQL expert. Convert the natural language question into a MySQL SQL query.

Available tables and schemas:
{state['schema_context']}
{relationship_text}
{business_text}
{examples_text}
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

            # Log completion
            if state.get("status_logger"):
                state["status_logger"].log_step(
                    step_name="Generate SQL",
                    status=StepStatus.COMPLETED,
                    details=f"SQL query generated successfully",
                    reasoning=[
                        f"Generated SQL: {sql_query[:200]}{'...' if len(sql_query) > 200 else ''}",
                        "Ready for validation and execution"
                    ],
                    metadata={"sql_query": sql_query}
                )

        except Exception as e:
            logger.error(f"Error generating SQL: {e}")
            state["sql_query"] = None
            state["execution_error"] = f"Failed to generate SQL: {str(e)}"

            # Log failure
            if state.get("status_logger"):
                state["status_logger"].log_step(
                    step_name="Generate SQL",
                    status=StepStatus.FAILED,
                    details=f"Failed to generate SQL query",
                    reasoning=[f"Error: {str(e)}"]
                )

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
        from utils.status_logger import StepStatus

        logger.info("Executing SQL query...")

        sql_query = state.get("sql_query")

        if not sql_query:
            state["execution_success"] = False
            state["execution_error"] = "No SQL query to execute"
            return state

        # Log execution start
        if state.get("status_logger"):
            state["status_logger"].log_step(
                step_name="Execute Query",
                status=StepStatus.RUNNING,
                details="Executing SQL query against MySQL database",
                reasoning=[
                    f"Running query: {sql_query[:150]}{'...' if len(sql_query) > 150 else ''}",
                    "Fetching results from database"
                ]
            )

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
                                    # Remove automatic LIMIT - let queries run as-is
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

                # Log failure
                if state.get("status_logger"):
                    state["status_logger"].log_step(
                        step_name="Execute Query",
                        status=StepStatus.FAILED,
                        details=f"Query execution failed",
                        reasoning=[f"Error: {state['execution_error']}"],
                        metadata={"error": state["execution_error"]}
                    )
            else:
                logger.info(
                    f"Query executed successfully: {result['row_count']} rows "
                    f"in {result.get('execution_time_seconds', 0):.3f}s"
                )

                # Log success
                if state.get("status_logger"):
                    state["status_logger"].log_step(
                        step_name="Execute Query",
                        status=StepStatus.COMPLETED,
                        details=f"Query returned {result['row_count']:,} rows in {result.get('execution_time_seconds', 0):.3f}s",
                        reasoning=[
                            f"✓ Query executed successfully",
                            f"Rows returned: {result['row_count']:,}",
                            f"Execution time: {result.get('execution_time_seconds', 0):.3f}s"
                        ],
                        metadata={
                            "row_count": result['row_count'],
                            "execution_time_seconds": result.get('execution_time_seconds', 0),
                            "column_names": result.get('column_names', [])
                        }
                    )

        except Exception as e:
            state["execution_success"] = False
            state["execution_error"] = str(e)
            logger.error(f"Error executing query: {e}")

            # Log exception
            if state.get("status_logger"):
                state["status_logger"].log_step(
                    step_name="Execute Query",
                    status=StepStatus.FAILED,
                    details=f"Exception during query execution",
                    reasoning=[f"Error: {str(e)}"]
                )

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

    def route_after_clarification(self, state: AgentState) -> str:
        """Route based on whether clarification is needed."""
        if state.get("needs_clarification", False):
            return "clarify"  # Return clarification question to user
        return "proceed"  # Continue with SQL generation

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
        """
        Format chat history for context.

        Now includes FULL conversation history for ChatGPT-level context understanding.
        Uses smart truncation only if conversation becomes extremely long (>50 messages).
        """
        if not chat_history:
            return "No previous conversation"

        # Use FULL chat history for best context (no arbitrary 5-message limit!)
        # Only truncate if conversation is extremely long (>50 messages = 25 exchanges)
        if len(chat_history) > 50:
            # Keep first 10 messages (context) + last 40 messages (recent conversation)
            context_start = chat_history[:10]
            recent = chat_history[-40:]
            messages_to_format = context_start + [{"role": "system", "content": "... [conversation continued] ..."}] + recent
        else:
            # Use ALL messages for full context
            messages_to_format = chat_history

        formatted = []

        for msg in messages_to_format:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "user":
                formatted.append(f"User: {content}")
            elif role == "assistant":
                formatted.append(f"Assistant: {content}")
            elif role == "system":
                formatted.append(f"[{content}]")

        return "\n".join(formatted) if formatted else "No previous conversation"

    # ===== Public API =====

    async def ask(
        self,
        question: str,
        chat_history: List[Dict[str, str]] = None,
        status_callback: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Process a natural language question and return an answer.

        Args:
            question: Natural language question
            chat_history: Previous conversation messages
            status_callback: Optional callback for real-time status updates

        Returns:
            Dictionary with answer, metadata, and reasoning_steps
        """
        from utils.status_logger import StatusLogger

        if chat_history is None:
            chat_history = []

        # Create StatusLogger with callback
        status_logger = StatusLogger(callback=status_callback)

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
            "status_logger": status_logger,  # Add StatusLogger to state
        }

        try:
            # Run graph (synchronous, but execute_query_node handles async internally)
            result_state = self.graph.invoke(initial_state)

            # Get status summary
            status_summary = status_logger.get_summary()

            return {
                "answer": result_state.get("answer", ""),
                "metadata": result_state.get("metadata", {}),
                "sql_query": result_state.get("sql_query"),
                "success": result_state.get("execution_success", False),
                "reasoning_steps": status_summary["steps"],  # Include reasoning
                "total_time": status_summary["total_time_seconds"]
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
                    "reasoning_steps": status_logger.get_summary()["steps"],
                    "total_time": status_logger.get_summary()["total_time_seconds"]
                }

            logger.error(f"Error in ask(): {e}", exc_info=True)
            return {
                "answer": "I encountered an issue processing your question. Please try again.",
                "metadata": {"error": str(e)},
                "sql_query": None,
                "success": False,
                "reasoning_steps": status_logger.get_summary()["steps"],
                "total_time": status_logger.get_summary()["total_time_seconds"]
            }

