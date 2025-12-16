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
from tools.denial_analysis_tool import DenialAnalysisTool

logger = logging.getLogger(__name__)


# ===== State Definition =====

class AgentState(TypedDict):
    """State for the data query agent."""
    # Input
    question: str
    chat_history: List[Dict[str, str]]

    # Question Interpretation (NEW for better inference)
    interpreted_question: Optional[str]
    question_type: Optional[str]  # 'definition', 'data_query', 'kpi_calculation', etc.
    detected_acronyms: List[str]
    reasoning: List[str]  # Verbose reasoning steps

    # Clarification (for better UX)
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

    # Weekly report
    is_weekly_report: bool
    company_name: Optional[str]

    # Denial analysis
    is_denial_analysis: bool

    # Multi-query analytical mode (NEW)
    is_analytical_query: bool
    analytical_pattern: Optional[str]  # trend_analysis, root_cause, comparison, correlation
    detected_kpis: List[str]
    detected_timeframes: List[str]
    query_plan: Optional[Dict[str, Any]]  # QueryPlan structure
    multi_query_results: Optional[Dict[str, Dict[str, Any]]]  # Dict[query_id, QueryResult]
    sql_generation_errors: List[Dict[str, str]]
    successful_queries: List[str]
    failed_queries: List[str]

    # Status logging
    status_logger: Optional[Any]  # StatusLogger instance


# ===== Multi-Query Data Structures =====

class SubQuery(TypedDict):
    """Represents a single sub-query in a multi-query plan."""
    query_id: str  # e.g., "q1", "q2"
    description: str  # What this query answers
    purpose: str  # baseline_data, contributing_factors, correlations
    priority: int  # 1=critical, 2=important, 3=supplementary
    sql_query: Optional[str]  # Generated SQL (None initially)
    depends_on: List[str]  # List of query_ids this depends on (for sequential execution)


class QueryPlan(TypedDict):
    """Represents a complete multi-query analytical plan."""
    analytical_pattern: str  # trend_analysis, root_cause, comparison, correlation
    detected_kpis: List[str]  # KPIs mentioned in question
    detected_timeframes: List[str]  # Time periods mentioned
    sub_queries: List[SubQuery]  # 2-5 focused sub-queries
    reasoning: List[str]  # LLM's reasoning for this decomposition


class QueryResult(TypedDict):
    """Results from executing a single sub-query."""
    query_id: str
    success: bool
    rows: Optional[List[Dict[str, Any]]]  # Query results
    row_count: int
    execution_time_seconds: float
    error: Optional[str]  # Error message if failed


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

        # Initialize denial analysis tool
        self.denial_analysis_tool = DenialAnalysisTool(duckdb_catalog)

        # Load schema profiles
        self.schema_profiles = self._load_schema_profiles()

        # Load few-shot examples for better SQL generation
        self.query_examples = self._load_query_examples()

        # Load table relationships for multi-table queries
        self.relationships = self._load_relationships()

        # Load business rules and domain knowledge
        self.business_rules = self._load_business_rules()

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

    def _load_query_examples(self) -> List[Dict[str, Any]]:
        """Load few-shot query examples from JSON file."""
        examples = []

        # Look for examples file in schemas directory
        examples_path = Path(__file__).parent.parent.parent / "schemas" / "query_examples_duckdb.json"

        if not examples_path.exists():
            logger.warning(f"Query examples file not found at {examples_path}")
            return examples

        try:
            with open(examples_path, 'r') as f:
                data = json.load(f)
                examples = data.get("examples", [])
                logger.info(f"Loaded {len(examples)} query examples")
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
                logger.info(f"Loaded {len(data.get('tables', {}))} table relationships")
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
                logger.info(f"Loaded business rules with {len(data.get('kpi_definitions', {}))} KPI definitions")
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
                context_parts.append(f"  SQL: {kpi_info['sql_template']}")

        # Check if question involves date filtering
        if any(word in question_lower for word in ['this month', 'last month', 'this year', 'current', 'ytd', 'last week']):
            common_filters = self.business_rules.get("common_filters", {})
            context_parts.append("\nCommon Date Filters:")
            for filter_name, filter_info in list(common_filters.items())[:3]:
                if isinstance(filter_info, dict):
                    context_parts.append(f"  {filter_name}: {filter_info.get('sql', '')}")

        return "\n".join(context_parts) if context_parts else ""

    def _interpret_question(self, question: str) -> Dict[str, Any]:
        """
        Interpret the question with verbose reasoning to understand user intent.

        This method adds intelligence and inference capabilities:
        - Detects terminology/definition questions
        - Expands acronyms to full terms
        - Provides reasoning steps (verbose logging)
        - Interprets what the user is really asking for

        Returns:
            Dict with interpreted_question, question_type, detected_acronyms, and reasoning steps
        """
        reasoning = []
        detected_acronyms = []
        question_lower = question.lower().strip()

        # Get terminology from business rules
        terminology = self.business_rules.get("terminology", {})
        kpi_definitions = self.business_rules.get("kpi_definitions", {})

        reasoning.append(f"📝 Original question: '{question}'")

        # Pattern 1: Definition/Terminology questions
        # Check if this is ACTUALLY a definition question (not a data query with year/date)

        import re

        # First, check if the question contains contextual indicators that make it a DATA query
        data_query_indicators = [
            r'for\s+(the\s+)?(year|month|quarter|week|day)',  # "for the year 2025"
            r'in\s+\d{4}',  # "in 2025"
            r'for\s+\d{4}',  # "for 2025"
            r'during\s+\d{4}',  # "during 2025"
            r'this\s+(year|month|quarter)',  # "this year"
            r'last\s+(year|month|quarter)',  # "last year"
            r'from\s+\d{4}',  # "from 2025"
            r'between\s+',  # "between 2024 and 2025"
        ]

        is_data_query_context = any(re.search(indicator, question_lower) for indicator in data_query_indicators)

        if is_data_query_context:
            reasoning.append(f"🔍 Detected contextual indicators (year/date) - this is a DATA QUERY, not a definition")
        else:
            # Now check if it's a definition question
            definition_patterns = [
                r'^what\s+(is|does|are|means?)\s+(the\s+)?([a-z\s]+?)(\?|$)',  # "what is net collection rate?"
                r'^define\s+([a-z\s]+?)(\?|$)',  # "define net collection rate"
                r'^explain\s+([a-z\s]+?)(\?|$)',  # "explain net collection rate"
                r'^([a-z\s]+?)\s+definition',  # "net collection rate definition"
                r'^([a-z\s]+?)\s+meaning',  # "net collection rate meaning"
            ]

            for pattern in definition_patterns:
                match = re.search(pattern, question_lower)
                if match:
                    # Extract the term being asked about (last captured group before ? or end)
                    captured_groups = match.groups()
                    # Get the actual term (skip 'the' and other determiners)
                    term_raw = captured_groups[-2] if len(captured_groups) > 1 else captured_groups[-1]
                    term = term_raw.strip().upper()

                    reasoning.append(f"🔍 Detected: This is a DEFINITION question about '{term}'")

                    # Check if it's an acronym we know
                    if term in terminology:
                        detected_acronyms.append(term)
                        full_term = terminology[term]
                        reasoning.append(f"✓ Found acronym: {term} = {full_term}")

                        # Check if it's also a KPI
                        for kpi_name, kpi_info in kpi_definitions.items():
                            if term in kpi_name.upper() or kpi_name.upper() in term:
                                reasoning.append(f"📊 This is a KPI metric: {kpi_name}")
                                reasoning.append(f"   Formula: {kpi_info['formula']}")

                                return {
                                    "interpreted_question": f"Explain the {kpi_name} metric and its calculation",
                                    "question_type": "kpi_definition",
                                    "detected_acronyms": detected_acronyms,
                                    "reasoning": reasoning,
                                    "kpi_name": kpi_name,
                                    "kpi_info": kpi_info
                                }

                        return {
                            "interpreted_question": f"Explain what {term} ({full_term}) means",
                            "question_type": "terminology_definition",
                            "detected_acronyms": detected_acronyms,
                            "reasoning": reasoning,
                            "term": term,
                            "definition": full_term
                        }

                    # Even if not in terminology dict, check KPI definitions by matching the full phrase
                    term_lower = term_raw.strip().lower()
                    for kpi_name, kpi_info in kpi_definitions.items():
                        # Check if the asked term matches the KPI name
                        kpi_name_lower = kpi_name.lower()
                        # Match if term is in KPI name or vice versa
                        if term_lower in kpi_name_lower or kpi_name_lower in term_lower:
                            reasoning.append(f"📊 This is a KPI metric: {kpi_name}")
                            reasoning.append(f"   Formula: {kpi_info['formula']}")

                            return {
                                "interpreted_question": f"Explain the {kpi_name} metric and its calculation",
                                "question_type": "kpi_definition",
                                "detected_acronyms": detected_acronyms,
                                "reasoning": reasoning,
                                "kpi_name": kpi_name,
                                "kpi_info": kpi_info
                            }

        # Pattern 2: Data queries with acronyms - expand them for better SQL generation
        expanded_question = question
        for acronym, full_term in terminology.items():
            # Case-insensitive replacement but preserve original casing context
            if acronym.lower() in question_lower:
                # Check if it's a standalone word (not part of another word)
                import re
                pattern = r'\b' + re.escape(acronym) + r'\b'
                if re.search(pattern, question, re.IGNORECASE):
                    detected_acronyms.append(acronym)
                    # Expand in the interpreted question for LLM
                    expanded_question = re.sub(pattern, f"{acronym} ({full_term})", expanded_question, flags=re.IGNORECASE)
                    reasoning.append(f"🔄 Expanded acronym: {acronym} → {full_term}")

        # Pattern 3: KPI calculation queries
        kpi_patterns = ['calculate', 'compute', 'rate', 'ratio', 'percentage']
        if any(pattern in question_lower for pattern in kpi_patterns):
            for kpi_name in kpi_definitions.keys():
                if any(word in question_lower for word in kpi_name.lower().split()):
                    reasoning.append(f"📊 Detected KPI calculation request: {kpi_name}")
                    return {
                        "interpreted_question": expanded_question,
                        "question_type": "kpi_calculation",
                        "detected_acronyms": detected_acronyms,
                        "reasoning": reasoning,
                        "kpi_name": kpi_name
                    }

        # Pattern 4: Categorization/Grouping queries
        grouping_patterns = ['categorized by', 'grouped by', 'breakdown by', 'segmented by', 'by']
        if any(pattern in question_lower for pattern in grouping_patterns):
            reasoning.append("📊 Detected: This is a DATA GROUPING/CATEGORIZATION query")

            # Check if grouping by a metric (like NCR or GCR)
            for acronym in detected_acronyms:
                if acronym in ['NCR', 'GCR']:
                    reasoning.append(f"⚠️  User wants to GROUP BY {acronym} - this requires calculating {acronym} ranges/buckets")

        # Default: treat as data query
        if not reasoning or len(reasoning) == 1:  # Only has the original question
            reasoning.append("💾 Interpreted as: DATA QUERY (will generate SQL)")

        return {
            "interpreted_question": expanded_question,
            "question_type": "data_query",
            "detected_acronyms": detected_acronyms,
            "reasoning": reasoning
        }

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

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("detect_intent", self.detect_intent_node)
        workflow.add_node("handle_weekly_report", self.handle_weekly_report_node)
        workflow.add_node("handle_denial_analysis", self.handle_denial_analysis_node)
        workflow.add_node("interpret_question", self.interpret_question_node)
        workflow.add_node("load_schema", self.load_schema_node)
        workflow.add_node("check_clarification", self.check_clarification_node)
        workflow.add_node("detect_analytical_pattern", self.detect_analytical_pattern_node)
        # Multi-query path nodes
        workflow.add_node("plan_multi_query", self.plan_multi_query_node)
        workflow.add_node("generate_multi_sql", self.generate_multi_sql_node)
        workflow.add_node("execute_multi_query", self.execute_multi_query_node)
        workflow.add_node("synthesize_analytical_response", self.synthesize_analytical_response_node)
        # Single-query path nodes (existing)
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
                "denial_analysis": "handle_denial_analysis",
                "normal_query": "interpret_question"  # Changed: interpret question first
            }
        )

        workflow.add_edge("handle_weekly_report", END)
        workflow.add_edge("handle_denial_analysis", END)

        # After interpretation, either answer definition or continue to schema loading
        workflow.add_conditional_edges(
            "interpret_question",
            self.route_after_interpretation,
            {
                "definition_answered": END,  # Definition questions answered directly
                "continue": "load_schema"  # Data queries continue to SQL generation
            }
        )

        # After loading schema, check if clarification is needed
        workflow.add_edge("load_schema", "check_clarification")

        # Route based on clarification check
        workflow.add_conditional_edges(
            "check_clarification",
            self.route_after_clarification,
            {
                "clarify": END,  # Return clarification question to user
                "proceed": "detect_analytical_pattern"  # Check if multi-query needed
            }
        )

        # Route based on analytical pattern detection
        workflow.add_conditional_edges(
            "detect_analytical_pattern",
            self.route_query_strategy,
            {
                "plan_multi_query": "plan_multi_query",  # Multi-query path for analytical questions
                "generate_sql": "generate_sql"  # Single query path for simple questions
            }
        )

        # ===== MULTI-QUERY PATH =====
        # plan_multi_query → check if planning succeeded
        workflow.add_conditional_edges(
            "plan_multi_query",
            self.check_query_plan,
            {
                "proceed": "generate_multi_sql",  # Planning succeeded, generate SQL
                "error": "handle_error"  # Planning failed
            }
        )

        # generate_multi_sql → check if any SQL was generated
        workflow.add_conditional_edges(
            "generate_multi_sql",
            self.check_multi_sql_generated,
            {
                "proceed": "execute_multi_query",  # At least one SQL generated
                "error": "handle_error"  # All SQL generation failed
            }
        )

        # execute_multi_query → check if any queries succeeded
        workflow.add_conditional_edges(
            "execute_multi_query",
            self.check_multi_execution,
            {
                "synthesize": "synthesize_analytical_response",  # At least one query succeeded
                "error": "handle_error"  # All queries failed
            }
        )

        # synthesize_analytical_response → END (answer ready)
        workflow.add_edge("synthesize_analytical_response", END)

        # ===== SINGLE-QUERY PATH (existing) =====
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
        """Detect if the question is asking for a weekly report or denial analysis."""
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

        # Check for denial analysis patterns
        denial_patterns = [
            "/denial",
            "denial analysis",
            "denial report",
            "denied claims",
            "show me denials",
            "denial metrics",
            "claims denial",
            "denial percentage"
        ]

        is_weekly_report = any(pattern in question for pattern in weekly_patterns)
        is_denial_analysis = any(pattern in question for pattern in denial_patterns)

        state["is_weekly_report"] = is_weekly_report
        state["is_denial_analysis"] = is_denial_analysis

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

        logger.info(f"Intent detection: is_weekly_report={is_weekly_report}, is_denial_analysis={is_denial_analysis}, company={company_name}")

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

    def handle_denial_analysis_node(self, state: AgentState) -> AgentState:
        """Handle denial analysis report generation."""
        logger.info(f"Generating denial analysis report for {state.get('company_name', 'Company')}...")

        try:
            # Generate denial analysis report (shows both visit_date and transaction_date)
            report = self.denial_analysis_tool.generate_report(
                company_name=state.get("company_name", "Company")
            )

            state["answer"] = report
            state["execution_success"] = True
            state["metadata"] = {
                "report_type": "denial_analysis",
                "company": state.get("company_name", "Company")
            }

        except Exception as e:
            logger.error(f"Error generating denial analysis report: {e}")
            state["answer"] = f"❌ Error generating denial analysis report: {str(e)}"
            state["execution_success"] = False

        return state

    def interpret_question_node(self, state: AgentState) -> AgentState:
        """
        Interpret the user's question with verbose reasoning.

        This adds intelligence:
        - Detects if it's a definition question vs data query
        - Expands acronyms (NCR → Net Collection Rate)
        - Shows reasoning steps (verbose logging)
        - Handles definition questions without SQL
        """
        from utils.status_logger import StepStatus

        logger.info("=" * 80)
        logger.info("🧠 QUESTION INTERPRETATION PHASE")
        logger.info("=" * 80)

        # Log to status
        if state.get("status_logger"):
            state["status_logger"].log_step(
                step_name="Interpret Question",
                status=StepStatus.RUNNING,
                details="Analyzing question to understand user intent",
                reasoning=[
                    f"Question: {state['question']}",
                    "Checking if this is a definition or data query",
                    "Detecting acronyms and expanding them"
                ]
            )

        # Interpret the question
        interpretation = self._interpret_question(state["question"])

        # Log reasoning steps (VERBOSE)
        logger.info("\n📋 REASONING PROCESS:")
        for step in interpretation["reasoning"]:
            logger.info(f"   {step}")

        # Store interpretation in state
        state["interpreted_question"] = interpretation["interpreted_question"]
        state["question_type"] = interpretation["question_type"]
        state["detected_acronyms"] = interpretation.get("detected_acronyms", [])
        state["reasoning"] = interpretation["reasoning"]

        logger.info(f"\n🎯 Question Type: {interpretation['question_type']}")
        logger.info(f"📝 Interpreted Question: {interpretation['interpreted_question']}")

        # Log completion
        if state.get("status_logger"):
            state["status_logger"].log_step(
                step_name="Interpret Question",
                status=StepStatus.COMPLETED,
                details=f"Identified as: {interpretation['question_type']}",
                reasoning=interpretation["reasoning"],
                metadata={
                    "question_type": interpretation["question_type"],
                    "detected_acronyms": interpretation.get("detected_acronyms", []),
                    "interpreted_question": interpretation["interpreted_question"]
                }
            )

        # Handle definition/terminology questions directly (no SQL needed)
        if interpretation["question_type"] == "kpi_definition":
            kpi_name = interpretation.get("kpi_name")
            kpi_info = interpretation.get("kpi_info")

            answer = f"**{kpi_name}**\n\n"
            answer += f"**Definition:** {kpi_info.get('description', 'N/A')}\n\n"
            answer += f"**Formula:** {kpi_info.get('formula', 'N/A')}\n\n"
            answer += f"**SQL Calculation:** `{kpi_info.get('sql_template', 'N/A')}`\n\n"
            answer += f"**Category:** {kpi_info.get('category', 'N/A')}\n\n"

            if kpi_info.get("required_columns"):
                answer += f"**Required Data:** {', '.join(kpi_info['required_columns'])}\n\n"

            answer += "\n💡 **Want to see actual data?** Ask me to calculate this metric for a specific time period!"

            state["answer"] = answer
            state["execution_success"] = True
            state["metadata"] = {
                "question_type": "kpi_definition",
                "kpi_name": kpi_name,
                "handled_without_sql": True
            }

            logger.info("\n✅ Definition question answered directly (no SQL needed)")
            logger.info("=" * 80)

        elif interpretation["question_type"] == "terminology_definition":
            term = interpretation.get("term")
            definition = interpretation.get("definition")

            answer = f"**{term}** stands for **{definition}**.\n\n"

            # Add context if available
            terminology = self.business_rules.get("terminology", {})
            if term in terminology:
                answer += f"{terminology[term]}\n\n"

            answer += "\n💡 **Want to see related metrics?** Ask me to calculate or show data related to this term!"

            state["answer"] = answer
            state["execution_success"] = True
            state["metadata"] = {
                "question_type": "terminology_definition",
                "term": term,
                "handled_without_sql": True
            }

            logger.info("\n✅ Terminology question answered directly (no SQL needed)")
            logger.info("=" * 80)

        else:
            # Data query - continue to SQL generation
            logger.info(f"\n➡️  Proceeding to SQL generation with interpreted question")
            logger.info("=" * 80)

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

                        # Add statistical context for better LLM understanding
                        stats_parts = []

                        # For categorical columns: show top values
                        if col.get("top_values"):
                            top_vals = col["top_values"][:3]  # Show top 3
                            vals_str = ", ".join([f"'{v['value']}' ({v['percentage']:.0f}%)"
                                                 for v in top_vals])
                            stats_parts.append(f"Common values: {vals_str}")

                        # For numeric columns: show range
                        elif col.get("data_category") == "numeric" and col.get("min") is not None:
                            min_val = col.get("min", 0)
                            max_val = col.get("max", 0)
                            avg_val = col.get("mean", 0)
                            stats_parts.append(f"Range: {min_val:.2f} to {max_val:.2f}, Avg: {avg_val:.2f}")

                        # For datetime columns: show date range
                        elif col.get("data_category") == "datetime" and col.get("min"):
                            min_date = col.get("min", "")
                            max_date = col.get("max", "")
                            stats_parts.append(f"Date range: {min_date} to {max_date}")

                        # Add null percentage if significant
                        null_pct = col.get("null_percentage", 0)
                        if null_pct > 5:  # Only show if >5% nulls
                            stats_parts.append(f"{null_pct:.1f}% null")

                        if stats_parts:
                            col_desc += f" [{'; '.join(stats_parts)}]"

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

    def detect_analytical_pattern_node(self, state: AgentState) -> AgentState:
        """
        Detect if the question requires multi-query analytical decomposition.

        Identifies analytical patterns (trend, root cause, comparison, correlation)
        and extracts KPIs and timeframes mentioned in the question.
        """
        from utils.status_logger import StepStatus

        logger.info("Detecting analytical pattern...")

        # Log start
        if state.get("status_logger"):
            state["status_logger"].log_step(
                step_name="Detect Analytical Pattern",
                status=StepStatus.RUNNING,
                details="Analyzing question to determine if multi-query decomposition is needed"
            )

        question_lower = state["question"].lower()

        # Pattern 1: Trend Analysis
        trend_keywords = ["trend", "increase", "decrease", "decline", "growth",
                         "change over time", "changed", "changing", "going up", "going down",
                         "rising", "falling", "grew", "dropped"]

        # Pattern 2: Root Cause Analysis
        root_cause_keywords = ["why", "what caused", "factors", "reasons",
                              "causes", "explain the", "behind", "drivers",
                              "what drove", "what led to"]

        # Pattern 3: Comparison Analysis
        comparison_keywords = ["compare", "vs", "versus", "difference between",
                              "month-over-month", "year-over-year", "yoy", "mom",
                              "compared to", "comparison"]

        # Pattern 4: Correlation Analysis
        correlation_keywords = ["correlation", "relationship", "impact",
                               "affect", "related to", "connection between",
                               "associated with", "linked to"]

        # Detect pattern
        analytical_pattern = None
        if any(kw in question_lower for kw in root_cause_keywords):
            analytical_pattern = "root_cause"
        elif any(kw in question_lower for kw in trend_keywords):
            analytical_pattern = "trend_analysis"
        elif any(kw in question_lower for kw in comparison_keywords):
            analytical_pattern = "comparison"
        elif any(kw in question_lower for kw in correlation_keywords):
            analytical_pattern = "correlation"

        # Detect KPIs mentioned
        detected_kpis = []
        if self.business_rules:
            kpi_defs = self.business_rules.get("kpi_definitions", {})
            terminology = self.business_rules.get("terminology", {})

            # First, check for acronyms using the terminology mapping
            for acronym in state.get("detected_acronyms", []):
                acronym_upper = acronym.upper()
                # Check if acronym is in terminology
                if acronym_upper in terminology:
                    full_term = terminology[acronym_upper]
                    # Find matching KPI
                    for kpi_name in kpi_defs.keys():
                        if full_term.lower() in kpi_name.lower() or kpi_name.lower() in full_term.lower():
                            if kpi_name not in detected_kpis:
                                detected_kpis.append(kpi_name)

            # Then check for full KPI names in the question
            for kpi_name in kpi_defs.keys():
                # Check both full name and lowercase
                if kpi_name.lower() in question_lower:
                    if kpi_name not in detected_kpis:
                        detected_kpis.append(kpi_name)

        # Detect timeframes
        detected_timeframes = []
        import re

        # Look for month names
        months = ["january", "february", "march", "april", "may", "june",
                 "july", "august", "september", "october", "november", "december",
                 "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
        for month in months:
            if month in question_lower:
                detected_timeframes.append(month.capitalize())

        # Look for years
        years = re.findall(r'\b(20\d{2})\b', question_lower)
        detected_timeframes.extend(years)

        # Look for quarter references
        if "q1" in question_lower or "quarter 1" in question_lower:
            detected_timeframes.append("Q1")
        if "q2" in question_lower or "quarter 2" in question_lower:
            detected_timeframes.append("Q2")
        if "q3" in question_lower or "quarter 3" in question_lower:
            detected_timeframes.append("Q3")
        if "q4" in question_lower or "quarter 4" in question_lower:
            detected_timeframes.append("Q4")

        # Determine if analytical query
        # Requires: pattern detected + KPI mentioned (or strong analytical intent)
        is_analytical = bool(analytical_pattern and (detected_kpis or len(detected_timeframes) > 1))

        # Update state
        state["is_analytical_query"] = is_analytical
        state["analytical_pattern"] = analytical_pattern
        state["detected_kpis"] = detected_kpis
        state["detected_timeframes"] = detected_timeframes

        logger.info(f"Analytical Detection: pattern={analytical_pattern}, "
                   f"kpis={detected_kpis}, timeframes={detected_timeframes}, "
                   f"is_analytical={is_analytical}")

        # Log completion
        if state.get("status_logger"):
            state["status_logger"].log_step(
                step_name="Detect Analytical Pattern",
                status=StepStatus.COMPLETED,
                details=f"Pattern: {analytical_pattern or 'none'}, Mode: {'MULTI-QUERY' if is_analytical else 'SINGLE-QUERY'}",
                reasoning=[
                    f"Analytical pattern detected: {analytical_pattern or 'none'}",
                    f"KPIs identified: {', '.join(detected_kpis) if detected_kpis else 'none'}",
                    f"Timeframes detected: {', '.join(detected_timeframes) if detected_timeframes else 'none'}",
                    f"Will use: {'Multi-query decomposition' if is_analytical else 'Single SQL query'}"
                ],
                metadata={
                    "is_analytical": is_analytical,
                    "pattern": analytical_pattern,
                    "kpi_count": len(detected_kpis),
                    "timeframe_count": len(detected_timeframes)
                }
            )

        return state

    def plan_multi_query_node(self, state: AgentState) -> AgentState:
        """
        Plan multi-query decomposition for analytical questions.

        Uses LLM to decompose the question into 2-5 focused sub-queries.
        Each sub-query has a specific purpose and priority.
        """
        from utils.status_logger import StepStatus
        import json

        logger.info("Planning multi-query decomposition...")

        # Log start
        if state.get("status_logger"):
            state["status_logger"].log_step(
                step_name="Plan Multi-Query Decomposition",
                status=StepStatus.RUNNING,
                details="Breaking down analytical question into focused sub-queries"
            )

        question = state["question"]
        analytical_pattern = state.get("analytical_pattern", "unknown")
        detected_kpis = state.get("detected_kpis", [])
        detected_timeframes = state.get("detected_timeframes", [])

        # Get business context for KPIs
        business_context = self._get_business_context(question)

        # Build the query planning prompt
        query_planning_prompt = f"""You are an expert data analyst specializing in breaking down complex analytical questions into focused sub-queries.

ANALYTICAL QUESTION: "{question}"

DETECTED CONTEXT:
- Pattern Type: {analytical_pattern}
- KPIs Mentioned: {', '.join(detected_kpis) if detected_kpis else 'none'}
- Timeframes: {', '.join(detected_timeframes) if detected_timeframes else 'none'}

BUSINESS RULES & KPI DEFINITIONS:
{business_context}

YOUR TASK:
Decompose this analytical question into 2-5 focused sub-queries. Each sub-query should:
1. Answer a specific aspect of the overall question
2. Be simple enough to translate into a single SQL query
3. Provide data needed to synthesize the final comprehensive answer

GUIDELINES:
- For ROOT CAUSE questions: Break down into baseline data + contributing factors
  * CRITICAL: If the question mentions a time range (e.g., "January to November"), get MONTH-BY-MONTH progression, NOT just start/end points
  * Example: "Why did NCR decline Jan to Nov?" → Query should get NCR for EACH month (Jan, Feb, Mar, Apr, May, Jun, Jul, Aug, Sep, Oct, Nov)
  * This reveals WHEN the decline happened and HOW it progressed over time

- For TREND questions: Get data points over time with proper granularity
  * Monthly data for questions spanning multiple months
  * Quarterly data for questions spanning multiple quarters
  * Yearly data for questions spanning multiple years

- For COMPARISON questions: Get separate data for each entity/timeframe being compared
  * Include proper time dimensions (month, quarter, year) for grouping

- For CORRELATION questions: Get both variables independently, then compare
  * Ensure both metrics are calculated over the same time periods

TIME-BASED QUERY RULES:
1. When a question mentions "from X to Y" time period, capture ALL time points in between, not just X and Y
2. Each sub-query should request data "by month" or "by quarter" to enable proper grouping
3. Contributing factor queries should use the SAME time granularity as the baseline query

Sub-query priorities:
- Priority 1 (Critical): Baseline data needed to answer the question
- Priority 2 (Important): Contributing factors or additional context
- Priority 3 (Supplementary): Nice-to-have supporting data

OUTPUT FORMAT (JSON):
{{
  "reasoning": [
    "Step-by-step explanation of how you broke down the question",
    "Why each sub-query is needed",
    "How the results will combine to answer the question"
  ],
  "sub_queries": [
    {{
      "query_id": "q1",
      "description": "Clear description of what this query answers",
      "purpose": "baseline_data | contributing_factors | correlations",
      "priority": 1,
      "depends_on": []
    }},
    {{
      "query_id": "q2",
      "description": "...",
      "purpose": "contributing_factors",
      "priority": 2,
      "depends_on": ["q1"]
    }}
  ]
}}

IMPORTANT:
- Generate 2-5 sub-queries (no more, no less)
- Each description should be specific enough to generate SQL
- Use detected KPIs and timeframes in your planning
- Return ONLY valid JSON, no explanations outside the JSON structure
"""

        try:
            messages = [
                SystemMessage(content="You are an expert at decomposing analytical questions into focused sub-queries. Always respond with valid JSON."),
                HumanMessage(content=query_planning_prompt)
            ]

            response = self.llm.invoke(messages)

            # Parse JSON response
            response_text = response.content.strip()

            # Clean up response (remove markdown if present)
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()

            query_plan_dict = json.loads(response_text)

            # Validate structure
            if "sub_queries" not in query_plan_dict or not query_plan_dict["sub_queries"]:
                raise ValueError("Query plan must contain at least one sub-query")

            # Enforce 2-5 sub-queries
            if len(query_plan_dict["sub_queries"]) < 2:
                raise ValueError("Query plan must have at least 2 sub-queries")
            if len(query_plan_dict["sub_queries"]) > 5:
                logger.warning(f"Query plan has {len(query_plan_dict['sub_queries'])} sub-queries, truncating to 5")
                query_plan_dict["sub_queries"] = query_plan_dict["sub_queries"][:5]

            # Add detected context to query plan
            query_plan_dict["analytical_pattern"] = analytical_pattern
            query_plan_dict["detected_kpis"] = detected_kpis
            query_plan_dict["detected_timeframes"] = detected_timeframes

            # Initialize sql_query field for each sub-query
            for sub_query in query_plan_dict["sub_queries"]:
                if "sql_query" not in sub_query:
                    sub_query["sql_query"] = None
                if "depends_on" not in sub_query:
                    sub_query["depends_on"] = []

            state["query_plan"] = query_plan_dict

            logger.info(f"✅ Query plan created with {len(query_plan_dict['sub_queries'])} sub-queries")
            for sq in query_plan_dict["sub_queries"]:
                logger.info(f"   {sq['query_id']}: {sq['description']} (priority {sq['priority']})")

            # Log completion
            if state.get("status_logger"):
                state["status_logger"].log_step(
                    step_name="Plan Multi-Query Decomposition",
                    status=StepStatus.COMPLETED,
                    details=f"Created plan with {len(query_plan_dict['sub_queries'])} sub-queries",
                    reasoning=query_plan_dict.get("reasoning", []),
                    metadata={
                        "sub_query_count": len(query_plan_dict["sub_queries"]),
                        "sub_queries": [f"{sq['query_id']}: {sq['description']}" for sq in query_plan_dict["sub_queries"]]
                    }
                )

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse query plan JSON: {e}")
            logger.error(f"Response was: {response_text[:500]}")
            state["error"] = f"Failed to parse query plan: {str(e)}"
            state["query_plan"] = None

            if state.get("status_logger"):
                state["status_logger"].log_step(
                    step_name="Plan Multi-Query Decomposition",
                    status=StepStatus.FAILED,
                    details=f"JSON parsing error: {str(e)}"
                )

        except Exception as e:
            logger.error(f"Error planning multi-query: {e}")
            state["error"] = f"Failed to plan multi-query: {str(e)}"
            state["query_plan"] = None

            if state.get("status_logger"):
                state["status_logger"].log_step(
                    step_name="Plan Multi-Query Decomposition",
                    status=StepStatus.FAILED,
                    details=f"Planning error: {str(e)}"
                )

        return state

    def generate_multi_sql_node(self, state: AgentState) -> AgentState:
        """
        Generate SQL for each sub-query in the multi-query plan.

        Uses focused prompts for each sub-query to generate simple, reliable SQL.
        Tracks failures but continues with other queries (graceful degradation).
        """
        from utils.status_logger import StepStatus

        logger.info("Generating SQL for multi-query plan...")

        # Log start
        if state.get("status_logger"):
            state["status_logger"].log_step(
                step_name="Generate Multi-Query SQL",
                status=StepStatus.RUNNING,
                details="Generating SQL for each sub-query in the plan"
            )

        query_plan = state.get("query_plan")
        if not query_plan or not query_plan.get("sub_queries"):
            logger.error("No query plan available for SQL generation")
            state["error"] = "No query plan available"
            return state

        sql_generation_errors = []
        successful_count = 0

        # Loop through each sub-query and generate SQL
        for sub_query in query_plan["sub_queries"]:
            query_id = sub_query["query_id"]
            description = sub_query["description"]
            purpose = sub_query.get("purpose", "unknown")

            logger.info(f"\n{'='*70}")
            logger.info(f"Generating SQL for {query_id}: {description}")
            logger.info(f"{'='*70}")

            try:
                # Build focused SQL generation prompt for this sub-query
                multi_sql_prompt = f"""You are a SQL expert. Generate a DuckDB SQL query for this specific data request.

SUB-QUERY REQUEST: {description}

PURPOSE: {purpose}

CONTEXT FROM ORIGINAL QUESTION:
- Pattern: {query_plan.get('analytical_pattern', 'unknown')}
- KPIs: {', '.join(query_plan.get('detected_kpis', []))}
- Timeframes: {', '.join(query_plan.get('detected_timeframes', []))}

AVAILABLE SCHEMA:
{state.get('schema_context', 'No schema available')}

BUSINESS RULES:
{self._get_business_context(description)}

CRITICAL DUCKDB RULES:
1. WHERE clause: CANNOT use column aliases - use full expressions
   WRONG: WHERE year = 2025
   CORRECT: WHERE EXTRACT(YEAR FROM visit_date) = 2025

2. GROUP BY clause: CANNOT use column aliases - use full expressions
   WRONG: GROUP BY year, month
   CORRECT: GROUP BY EXTRACT(YEAR FROM visit_date), EXTRACT(MONTH FROM visit_date)

3. DATE extraction: When grouping by date parts, only select the extracted values
   WRONG: SELECT EXTRACT(MONTH FROM visit_date) AS month, visit_date, COUNT(*)
   CORRECT: SELECT EXTRACT(MONTH FROM visit_date) AS month, COUNT(*)

4. Return ONLY the SQL query without explanation, markdown, or formatting
5. Do not include markdown code blocks or backticks
6. Generate ONLY SELECT queries (read-only)

KPI FORMULA RULES (MANDATORY):
1. Use EXACT formulas from business rules - DO NOT modify or substitute columns
   Example: If business rule says "SUM(adjustments)", use EXACTLY that, NOT "SUM(contract_adjustment + non_contract_adjustment)"

2. For Net Collection Rate (NCR):
   REQUIRED FORMULA: (SUM(total_payment) / (SUM(charge) - SUM(adjustments))) * 100
   DO NOT use variations like: contract_adjustment, non_contract_adjustment, or other adjustment columns

3. For Gross Collection Rate (GCR):
   REQUIRED FORMULA: (SUM(total_payment) / SUM(charge)) * 100

4. For Days in AR:
   REQUIRED FORMULA: SUM(balance) / (SUM(charge) / period_days)
   Where period_days should be calculated from the actual date range in the query

TIME-BASED QUERY REQUIREMENTS:
1. If the description mentions "by month", "month-by-month", or a time range (e.g., "January to November"):
   - ALWAYS include: GROUP BY EXTRACT(MONTH FROM visit_date), EXTRACT(YEAR FROM visit_date)
   - ALWAYS select: EXTRACT(MONTH FROM visit_date) AS month, EXTRACT(YEAR FROM visit_date) AS year

2. If the description mentions "by quarter":
   - ALWAYS include: GROUP BY EXTRACT(QUARTER FROM visit_date), EXTRACT(YEAR FROM visit_date)

3. If the description mentions "by year":
   - ALWAYS include: GROUP BY EXTRACT(YEAR FROM visit_date)

4. ALWAYS add ORDER BY for time dimensions to show chronological progression:
   - ORDER BY year, month (for monthly data)
   - ORDER BY year, quarter (for quarterly data)
   - ORDER BY year (for yearly data)

Generate a FOCUSED, SIMPLE query that answers ONLY this specific sub-query request.
Keep it simple - this is just one piece of the larger analysis.
"""

                messages = [
                    SystemMessage(content="You are a SQL expert. Generate focused, simple SQL queries. Return only the SQL, no explanations."),
                    HumanMessage(content=multi_sql_prompt)
                ]

                response = self.llm.invoke(messages)
                sql_query = response.content.strip()

                # Clean up SQL (remove markdown formatting if present)
                sql_query = sql_query.replace("```sql", "").replace("```", "").strip()

                # Store SQL in the sub-query
                sub_query["sql_query"] = sql_query
                successful_count += 1

                logger.info(f"✅ Generated SQL for {query_id}")
                logger.info(f"SQL: {sql_query[:200]}...")

                # Also print to console for visibility
                print(f"\n{'='*70}")
                print(f"✅ Generated SQL for {query_id}: {description}")
                print(f"{'='*70}")
                print(sql_query)
                print(f"{'='*70}\n")

            except Exception as e:
                logger.error(f"❌ Failed to generate SQL for {query_id}: {e}")
                sub_query["sql_query"] = None
                sql_generation_errors.append({
                    "query_id": query_id,
                    "description": description,
                    "error": str(e)
                })

        # Update state
        state["query_plan"] = query_plan
        state["sql_generation_errors"] = sql_generation_errors

        # Log completion
        total_queries = len(query_plan["sub_queries"])
        failed_count = len(sql_generation_errors)

        if successful_count > 0:
            logger.info(f"\n✅ SQL generation complete: {successful_count}/{total_queries} succeeded")

            if state.get("status_logger"):
                state["status_logger"].log_step(
                    step_name="Generate Multi-Query SQL",
                    status=StepStatus.COMPLETED,
                    details=f"Generated SQL for {successful_count}/{total_queries} sub-queries",
                    metadata={
                        "successful_count": successful_count,
                        "failed_count": failed_count,
                        "total_queries": total_queries
                    }
                )
        else:
            logger.error("❌ Failed to generate SQL for all sub-queries")
            state["error"] = "Failed to generate SQL for all sub-queries"

            if state.get("status_logger"):
                state["status_logger"].log_step(
                    step_name="Generate Multi-Query SQL",
                    status=StepStatus.FAILED,
                    details="Failed to generate SQL for all sub-queries"
                )

        return state

    def execute_multi_query_node(self, state: AgentState) -> AgentState:
        """
        Execute multiple SQL queries in parallel with graceful degradation.

        Queries are grouped by priority and executed in priority order.
        Within each priority, queries run in parallel for performance.
        Continues with partial results if some queries fail.
        """
        from utils.status_logger import StepStatus
        import time
        from concurrent.futures import ThreadPoolExecutor, as_completed

        logger.info("Executing multi-query plan...")

        # Log start
        if state.get("status_logger"):
            state["status_logger"].log_step(
                step_name="Execute Multi-Query Plan",
                status=StepStatus.RUNNING,
                details="Executing SQL queries in parallel"
            )

        query_plan = state.get("query_plan")
        if not query_plan or not query_plan.get("sub_queries"):
            logger.error("No query plan available for execution")
            state["error"] = "No query plan available"
            return state

        # Group queries by priority
        queries_by_priority = {}
        for sub_query in query_plan["sub_queries"]:
            if sub_query.get("sql_query"):  # Only include queries with SQL
                priority = sub_query.get("priority", 2)
                if priority not in queries_by_priority:
                    queries_by_priority[priority] = []
                queries_by_priority[priority].append(sub_query)

        # Initialize results tracking
        multi_query_results = {}
        successful_queries = []
        failed_queries = []

        def execute_single_query(sub_query):
            """Execute a single sub-query and return results."""
            query_id = sub_query["query_id"]
            sql_query = sub_query["sql_query"]
            start_time = time.time()

            try:
                logger.info(f"Executing {query_id}...")
                result = self.catalog.execute_query(sql_query, timeout=60, max_rows=10000)

                execution_time = time.time() - start_time

                if result["success"]:
                    logger.info(f"✅ {query_id} completed: {result['row_count']} rows in {execution_time:.2f}s")
                    return {
                        "query_id": query_id,
                        "success": True,
                        "rows": result.get("rows", []),
                        "row_count": result.get("row_count", 0),
                        "execution_time_seconds": execution_time,
                        "error": None
                    }
                else:
                    logger.error(f"❌ {query_id} failed: {result.get('error', 'Unknown error')}")
                    return {
                        "query_id": query_id,
                        "success": False,
                        "rows": None,
                        "row_count": 0,
                        "execution_time_seconds": execution_time,
                        "error": result.get("error", "Unknown error")
                    }

            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"❌ {query_id} exception: {e}")
                return {
                    "query_id": query_id,
                    "success": False,
                    "rows": None,
                    "row_count": 0,
                    "execution_time_seconds": execution_time,
                    "error": str(e)
                }

        # Execute queries by priority (priority 1 first, then 2, then 3)
        for priority in sorted(queries_by_priority.keys()):
            queries = queries_by_priority[priority]
            logger.info(f"\n{'='*70}")
            logger.info(f"Executing Priority {priority} queries ({len(queries)} queries)")
            logger.info(f"{'='*70}")

            # Execute queries in parallel within this priority group
            with ThreadPoolExecutor(max_workers=min(len(queries), 5)) as executor:
                future_to_query = {executor.submit(execute_single_query, sq): sq for sq in queries}

                for future in as_completed(future_to_query):
                    result = future.result()
                    query_id = result["query_id"]
                    multi_query_results[query_id] = result

                    if result["success"]:
                        successful_queries.append(query_id)
                    else:
                        failed_queries.append(query_id)

        # Update state
        state["multi_query_results"] = multi_query_results
        state["successful_queries"] = successful_queries
        state["failed_queries"] = failed_queries

        # Determine overall success
        # Success if at least one priority 1 query succeeded
        priority_1_queries = [sq["query_id"] for sq in query_plan["sub_queries"] if sq.get("priority") == 1]
        priority_1_successes = [qid for qid in priority_1_queries if qid in successful_queries]

        execution_success = len(priority_1_successes) > 0

        state["execution_success"] = execution_success

        # Log completion
        total_queries = len(multi_query_results)
        success_count = len(successful_queries)
        failed_count = len(failed_queries)

        logger.info(f"\n{'='*70}")
        logger.info(f"Multi-Query Execution Complete")
        logger.info(f"Total: {total_queries}, Succeeded: {success_count}, Failed: {failed_count}")
        logger.info(f"{'='*70}")

        if execution_success:
            if state.get("status_logger"):
                state["status_logger"].log_step(
                    step_name="Execute Multi-Query Plan",
                    status=StepStatus.COMPLETED,
                    details=f"Executed {success_count}/{total_queries} queries successfully",
                    metadata={
                        "total_queries": total_queries,
                        "successful_count": success_count,
                        "failed_count": failed_count,
                        "successful_queries": successful_queries,
                        "failed_queries": failed_queries
                    }
                )
        else:
            logger.error("❌ All critical (priority 1) queries failed")
            state["error"] = "All critical queries failed"

            if state.get("status_logger"):
                state["status_logger"].log_step(
                    step_name="Execute Multi-Query Plan",
                    status=StepStatus.FAILED,
                    details="All critical (priority 1) queries failed"
                )

        return state

    def synthesize_analytical_response_node(self, state: AgentState) -> AgentState:
        """
        Synthesize comprehensive analytical response from multiple query results.

        Uses LLM to analyze all results and generate a narrative explanation
        with specific numbers, trends, and data tables.
        """
        from utils.status_logger import StepStatus
        import json

        logger.info("Synthesizing analytical response...")

        # Log start
        if state.get("status_logger"):
            state["status_logger"].log_step(
                step_name="Synthesize Analytical Response",
                status=StepStatus.RUNNING,
                details="Combining query results into comprehensive narrative"
            )

        question = state["question"]
        query_plan = state.get("query_plan", {})
        multi_query_results = state.get("multi_query_results", {})
        successful_queries = state.get("successful_queries", [])
        failed_queries = state.get("failed_queries", [])

        if not multi_query_results:
            logger.error("No query results to synthesize")
            state["answer"] = "I couldn't retrieve the necessary data to answer your question."
            return state

        # Format query results for LLM
        formatted_results = []
        for query_id in successful_queries:
            result = multi_query_results[query_id]

            # Find the sub-query description
            sub_query = next((sq for sq in query_plan.get("sub_queries", []) if sq["query_id"] == query_id), None)
            description = sub_query["description"] if sub_query else "Unknown"

            # Limit rows to first 50 for LLM context
            rows = result.get("rows", [])[:50]
            row_count = result.get("row_count", 0)

            formatted_results.append({
                "query_id": query_id,
                "description": description,
                "row_count": row_count,
                "rows_shown": len(rows),
                "data": rows
            })

        # Build synthesis prompt
        synthesis_prompt = f"""You are an expert data analyst. Analyze the following query results and provide a comprehensive answer to the user's question.

ORIGINAL QUESTION: "{question}"

ANALYTICAL CONTEXT:
- Pattern: {query_plan.get('analytical_pattern', 'unknown')}
- KPIs: {', '.join(query_plan.get('detected_kpis', []))}
- Timeframes: {', '.join(query_plan.get('detected_timeframes', []))}

QUERY PLAN REASONING:
{chr(10).join(f"• {r}" for r in query_plan.get('reasoning', ['No reasoning provided']))}

QUERY RESULTS:
{json.dumps(formatted_results, indent=2, default=str)}

STATUS:
- Successful queries: {len(successful_queries)}/{len(multi_query_results)}
- Failed queries: {len(failed_queries)}
{f"- Note: Some queries failed: {', '.join(failed_queries)}" if failed_queries else ""}

YOUR TASK:
Generate a comprehensive analytical response that:

1. **NARRATIVE EXPLANATION (2-3 paragraphs)**:
   - Answer the question directly with specific insights
   - Explain trends, causes, or correlations found in the data
   - Cite SPECIFIC NUMBERS from the results (don't just say "increased", say "increased from X to Y")
   - Connect multiple data points to tell a coherent story

2. **KEY FINDINGS (Bullet Points)**:
   - 3-5 most important insights with specific numbers
   - Each finding should be actionable or meaningful

3. **DATA TABLES (Markdown)**:
   - Include relevant data tables to support your narrative
   - Format as markdown tables
   - Show key metrics and trends
   - Limit to most relevant data (don't dump everything)

OUTPUT FORMAT:
Use markdown formatting. Structure your response as:

## Analysis: [Brief Title]

[2-3 paragraph narrative explanation with specific numbers]

### Key Findings
- [Finding 1 with specific numbers]
- [Finding 2 with specific numbers]
- [Finding 3 with specific numbers]

### Supporting Data
[Relevant markdown tables]

CRITICAL RULES:
- ALWAYS cite specific numbers from the data (e.g., "85.3%" not "around 85%")
- Explain WHY changes occurred, don't just describe WHAT changed
- If data is missing due to failed queries, acknowledge it briefly but focus on available data
- Keep tables focused - show only the most relevant rows/columns
- Use clear, professional language appropriate for business stakeholders

Generate your comprehensive analytical response now:
"""

        try:
            messages = [
                SystemMessage(content="You are an expert data analyst. Provide comprehensive, data-driven analytical responses with specific numbers and clear explanations."),
                HumanMessage(content=synthesis_prompt)
            ]

            response = self.llm.invoke(messages)
            answer = response.content.strip()

            state["answer"] = answer

            logger.info("✅ Analytical response synthesized")
            logger.info(f"Response length: {len(answer)} characters")

            # Log completion
            if state.get("status_logger"):
                state["status_logger"].log_step(
                    step_name="Synthesize Analytical Response",
                    status=StepStatus.COMPLETED,
                    details="Generated comprehensive analytical response with narrative and data tables",
                    metadata={
                        "response_length": len(answer),
                        "queries_synthesized": len(successful_queries),
                        "failed_queries_count": len(failed_queries)
                    }
                )

        except Exception as e:
            logger.error(f"Error synthesizing response: {e}")

            # Fallback: return raw data if synthesis fails
            fallback_answer = f"## Query Results\n\n"
            fallback_answer += f"I retrieved data for your question but encountered an error synthesizing the response.\n\n"

            for result_info in formatted_results:
                fallback_answer += f"### {result_info['description']}\n\n"
                fallback_answer += f"Found {result_info['row_count']} rows\n\n"

                if result_info['data']:
                    # Simple table formatting
                    first_row = result_info['data'][0]
                    headers = list(first_row.keys())
                    fallback_answer += "| " + " | ".join(headers) + " |\n"
                    fallback_answer += "| " + " | ".join(["---"] * len(headers)) + " |\n"

                    for row in result_info['data'][:10]:
                        fallback_answer += "| " + " | ".join(str(row.get(h, "")) for h in headers) + " |\n"

                    fallback_answer += "\n"

            state["answer"] = fallback_answer
            state["error"] = f"Synthesis failed: {str(e)}"

            if state.get("status_logger"):
                state["status_logger"].log_step(
                    step_name="Synthesize Analytical Response",
                    status=StepStatus.FAILED,
                    details=f"Synthesis error: {str(e)}, using fallback format"
                )

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
        system_prompt = f"""You are a SQL expert. Convert the natural language question into a DuckDB SQL query.

Available tables and schemas:
{state['schema_context']}
{relationship_text}
{business_text}
{examples_text}
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

        # Generate SQL using interpreted question (with expanded acronyms)
        question_for_sql = state.get("interpreted_question") or state["question"]

        logger.info(f"📝 Question for SQL generation: {question_for_sql}")
        if state.get("detected_acronyms"):
            logger.info(f"🔄 Expanded acronyms: {', '.join(state['detected_acronyms'])}")

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=question_for_sql)  # Use interpreted question with expanded acronyms
            ]

            response = self.llm.invoke(messages)

            sql_query = response.content.strip()

            # Clean up SQL (remove markdown formatting if present)
            sql_query = sql_query.replace("```sql", "").replace("```", "").strip()

            state["sql_query"] = sql_query

            logger.info(f"✅ Generated SQL: {sql_query}")

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
        if state.get("is_denial_analysis", False):
            return "denial_analysis"
        return "normal_query"

    def route_after_interpretation(self, state: AgentState) -> str:
        """Route based on whether interpretation answered the question."""
        # Check if it's a definition question that was answered
        question_type = state.get("question_type")
        if question_type in ["kpi_definition", "terminology_definition"]:
            logger.info("📤 Returning definition answer directly (no SQL needed)")
            return "definition_answered"
        # Otherwise continue to schema loading and SQL generation
        return "continue"

    def route_after_clarification(self, state: AgentState) -> str:
        """Route based on whether clarification is needed."""
        if state.get("needs_clarification", False):
            return "clarify"  # Return clarification question to user
        return "proceed"  # Continue with SQL generation

    def route_query_strategy(self, state: AgentState) -> str:
        """Route between single-query and multi-query analytical strategy."""
        if state.get("is_analytical_query", False):
            logger.info("📊 Routing to MULTI-QUERY analytical path")
            return "plan_multi_query"
        logger.info("📝 Routing to SINGLE-QUERY path")
        return "generate_sql"

    def check_query_plan(self, state: AgentState) -> str:
        """Check if query planning succeeded."""
        if state.get("query_plan") and state.get("query_plan").get("sub_queries"):
            return "proceed"
        return "error"

    def check_multi_sql_generated(self, state: AgentState) -> str:
        """Check if at least one sub-query has SQL generated."""
        query_plan = state.get("query_plan", {})
        sub_queries = query_plan.get("sub_queries", [])

        # Check if at least one sub-query has SQL
        has_sql = any(sq.get("sql_query") for sq in sub_queries)

        if has_sql:
            return "proceed"
        return "error"

    def check_multi_execution(self, state: AgentState) -> str:
        """Check if at least one query executed successfully."""
        if state.get("execution_success", False):
            return "synthesize"
        return "error"

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
        """
        Format chat history for context.

        Now includes FULL conversation history for ChatGPT-level context understanding.
        Uses smart truncation only if conversation becomes extremely long (>50 messages).
        """
        if not chat_history:
            return "No previous conversation"

        # Use FULL chat history for best context (no arbitrary 10-message limit!)
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
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "user":
                formatted.append(f"User: {content}")
            elif role == "assistant":
                formatted.append(f"Assistant: {content}")
            elif role == "system":
                formatted.append(f"[{content}]")

        return "\n".join(formatted) if formatted else "No previous conversation"

    # ===== Public interface =====

    def ask(
        self,
        question: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        status_callback: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Ask a natural language question.

        Args:
            question: Natural language question
            chat_history: Previous chat messages
            status_callback: Optional callback for real-time status updates

        Returns:
            Dictionary with answer, metadata, and reasoning_steps
        """
        # Print to console for visibility
        print("\n" + "=" * 80)
        print("🚀 AR CHAT AGENT - Processing Question")
        print("=" * 80)
        print(f"Question: {question}")
        print("=" * 80 + "\n")

        from utils.status_logger import StatusLogger

        # Create StatusLogger with callback
        status_logger = StatusLogger(callback=status_callback)

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
            "company_name": None,
            "status_logger": status_logger  # Add StatusLogger to state
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
                status_summary = status_logger.get_summary()
                return {
                    "answer": "I don't understand your question. Please try to be more specific.",
                    "metadata": {
                        "error": "Recursion limit reached - query generation failed after multiple retries",
                        "sql_query": initial_state.get("sql_query"),
                        "retry_count": initial_state.get("retry_count", 0)
                    },
                    "sql_query": initial_state.get("sql_query"),
                    "success": False,
                    "reasoning_steps": status_summary["steps"],
                    "total_time": status_summary["total_time_seconds"]
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

        # Get status summary
        status_summary = status_logger.get_summary()

        # Print completion message
        print("\n" + "=" * 80)
        print("✅ AR CHAT AGENT - Processing Complete")
        print("=" * 80)
        print(f"Total Time: {status_summary['total_time_seconds']:.2f}s")
        print(f"Total Steps: {status_summary['total_steps']}")
        print(f"Success: {final_state.get('execution_success', False)}")
        print("=" * 80 + "\n")

        return {
            "answer": answer,
            "metadata": final_state.get("metadata", {}),
            "sql_query": final_state.get("sql_query"),
            "success": final_state.get("execution_success", False),
            "reasoning_steps": status_summary["steps"],  # Include reasoning
            "total_time": status_summary["total_time_seconds"]
        }


# Example CLI entry point removed; this module is used by the FastAPI app.
