"""
Agent Logging Enhancement
Provides enhanced logging wrapper functions for agent nodes.
Import and use these to add detailed thought process logging to any agent.
"""
from typing import Dict, Any
from utils.status_logger import StatusLogger, StepStatus


def log_detect_intent(status_logger: StatusLogger, question: str, intent_result: Dict[str, Any]):
    """Log intent detection step."""
    reasoning = [
        f"Analyzing question: '{question}'",
        f"Checking for special commands (/weekly, /denial, etc.)",
    ]

    if intent_result.get("is_weekly_report"):
        reasoning.append("✓ Detected: Weekly Report Request")
        reasoning.append(f"  Company: {intent_result.get('company_name', 'default')}")
        status_logger.log_step(
            step_name="Detect Intent",
            status=StepStatus.COMPLETED,
            details="Identified as weekly report request - routing to specialized tool",
            reasoning=reasoning,
            metadata=intent_result
        )
    elif intent_result.get("is_denial_analysis"):
        reasoning.append("✓ Detected: Denial Analysis Request")
        status_logger.log_step(
            step_name="Detect Intent",
            status=StepStatus.COMPLETED,
            details="Identified as denial analysis request - routing to specialized tool",
            reasoning=reasoning,
            metadata=intent_result
        )
    else:
        reasoning.append("✓ Detected: Normal Data Query")
        reasoning.append("  Will proceed with SQL generation workflow")
        status_logger.log_step(
            step_name="Detect Intent",
            status=StepStatus.COMPLETED,
            details="Identified as standard data query - will generate SQL",
            reasoning=reasoning
        )


def log_interpret_question(status_logger: StatusLogger, interpretation: Dict[str, Any]):
    """Log question interpretation step."""
    reasoning = interpretation.get("reasoning", [])
    question_type = interpretation.get("question_type", "unknown")
    detected_acronyms = interpretation.get("detected_acronyms", [])

    details = f"Question type: {question_type}"
    if detected_acronyms:
        details += f" | Acronyms found: {', '.join(detected_acronyms)}"

    status_logger.log_step(
        step_name="Interpret Question",
        status=StepStatus.COMPLETED,
        details=details,
        reasoning=reasoning,
        metadata={
            "question_type": question_type,
            "detected_acronyms": detected_acronyms,
            "interpreted_question": interpretation.get("interpreted_question")
        }
    )


def log_load_schema(status_logger: StatusLogger, tables: list, schema_context: str):
    """Log schema loading step."""
    reasoning = [
        f"Found {len(tables)} tables in database",
        "Extracting column information for each table",
        "Building schema context for SQL generation"
    ]

    # Show first few tables
    if tables:
        table_list = ", ".join(tables[:5])
        if len(tables) > 5:
            table_list += f", ... ({len(tables)-5} more)"
        reasoning.append(f"Tables: {table_list}")

    status_logger.log_step(
        step_name="Load Schema",
        status=StepStatus.COMPLETED,
        details=f"Loaded schema information for {len(tables)} tables",
        reasoning=reasoning,
        metadata={
            "table_count": len(tables),
            "tables": tables
        }
    )


def log_check_clarification(status_logger: StatusLogger, needs_clarification: bool, clarification_data: Dict[str, Any]):
    """Log clarification check step."""
    if needs_clarification:
        reasoning = [
            "❓ Question is ambiguous - clarification needed",
            f"Ambiguity type: {clarification_data.get('ambiguity_type')}",
            "Asking user to provide more details"
        ]

        status_logger.log_step(
            step_name="Check Clarification",
            status=StepStatus.COMPLETED,
            details="Question needs clarification - asking follow-up question",
            reasoning=reasoning,
            metadata=clarification_data
        )
    else:
        reasoning = [
            "✓ Question is clear and unambiguous",
            "No clarification needed",
            "Proceeding to SQL generation"
        ]

        status_logger.log_step(
            step_name="Check Clarification",
            status=StepStatus.COMPLETED,
            details="No clarification needed - question is clear",
            reasoning=reasoning
        )


def log_generate_sql(status_logger: StatusLogger, question: str, sql_query: str, examples_used: int = 0, is_retry: bool = False):
    """Log SQL generation step."""
    reasoning = [
        f"Converting question to SQL query",
        f"Using schema context and business rules",
    ]

    if examples_used > 0:
        reasoning.append(f"✓ Found {examples_used} relevant example queries to learn from")

    if is_retry:
        reasoning.append("⚠️  This is a RETRY after previous error")
        reasoning.append("   Incorporating error feedback to fix the query")

    reasoning.append(f"Generated SQL: {sql_query[:100]}{'...' if len(sql_query) > 100 else ''}")

    status = StepStatus.COMPLETED if sql_query else StepStatus.FAILED

    status_logger.log_step(
        step_name="Generate SQL",
        status=status,
        details="SQL query generated from natural language" + (" (retry)" if is_retry else ""),
        reasoning=reasoning,
        metadata={
            "sql_query": sql_query,
            "is_retry": is_retry,
            "examples_used": examples_used
        }
    )


def log_validate_sql(status_logger: StatusLogger, sql_query: str, validation_result: Dict[str, Any]):
    """Log SQL validation step."""
    is_valid = validation_result.get("valid", False)

    reasoning = [
        "Checking SQL for security and correctness",
        "• Must be SELECT query only (no modifications)",
        "• No access to sensitive columns",
        "• Proper syntax and structure"
    ]

    if is_valid:
        reasoning.append("✓ All validation checks passed")
        if validation_result.get("warnings"):
            reasoning.append(f"⚠️  Warnings: {len(validation_result['warnings'])}")
            for warning in validation_result["warnings"]:
                reasoning.append(f"   - {warning}")

        status_logger.log_step(
            step_name="Validate SQL",
            status=StepStatus.COMPLETED,
            details="SQL query passed all validation checks",
            reasoning=reasoning,
            metadata=validation_result
        )
    else:
        error = validation_result.get("error", "Unknown validation error")
        reasoning.append(f"❌ Validation failed: {error}")

        status_logger.log_step(
            step_name="Validate SQL",
            status=StepStatus.FAILED,
            details=f"SQL validation failed - will retry",
            reasoning=reasoning,
            metadata=validation_result
        )


def log_execute_query(status_logger: StatusLogger, sql_query: str, result: Dict[str, Any]):
    """Log query execution step."""
    success = result.get("success", False)

    reasoning = [
        f"Executing SQL query against database",
        f"Query: {sql_query[:150]}{'...' if len(sql_query) > 150 else ''}",
    ]

    if success:
        row_count = result.get("row_count", 0)
        exec_time = result.get("execution_time_seconds", 0)

        reasoning.append(f"✓ Query executed successfully")
        reasoning.append(f"   Rows returned: {row_count:,}")
        reasoning.append(f"   Execution time: {exec_time:.3f}s")

        status_logger.log_step(
            step_name="Execute Query",
            status=StepStatus.COMPLETED,
            details=f"Query returned {row_count:,} rows in {exec_time:.3f}s",
            reasoning=reasoning,
            metadata={
                "row_count": row_count,
                "execution_time_seconds": exec_time,
                "column_names": result.get("column_names", [])
            }
        )
    else:
        error = result.get("error", "Unknown execution error")
        reasoning.append(f"❌ Query execution failed: {error}")

        status_logger.log_step(
            step_name="Execute Query",
            status=StepStatus.FAILED,
            details="Query execution failed - will retry with fixes",
            reasoning=reasoning,
            metadata={"error": error}
        )


def log_format_response(status_logger: StatusLogger, row_count: int):
    """Log response formatting step."""
    reasoning = [
        f"Converting {row_count} rows of data into natural language answer",
        "Using LLM to generate human-readable summary",
        "Including key insights and patterns from the data"
    ]

    status_logger.log_step(
        step_name="Format Response",
        status=StepStatus.COMPLETED,
        details=f"Generated natural language answer from {row_count} rows",
        reasoning=reasoning,
        metadata={"row_count": row_count}
    )


def log_error(status_logger: StatusLogger, error_msg: str, error_type: str = "general"):
    """Log error handling step."""
    reasoning = [
        f"Error occurred: {error_msg}",
        "Generating user-friendly error message",
        "Technical details logged for debugging"
    ]

    status_logger.log_step(
        step_name="Handle Error",
        status=StepStatus.FAILED,
        details="An error occurred - returning user-friendly message",
        reasoning=reasoning,
        metadata={"error": error_msg, "error_type": error_type}
    )


def log_decision(status_logger: StatusLogger, decision_point: str, chosen_path: str, available_paths: list, reason: str):
    """Log a routing/decision point."""
    status_logger.log_decision(
        decision=decision_point,
        options=available_paths,
        chosen=chosen_path,
        reasoning=reason
    )
