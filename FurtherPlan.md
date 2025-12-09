# AI Agent Improvement Plan - TrueBilling Backend

## Executive Summary

This document outlines a comprehensive strategy to significantly improve the AI agent's ability to understand and respond to natural language queries about healthcare billing data. The plan addresses current limitations and proposes concrete enhancements across 5 key areas.

---

## Current State Analysis

### What Works Well ✅
- **Robust error recovery**: 2-level retry mechanism with error-specific guidance
- **Strong security**: SELECT-only enforcement, forbidden keyword blocking
- **Dual database support**: Separate optimized agents for DuckDB (uploaded files) and MySQL (live databases)
- **Chat history context**: Last 5-10 messages for conversation continuity
- **Database-specific prompts**: Tailored SQL generation rules for DuckDB vs MySQL

### Critical Gaps Identified ❌

1. **Insufficient Schema Context**: Schema types provided, but no value ranges, relationships, or business semantics
2. **No Query Learning**: Each query starts fresh without leveraging successful patterns
3. **Poor Error UX**: Generic "I don't understand" messages instead of actionable guidance
4. **Limited Relationship Understanding**: No foreign key or join path information
5. **No Data Statistics**: Missing min/max values, cardinality, distributions
6. **Ambiguity Handling**: Doesn't ask clarifying questions
7. **No Confidence Scoring**: Users can't assess answer reliability

---

## Improvement Strategy - 5 Pillars

### 📊 PILLAR 1: Enhanced Schema & Metadata Context

#### 1.1 SQL Schema File Integration ⭐ **HIGH PRIORITY**

**What to Provide:**
```sql
-- Generate complete DDL for all tables
CREATE TABLE visits (
    visit_id VARCHAR(50) PRIMARY KEY,
    patient_id VARCHAR(50) NOT NULL,
    visit_date DATE NOT NULL,
    visit_status VARCHAR(50) DEFAULT 'Pending',
    charge DECIMAL(10,2),
    balance DECIMAL(10,2),
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id),
    INDEX idx_visit_date (visit_date),
    INDEX idx_status (visit_status)
);

-- Include constraints, indexes, foreign keys, defaults
```

**Implementation:**
1. **For MySQL**: Extract DDL using:
   ```sql
   SHOW CREATE TABLE table_name;
   ```
2. **For DuckDB**: Generate DDL from Parquet schema + add semantic relationships manually
3. **Parse DDL** to extract:
   - Primary keys
   - Foreign keys (join paths!)
   - Indexes (performance hints)
   - Default values (common values)
   - NOT NULL constraints (required fields)

**Benefits:**
- ✅ LLM understands table relationships → better join queries
- ✅ Index awareness → can suggest optimizations
- ✅ Constraint knowledge → validates user assumptions
- ✅ Default values → understands common states

**File Location:** `schemas/{database_name}_ddl.sql`

---

#### 1.2 ER Diagram Context ⭐ **HIGH PRIORITY**

**What to Generate:**
- **Visual ER Diagram** (PNG/SVG) for human review
- **Machine-readable relationship file** (JSON) for LLM context

**Relationship JSON Structure:**
```json
{
  "tables": {
    "visits": {
      "primary_key": "visit_id",
      "foreign_keys": {
        "patient_id": {
          "references": "patients.patient_id",
          "relationship": "many-to-one",
          "cardinality": "0..*"
        }
      },
      "referenced_by": [
        {
          "table": "payments",
          "column": "visit_id",
          "relationship": "one-to-many"
        }
      ]
    }
  },
  "common_join_paths": [
    {
      "description": "Get patient visits with payments",
      "path": "patients → visits → payments",
      "sql_template": "FROM patients p JOIN visits v ON p.patient_id = v.patient_id JOIN payments pm ON v.visit_id = pm.visit_id"
    }
  ]
}
```

**Implementation Options:**

**Option A: Manual ER Diagram Creation**
- Use tools: MySQL Workbench, dbdiagram.io, draw.io
- Export as JSON for LLM consumption

**Option B: Automatic Generation from DDL**
- Tool: `sql-metadata` Python library
- Tool: `eralchemy` for visual diagrams
- Script to parse FOREIGN KEY constraints

**Option C: Hybrid Approach (RECOMMENDED)**
1. Auto-generate from INFORMATION_SCHEMA
2. Manual refinement for semantic relationships (e.g., visit_status → status_codes lookup)
3. Add business rules as comments

**Benefits:**
- ✅ **Massive improvement in multi-table queries**
- ✅ LLM knows how to join tables correctly
- ✅ Reduces hallucination of non-existent relationships
- ✅ Enables complex analytical queries

**File Location:** `schemas/{database_name}_relationships.json`

---

#### 1.3 Data Profile Statistics ⭐ **CRITICAL - ALREADY EXISTS BUT NOT USED!**

**Current State:**
- You already have `schema_profiler.py` that generates:
  - Row counts
  - Column data types
  - Semantic hints (temporal, identifier, monetary)
  - **BUT THIS IS NOT PASSED TO THE LLM!**

**Enhancement Required:**
Extend schema profiler to include:

```python
{
  "table_name": "visits",
  "row_count": 15000,
  "columns": {
    "visit_status": {
      "type": "varchar",
      "semantic_type": "categorical",
      "unique_values": 12,
      "top_values": [
        {"value": "Claim Created", "count": 8500, "percentage": 56.7},
        {"value": "Pending", "count": 3200, "percentage": 21.3},
        {"value": "On Hold", "count": 1800, "percentage": 12.0}
      ],
      "null_count": 0,
      "null_percentage": 0.0
    },
    "charge": {
      "type": "decimal",
      "semantic_type": "monetary",
      "min": 0.00,
      "max": 15000.00,
      "avg": 250.50,
      "median": 180.00,
      "std_dev": 120.30,
      "null_count": 50,
      "null_percentage": 0.33
    },
    "visit_date": {
      "type": "date",
      "semantic_type": "temporal",
      "min": "2020-01-01",
      "max": "2024-12-31",
      "null_count": 0,
      "date_range_years": 5
    }
  }
}
```

**Pass to LLM as:**
```
Table: visits (15,000 rows)
  - visit_status (categorical): 12 unique values
    • Top values: "Claim Created" (57%), "Pending" (21%), "On Hold" (12%)
  - charge (monetary): Range $0 - $15,000, Avg $250, Median $180
  - visit_date (temporal): 2020-01-01 to 2024-12-31 (5 years)
```

**Benefits:**
- ✅ **LLM knows valid values** → no hallucinated status codes
- ✅ **Understands data ranges** → better WHERE clause conditions
- ✅ **Cardinality awareness** → optimizes GROUP BY queries
- ✅ **NULL handling** → knows which columns can be NULL

**Implementation:**
1. Update `schema_profiler.py` to generate extended stats
2. Modify agent prompts to include profile summary
3. Cache profiles (regenerate daily/weekly)

---

#### 1.4 Business Rules & Domain Knowledge

**What to Add:**
Create a `business_rules.json` file with healthcare billing domain knowledge:

```json
{
  "date_columns": {
    "visit_date": {
      "description": "Date when service was performed",
      "typical_use": "Filter by service date, group by month/year",
      "common_queries": [
        "visits this month",
        "revenue by quarter"
      ]
    },
    "transaction_date": {
      "description": "Date when payment was received (can be weeks/months after visit_date)",
      "typical_use": "Cash flow analysis, payment tracking",
      "common_queries": [
        "payments received this week",
        "collections by month"
      ]
    },
    "bill_date": {
      "description": "Date when claim was submitted to payer",
      "typical_use": "AR aging calculations",
      "common_queries": [
        "claims submitted last month"
      ]
    }
  },
  "kpi_definitions": {
    "Days in AR": {
      "formula": "CEILING(Ending AR / Average Daily Charges)",
      "required_columns": ["balance", "charge", "visit_date"],
      "description": "Measures how long it takes to collect payment"
    },
    "Gross Collection Rate": {
      "formula": "(Total Payments / Total Charges) × 100",
      "required_columns": ["total_payment", "charge"],
      "description": "Percentage of charges actually collected"
    }
  },
  "common_filters": {
    "current_year": "YEAR(visit_date) = YEAR(CURRENT_DATE)",
    "last_month": "visit_date >= DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month') AND visit_date < DATE_TRUNC('month', CURRENT_DATE)",
    "billed_claims": "visit_status = 'Claim Created'"
  },
  "value_constraints": {
    "visit_status": {
      "valid_values": [
        "Claim Created",
        "Pending",
        "On Hold",
        "Issues Pending",
        "Approved",
        "Rejected"
      ],
      "default": "Pending"
    }
  }
}
```

**Pass to LLM:**
```
Domain Knowledge:
- visit_date: Date of service (use for service date analysis)
- transaction_date: Payment received date (use for cash flow)
- Days in AR: CEILING(balance / (charges / period_days))

Common Patterns:
- Current year filter: YEAR(visit_date) = 2025
- Billed claims: visit_status = 'Claim Created'
```

**Benefits:**
- ✅ Disambiguates date columns
- ✅ Knows healthcare billing terminology
- ✅ Pre-built filter patterns
- ✅ Validates business logic

---

### 🧠 PILLAR 2: Query Intelligence & Learning

#### 2.1 Few-Shot Example Library ⭐ **HIGH PRIORITY**

**Problem:** LLM generates queries from scratch each time → high error rate

**Solution:** Provide 10-20 successful query examples in the prompt

**Example Library Structure:**
```json
{
  "examples": [
    {
      "question": "How many visits were there in 2024?",
      "sql": "SELECT COUNT(*) as visit_count FROM visits WHERE YEAR(visit_date) = 2024",
      "explanation": "Simple count with year filter",
      "difficulty": "easy"
    },
    {
      "question": "What is the total revenue by month for 2024?",
      "sql": "SELECT EXTRACT(MONTH FROM visit_date) as month, EXTRACT(YEAR FROM visit_date) as year, SUM(charge) as revenue FROM visits WHERE YEAR(visit_date) = 2024 GROUP BY EXTRACT(YEAR FROM visit_date), EXTRACT(MONTH FROM visit_date) ORDER BY year, month",
      "explanation": "DuckDB-specific: GROUP BY requires full EXTRACT expression, not alias",
      "difficulty": "medium",
      "common_errors": [
        "GROUP BY month, year (incorrect - can't use alias)",
        "WHERE month = 1 (incorrect - can't use SELECT alias in WHERE)"
      ]
    },
    {
      "question": "Show me patients with unpaid balances over $500",
      "sql": "SELECT patient_id, SUM(balance) as total_balance FROM visits GROUP BY patient_id HAVING SUM(balance) > 500 ORDER BY total_balance DESC",
      "explanation": "Use HAVING for aggregate filters, not WHERE",
      "difficulty": "medium"
    }
  ]
}
```

**Implementation:**
1. Create `query_examples_duckdb.json` and `query_examples_mysql.json`
2. Include 3-5 examples in system prompt (rotate based on query similarity)
3. Use vector similarity search to find most relevant examples (advanced)

**Prompt Structure:**
```
Here are some example queries to guide you:

Example 1: "How many visits in 2024?"
SQL: SELECT COUNT(*) FROM visits WHERE YEAR(visit_date) = 2024

Example 2: "Revenue by month"
SQL: SELECT EXTRACT(MONTH FROM visit_date) as month, SUM(charge) FROM visits GROUP BY EXTRACT(MONTH FROM visit_date)
Note: DuckDB requires full expression in GROUP BY, not alias

Now generate SQL for this question: [user_question]
```

**Benefits:**
- ✅ **50-70% reduction in query errors** (industry standard)
- ✅ Shows correct DuckDB vs MySQL syntax patterns
- ✅ Teaches common pitfalls (GROUP BY alias issue)

---

#### 2.2 Query History Cache with RAG ⭐ **MEDIUM PRIORITY**

**Problem:** No learning from past successful queries

**Solution:** Store successful queries and retrieve similar ones

**Architecture:**
```python
# Store in database or Redis
query_cache = {
    "session_id": "chat_123",
    "timestamp": "2025-01-15T10:30:00Z",
    "question": "How many visits last month?",
    "sql": "SELECT COUNT(*) FROM visits WHERE...",
    "success": true,
    "row_count": 1,
    "execution_time_ms": 45,
    "embedding": [0.123, 0.456, ...]  # Vector embedding for similarity search
}
```

**Retrieval Process:**
1. User asks: "Show me visits from December"
2. Embed question using OpenAI embeddings
3. Find top 3 similar cached queries (cosine similarity)
4. Include in prompt: "Similar past queries: [...]"

**Benefits:**
- ✅ Learns from user's specific dataset
- ✅ Improves over time automatically
- ✅ Personalizes to company terminology

**Implementation Options:**
- **Simple**: Store in PostgreSQL with `pgvector` extension
- **Advanced**: Use Pinecone, Weaviate, or ChromaDB

---

#### 2.3 Query Confidence Scoring ⭐ **LOW PRIORITY**

**Problem:** Users don't know if they can trust the answer

**Solution:** Add confidence score to responses

**Scoring Factors:**
```python
confidence_score = (
    0.3 * schema_coverage_score +      # How many schema elements matched?
    0.3 * validation_success_score +   # Did SQL validate cleanly?
    0.2 * execution_success_score +    # Did query run without errors?
    0.2 * result_count_score           # Did query return reasonable row count?
)

# Example:
# - All columns exist in schema: +30%
# - SQL validated without warnings: +30%
# - Executed in < 1 second: +20%
# - Returned 1-1000 rows (reasonable): +20%
# Total: 100% confidence
```

**User Experience:**
```
Answer: "You had 1,250 visits in November 2024."
Confidence: High (95%)

Answer: "I found 0 records matching your criteria."
Confidence: Medium (65%) - This might be correct, or the query might be wrong.
```

**Benefits:**
- ✅ Helps users spot incorrect answers
- ✅ Encourages follow-up questions on low confidence
- ✅ Builds trust in the system

---

### 🔧 PILLAR 3: New Agent Tools & Capabilities

#### 3.1 Schema Introspection Tool

**Purpose:** Let agent explore schema dynamically before generating SQL

**Tool Definition:**
```python
class SchemaIntrospectionTool:
    def get_column_values(self, table: str, column: str, limit: int = 10):
        """Get sample distinct values from a column"""
        # Returns: ["Claim Created", "Pending", "On Hold", ...]

    def check_column_exists(self, table: str, column: str):
        """Verify column exists before using in query"""
        # Returns: True/False

    def get_related_tables(self, table: str):
        """Find tables with foreign key relationships"""
        # Returns: [{"table": "payments", "via": "visit_id"}]

    def get_column_stats(self, table: str, column: str):
        """Get min/max/avg for numeric columns"""
        # Returns: {"min": 0, "max": 15000, "avg": 250}
```

**LLM Workflow:**
```
User: "Show me visits with status X"
Agent: [Calls get_column_values("visits", "visit_status")]
Agent: "Available statuses: Claim Created, Pending, On Hold, ..."
Agent: "Did you mean one of these?"
OR
Agent: [Generates SQL with correct status value]
```

**Benefits:**
- ✅ No more hallucinated column names
- ✅ Can validate user assumptions
- ✅ Enables data exploration

---

#### 3.2 Query Explanation Tool

**Purpose:** Explain generated SQL to users

**Tool Definition:**
```python
class QueryExplainerTool:
    def explain_sql(self, sql: str):
        """Convert SQL to natural language explanation"""
        # Input: SELECT COUNT(*) FROM visits WHERE YEAR(visit_date) = 2024
        # Output: "Counting all visits where the visit year is 2024"
```

**Use Case:**
```
User: "How many visits in 2024?"
Agent SQL: SELECT COUNT(*) FROM visits WHERE YEAR(visit_date) = 2024
Agent Response:
  "You had 1,250 visits in 2024.

  (I counted all records in the visits table where the visit year equals 2024)"
```

**Benefits:**
- ✅ Transparency builds trust
- ✅ Users learn SQL patterns
- ✅ Helps spot incorrect queries

---

#### 3.3 Clarification Question Tool ⭐ **HIGH PRIORITY**

**Problem:** Agent fails silently instead of asking clarifying questions

**Solution:** Add decision logic to ask questions when ambiguous

**Ambiguity Detection:**
```python
ambiguity_patterns = {
    "date_range_missing": {
        "trigger": ["total", "sum", "count"] + no date filter,
        "question": "For what time period? (This month, this year, all time?)"
    },
    "multiple_date_columns": {
        "trigger": ["when", "date"] + multiple date columns in schema,
        "question": "Do you mean visit_date (service date) or transaction_date (payment date)?"
    },
    "unclear_metric": {
        "trigger": ["revenue"] + multiple money columns,
        "question": "Do you want gross charges or net collections?"
    }
}
```

**User Experience:**
```
User: "What's my total revenue?"
Agent: "I need to clarify:
  1. For what time period? (This month, this year, all time)
  2. Do you want gross charges or net collections?

  Please specify, or I can show you revenue for this year."
```

**Benefits:**
- ✅ **Massive improvement in user experience**
- ✅ Reduces failed queries
- ✅ Educates users on data structure

---

#### 3.4 Multi-Step Query Decomposer

**Problem:** Complex queries fail; simple queries succeed

**Solution:** Break complex questions into steps

**Example:**
```
User: "Show me patients with the highest unpaid balance who had visits in Q4 2024"

Agent Decomposition:
  Step 1: Get visits in Q4 2024
    → SELECT visit_id, patient_id FROM visits
       WHERE visit_date >= '2024-10-01' AND visit_date <= '2024-12-31'

  Step 2: Calculate unpaid balance per patient
    → SELECT patient_id, SUM(balance) as total_balance
       FROM visits WHERE visit_id IN (...)
       GROUP BY patient_id

  Step 3: Sort and limit
    → ORDER BY total_balance DESC LIMIT 10

Final Combined Query: [...]
```

**Benefits:**
- ✅ Handles complex analytical queries
- ✅ Can validate each step
- ✅ Better error recovery

---

### 📝 PILLAR 4: Prompt Engineering Enhancements

#### 4.1 Structured Prompt Template

**Current:** Long text block prompt
**Proposed:** Structured sections with clear hierarchy

```python
SYSTEM_PROMPT = """
# Role
You are an expert SQL query generator for healthcare billing data.

# Database Type
{database_type} (DuckDB/MySQL)

# Available Schema
{schema_context}

# Relationships
{relationship_context}

# Data Characteristics
{profile_statistics}

# SQL Generation Rules
1. CRITICAL - WHERE Clause:
   - NEVER use SELECT aliases in WHERE clause
   - Use full expressions: WHERE EXTRACT(YEAR FROM date_col) = 2024

2. CRITICAL - GROUP BY Clause ({database_specific_rules}):
   {group_by_rules}

3. Date Handling:
   {date_handling_rules}

# Few-Shot Examples
{relevant_examples}

# Business Context
{business_rules}

# Previous Conversation
{chat_history}

# Your Task
Generate a SQL query for: "{user_question}"

# Validation Context (if retry)
Previous attempt failed with error: {error_message}
Specific guidance: {error_specific_guidance}

# Output Format
Return ONLY the SQL query, no markdown or explanation.
"""
```

**Benefits:**
- ✅ Clear hierarchy
- ✅ Easier to maintain
- ✅ Can toggle sections on/off

---

#### 4.2 Error-Specific Retry Guidance

**Current:** Generic retry with same prompt
**Proposed:** Specialized prompts per error type

```python
ERROR_GUIDANCE = {
    "UNKNOWN COLUMN": """
The column '{column}' does not exist.
Available columns in {table}: {actual_columns}
Did you mean: {suggested_column}?
Regenerate the query using the correct column name.
""",

    "GROUP BY error": """
You used SELECT alias '{alias}' in GROUP BY, which is not allowed in {database}.
Replace GROUP BY {alias} with GROUP BY {full_expression}
Example: GROUP BY EXTRACT(YEAR FROM date_col) instead of GROUP BY year
""",

    "AGGREGATE function": """
You're mixing aggregated and non-aggregated columns.
All non-aggregated columns must be in GROUP BY.
Missing from GROUP BY: {missing_columns}
""",

    "Syntax error near": """
SQL syntax error near '{token}'.
Common issues:
- Missing comma in SELECT list
- Unclosed quote or parenthesis
- Reserved keyword used without quoting
Check your SQL syntax carefully.
"""
}
```

---

#### 4.3 Progressive Context Loading

**Problem:** Prompt too long → token limit exceeded

**Solution:** Load context progressively based on query complexity

```python
def build_prompt(question, complexity_level):
    base_context = [
        "system_role",
        "basic_schema",
        "critical_rules"
    ]

    if complexity_level == "simple":
        # Just column names, no stats
        return base_context + ["minimal_schema"]

    elif complexity_level == "medium":
        # Add common examples, basic stats
        return base_context + ["full_schema", "top_5_examples"]

    elif complexity_level == "complex":
        # Full context
        return base_context + [
            "full_schema",
            "relationship_graph",
            "data_profiles",
            "relevant_examples",
            "business_rules"
        ]
```

**Complexity Detection:**
```python
def assess_complexity(question):
    score = 0
    if "join" in question.lower() or multiple table keywords: score += 2
    if contains_aggregate_terms: score += 1
    if contains_date_math: score += 1
    if contains_subquery_indicators: score += 2

    return "complex" if score >= 3 else "medium" if score >= 1 else "simple"
```

---

### 🎯 PILLAR 5: User Experience Improvements

#### 5.1 Better Error Messages

**Current:**
```
"I don't understand your question. Please be more specific."
```

**Proposed:**
```
"I had trouble with this query. Here's what went wrong:

❌ Problem: The column 'total_revenue' doesn't exist in the visits table.

💡 Did you mean one of these?
   - charge (total billed amount)
   - total_payment (amount collected)
   - balance (unpaid amount)

Please rephrase your question using the correct column name."
```

**Error Categories:**
1. **Schema Error**: Column/table doesn't exist → Suggest alternatives
2. **Syntax Error**: SQL generation failed → Show what's available
3. **Ambiguity Error**: Multiple interpretations → Ask for clarification
4. **Data Error**: Query returned unexpected results → Suggest refinements

---

#### 5.2 Suggested Follow-Up Questions

**After every answer, suggest 2-3 related questions:**

```
Answer: "You had 1,250 visits in November 2024."

📊 You might also want to know:
  • How does this compare to October?
  • What's the breakdown by visit status?
  • What's the total revenue for November?
```

**Implementation:**
```python
def suggest_followups(question, result):
    suggestions = []

    if "count" in question:
        suggestions.append("Break this down by {categorical_column}")

    if "total" in question or "sum" in question:
        suggestions.append("Show me the trend over time")

    if time_period_detected:
        suggestions.append("Compare to previous period")

    return suggestions[:3]
```

---

#### 5.3 Query History UI

**Feature:** Let users see and re-run past queries

**UI Display:**
```
Recent Queries:
  1. "How many visits in November?" → 1,250 visits
     [View SQL] [Re-run] [Modify]

  2. "Total revenue by month" → 12 rows returned
     [View SQL] [Re-run] [Modify]
```

**Backend Storage:**
```python
# Add to chat_messages table
{
    "chat_id": "abc123",
    "timestamp": "2025-01-15T10:30:00Z",
    "question": "How many visits?",
    "sql": "SELECT COUNT(*) FROM visits",
    "answer": "1,250 visits",
    "metadata": {
        "execution_time_ms": 45,
        "row_count": 1,
        "confidence": 0.95
    }
}
```

---

## Implementation Roadmap

### Phase 1: Quick Wins (1-2 weeks)

**Priority: Immediate Impact**

1. ✅ **Enable Data Profiles in Prompts** (CRITICAL)
   - File: `data_query_agent.py`, `database_query_agent.py`
   - Change: Pass profile statistics to LLM
   - Impact: 30-40% improvement in query accuracy

2. ✅ **Add Few-Shot Examples**
   - File: Create `query_examples_duckdb.json`, `query_examples_mysql.json`
   - Change: Include 5-10 examples in system prompt
   - Impact: 50-70% reduction in common errors

3. ✅ **Better Error Messages**
   - File: `data_query_agent.py` → `handle_error()` method
   - Change: Return specific error categories instead of generic message
   - Impact: Huge UX improvement

4. ✅ **SQL Schema File Integration**
   - Action: Generate DDL for all tables
   - File: `schemas/mysql_ddl.sql`
   - Change: Extract foreign keys, indexes from DDL
   - Impact: Better understanding of relationships

### Phase 2: Core Enhancements (3-4 weeks)

**Priority: Foundation Building**

5. ✅ **ER Diagram & Relationship Context**
   - Action: Create `schemas/relationships.json`
   - Change: Pass join paths to LLM
   - Impact: Enables multi-table queries

6. ✅ **Clarification Question System**
   - File: `data_query_agent.py` → Add `detect_ambiguity()` node
   - Change: Ask clarifying questions instead of failing
   - Impact: Major UX improvement

7. ✅ **Business Rules Context**
   - File: Create `business_rules.json`
   - Change: Add healthcare billing domain knowledge
   - Impact: Better terminology understanding

8. ✅ **Schema Introspection Tool**
   - File: `src/agents/tools/schema_tools.py` (new)
   - Change: Add dynamic schema exploration
   - Impact: Reduce hallucination

### Phase 3: Intelligence Layer (4-6 weeks)

**Priority: Learning & Optimization**

9. ✅ **Query History Cache with RAG**
   - Database: Add `query_cache` table
   - Tool: Integrate OpenAI embeddings + vector search
   - Impact: System learns from experience

10. ✅ **Query Confidence Scoring**
    - File: `data_query_agent.py` → Add `calculate_confidence()` method
    - Change: Return confidence score with every answer
    - Impact: Users trust answers more

11. ✅ **Multi-Step Query Decomposer**
    - File: `data_query_agent.py` → Add `decompose_query()` node
    - Change: Break complex queries into steps
    - Impact: Handle complex analytical queries

12. ✅ **Structured Prompt Template**
    - File: `src/agents/prompts/` (new directory)
    - Change: Modular, maintainable prompt system
    - Impact: Easier to optimize prompts

### Phase 4: Advanced Features (6-8 weeks)

**Priority: Polish & Scale**

13. ✅ **Query History UI**
    - Frontend: Add query history panel
    - Backend: API endpoint for history retrieval
    - Impact: Better user workflow

14. ✅ **Suggested Follow-Up Questions**
    - File: `data_query_agent.py` → Add `suggest_followups()` method
    - Change: Generate contextual suggestions
    - Impact: Guides users to insights

15. ✅ **Progressive Context Loading**
    - File: `data_query_agent.py` → Add complexity assessment
    - Change: Load context based on query complexity
    - Impact: Optimize token usage

16. ✅ **Query Explanation Tool**
    - File: `src/agents/tools/explainer_tool.py` (new)
    - Change: Auto-explain generated SQL
    - Impact: Transparency & trust

---

## Technical Implementation Details

### Data Profile Enhancement

**Current Code Location:**
```
src/data_processing/schema_profiler.py
```

**Required Changes:**

```python
# BEFORE (current)
def generate_schema_profile(df, table_name):
    return {
        "table_name": table_name,
        "row_count": len(df),
        "columns": {col: {"type": str(dtype)} for col, dtype in df.dtypes.items()}
    }

# AFTER (enhanced)
def generate_schema_profile(df, table_name):
    profile = {
        "table_name": table_name,
        "row_count": len(df),
        "columns": {}
    }

    for col in df.columns:
        col_profile = {
            "type": str(df[col].dtype),
            "null_count": int(df[col].isnull().sum()),
            "null_percentage": round(df[col].isnull().sum() / len(df) * 100, 2)
        }

        # For categorical columns
        if df[col].dtype == 'object' or df[col].nunique() < 20:
            col_profile["unique_values"] = int(df[col].nunique())
            value_counts = df[col].value_counts().head(10)
            col_profile["top_values"] = [
                {
                    "value": str(val),
                    "count": int(count),
                    "percentage": round(count / len(df) * 100, 2)
                }
                for val, count in value_counts.items()
            ]

        # For numeric columns
        elif pd.api.types.is_numeric_dtype(df[col]):
            col_profile["min"] = float(df[col].min())
            col_profile["max"] = float(df[col].max())
            col_profile["avg"] = float(df[col].mean())
            col_profile["median"] = float(df[col].median())
            col_profile["std_dev"] = float(df[col].std())

        # For date columns
        elif pd.api.types.is_datetime64_dtype(df[col]):
            col_profile["min"] = df[col].min().isoformat()
            col_profile["max"] = df[col].max().isoformat()
            col_profile["date_range_days"] = (df[col].max() - df[col].min()).days

        profile["columns"][col] = col_profile

    return profile
```

**Agent Prompt Integration:**

```python
# File: src/agents/data_query_agent.py

def load_schema(state):
    # ... existing code ...

    # NEW: Add profile statistics to schema context
    schema_context = f"Available tables:\n\n"

    for table in tables:
        schema_context += f"Table: {table['name']} ({table['row_count']:,} rows)\n"
        schema_context += "  Columns:\n"

        for col_name, col_info in table['columns'].items():
            # Base info
            schema_context += f"    - {col_name} ({col_info['type']})"

            # Add profile statistics
            if 'top_values' in col_info:
                top_vals = ", ".join([f"'{v['value']}' ({v['percentage']:.0f}%)"
                                      for v in col_info['top_values'][:3]])
                schema_context += f" - Common values: {top_vals}"

            elif 'min' in col_info and 'max' in col_info:
                schema_context += f" - Range: {col_info['min']:.2f} to {col_info['max']:.2f}, Avg: {col_info['avg']:.2f}"

            elif 'date_range_days' in col_info:
                schema_context += f" - Date range: {col_info['min']} to {col_info['max']}"

            if col_info['null_percentage'] > 0:
                schema_context += f" ({col_info['null_percentage']:.1f}% null)"

            schema_context += "\n"

    state["schema_context"] = schema_context
    return state
```

---

### Relationship JSON Generation Script

**Create new file:** `scripts/generate_relationships.py`

```python
import json
from src.data_processing.mysql_catalog import MySQLCatalog

async def generate_relationships(database_name):
    """
    Extract foreign key relationships from MySQL INFORMATION_SCHEMA
    """
    catalog = MySQLCatalog(database_name)

    # Query foreign keys
    query = """
    SELECT
        TABLE_NAME,
        COLUMN_NAME,
        REFERENCED_TABLE_NAME,
        REFERENCED_COLUMN_NAME
    FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
    WHERE REFERENCED_TABLE_NAME IS NOT NULL
    AND TABLE_SCHEMA = DATABASE()
    """

    result = await catalog.execute_query(query)

    relationships = {}

    for row in result['rows']:
        table = row['TABLE_NAME']
        if table not in relationships:
            relationships[table] = {
                "foreign_keys": {},
                "referenced_by": []
            }

        relationships[table]["foreign_keys"][row['COLUMN_NAME']] = {
            "references": f"{row['REFERENCED_TABLE_NAME']}.{row['REFERENCED_COLUMN_NAME']}",
            "relationship": "many-to-one"
        }

        # Add reverse relationship
        ref_table = row['REFERENCED_TABLE_NAME']
        if ref_table not in relationships:
            relationships[ref_table] = {
                "foreign_keys": {},
                "referenced_by": []
            }

        relationships[ref_table]["referenced_by"].append({
            "table": table,
            "column": row['COLUMN_NAME'],
            "relationship": "one-to-many"
        })

    # Save to file
    output = {
        "database": database_name,
        "tables": relationships,
        "common_join_paths": generate_join_paths(relationships)
    }

    with open(f"schemas/{database_name}_relationships.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"✅ Relationships saved to schemas/{database_name}_relationships.json")

def generate_join_paths(relationships):
    """Generate common 2-3 table join paths"""
    paths = []

    # Example: Find all 2-table join paths
    for table, info in relationships.items():
        for fk_col, fk_info in info.get("foreign_keys", {}).items():
            ref_table = fk_info["references"].split(".")[0]
            paths.append({
                "description": f"Join {table} with {ref_table}",
                "tables": [table, ref_table],
                "path": f"{table} → {ref_table}",
                "sql_template": f"FROM {table} JOIN {ref_table} ON {table}.{fk_col} = {fk_info['references']}"
            })

    return paths

# Run script
if __name__ == "__main__":
    import asyncio
    asyncio.run(generate_relationships("your_database_name"))
```

---

### Few-Shot Example Library

**Create file:** `schemas/query_examples_duckdb.json`

```json
{
  "version": "1.0",
  "database_type": "duckdb",
  "examples": [
    {
      "id": "simple_count",
      "question": "How many visits were there in 2024?",
      "sql": "SELECT COUNT(*) as visit_count FROM visits WHERE YEAR(visit_date) = 2024",
      "explanation": "Simple count with year filter using YEAR() function",
      "difficulty": "easy",
      "tags": ["count", "filter", "year"]
    },
    {
      "id": "group_by_month",
      "question": "Show me total charges by month for 2024",
      "sql": "SELECT EXTRACT(MONTH FROM visit_date) as month, EXTRACT(YEAR FROM visit_date) as year, SUM(charge) as total_charges FROM visits WHERE YEAR(visit_date) = 2024 GROUP BY EXTRACT(YEAR FROM visit_date), EXTRACT(MONTH FROM visit_date) ORDER BY year, month",
      "explanation": "CRITICAL: DuckDB requires full EXTRACT expression in GROUP BY, cannot use alias 'month' or 'year'",
      "difficulty": "medium",
      "tags": ["aggregate", "group_by", "date", "duckdb_specific"],
      "common_errors": [
        "GROUP BY month, year (WRONG - can't use SELECT alias)",
        "WHERE month = 1 (WRONG - can't use SELECT alias in WHERE)"
      ]
    },
    {
      "id": "top_patients",
      "question": "Show me the top 10 patients with the highest unpaid balance",
      "sql": "SELECT patient_id, SUM(balance) as total_balance FROM visits GROUP BY patient_id HAVING SUM(balance) > 0 ORDER BY total_balance DESC LIMIT 10",
      "explanation": "Use HAVING for filtering on aggregate functions (after GROUP BY), not WHERE",
      "difficulty": "medium",
      "tags": ["aggregate", "having", "limit"]
    },
    {
      "id": "join_example",
      "question": "Show me visits with their patient names",
      "sql": "SELECT v.visit_id, v.visit_date, p.patient_name FROM visits v JOIN patients p ON v.patient_id = p.patient_id LIMIT 100",
      "explanation": "Simple join between visits and patients tables using shared patient_id",
      "difficulty": "medium",
      "tags": ["join", "alias"]
    },
    {
      "id": "status_breakdown",
      "question": "What's the breakdown of visits by status?",
      "sql": "SELECT visit_status, COUNT(*) as count, ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as percentage FROM visits GROUP BY visit_status ORDER BY count DESC",
      "explanation": "Window function to calculate percentage of total",
      "difficulty": "hard",
      "tags": ["window_function", "percentage", "group_by"]
    }
  ]
}
```

**Integration in Agent:**

```python
# File: src/agents/data_query_agent.py

import json

# Load examples at startup
with open("schemas/query_examples_duckdb.json") as f:
    QUERY_EXAMPLES = json.load(f)["examples"]

def get_relevant_examples(question, n=3):
    """
    Simple keyword matching to find relevant examples
    (Can be enhanced with embeddings for semantic search)
    """
    question_lower = question.lower()

    # Score each example
    scored_examples = []
    for ex in QUERY_EXAMPLES:
        score = 0

        # Check if question words appear in example question or tags
        for word in question_lower.split():
            if word in ex["question"].lower():
                score += 2
            if word in " ".join(ex["tags"]):
                score += 1

        if score > 0:
            scored_examples.append((score, ex))

    # Return top N
    scored_examples.sort(reverse=True, key=lambda x: x[0])
    return [ex for score, ex in scored_examples[:n]]

def generate_sql(state):
    # ... existing code ...

    # NEW: Add relevant examples to prompt
    examples = get_relevant_examples(state["question"])

    example_text = "\n\nRelevant Example Queries:\n"
    for ex in examples:
        example_text += f"\nExample: {ex['question']}\n"
        example_text += f"SQL: {ex['sql']}\n"
        if ex.get("explanation"):
            example_text += f"Note: {ex['explanation']}\n"

    system_prompt = SYSTEM_PROMPT_TEMPLATE + example_text

    # ... rest of SQL generation ...
```

---

## Expected Impact Summary

| Enhancement | Implementation Effort | Expected Improvement | Priority |
|-------------|---------------------|---------------------|----------|
| **Enable Data Profiles** | 1 day | 30-40% better queries | ⭐⭐⭐⭐⭐ CRITICAL |
| **Few-Shot Examples** | 2-3 days | 50-70% fewer errors | ⭐⭐⭐⭐⭐ CRITICAL |
| **ER Diagram/Relationships** | 1 week | Multi-table queries work | ⭐⭐⭐⭐⭐ CRITICAL |
| **Better Error Messages** | 2 days | Huge UX improvement | ⭐⭐⭐⭐ HIGH |
| **Clarification Questions** | 1 week | Fewer failed queries | ⭐⭐⭐⭐ HIGH |
| **Business Rules** | 3-4 days | Better terminology | ⭐⭐⭐ MEDIUM |
| **Query History Cache** | 2 weeks | Learns over time | ⭐⭐⭐ MEDIUM |
| **Schema Introspection** | 1 week | Reduce hallucination | ⭐⭐⭐ MEDIUM |
| **Confidence Scoring** | 3-4 days | Builds trust | ⭐⭐ LOW |
| **Multi-Step Queries** | 2 weeks | Complex queries work | ⭐⭐ LOW |

---

## Estimated Overall Improvement

**After Phase 1 (Quick Wins):**
- Query success rate: 40% → 70% (+75% improvement)
- User satisfaction: Moderate → High

**After Phase 2 (Core Enhancements):**
- Query success rate: 70% → 85% (+21% improvement)
- Multi-table queries: Mostly fail → Mostly succeed

**After Phase 3 (Intelligence Layer):**
- Query success rate: 85% → 95% (+12% improvement)
- System learns from usage patterns

**After Phase 4 (Advanced Features):**
- User experience: Good → Excellent
- Query success rate: 95%+ maintained with better UX

---

## Cost Considerations

### Computational Costs
- **Profile generation**: One-time per table (cache daily/weekly)
- **Embedding generation**: $0.0001 per query (query cache RAG)
- **LLM token increase**: +20-30% tokens with full context
  - Mitigation: Progressive context loading

### Development Costs
- Phase 1: 1-2 weeks (2 developers)
- Phase 2: 3-4 weeks (2 developers)
- Phase 3: 4-6 weeks (2-3 developers)
- Phase 4: 6-8 weeks (2-3 developers)

**Total estimated effort:** 14-20 weeks (3.5-5 months)

---

## Next Steps

### Immediate Actions (This Week)

1. ✅ **Generate SQL DDL file**
   - Export all table schemas to `schemas/mysql_ddl.sql`
   - Command: `mysqldump --no-data --databases your_db > schemas/mysql_ddl.sql`

2. ✅ **Create relationship mapping**
   - Run `scripts/generate_relationships.py`
   - Review and add semantic relationships manually

3. ✅ **Enable data profiles in prompts**
   - Modify `src/agents/data_query_agent.py` to include profile stats
   - Test with 5-10 sample queries

4. ✅ **Create few-shot example library**
   - Document 10 successful queries in `schemas/query_examples_duckdb.json`
   - Include DuckDB-specific GROUP BY examples

5. ✅ **Test & measure baseline**
   - Create test suite of 20 queries (easy, medium, hard)
   - Measure current success rate
   - Compare before/after Phase 1

### Questions to Answer

Before proceeding, please confirm:

1. **ER Diagram**: Do you want auto-generated or manual creation? (Recommendation: Auto-generate, then refine)

2. **Embedding Service**: For query history RAG, which embedding provider?
   - OpenAI embeddings (easiest integration)
   - Open-source (Sentence Transformers)
   - Other?

3. **Context Priority**: Which enhancement should we implement FIRST?
   - Recommendation: Data profiles (biggest impact, lowest effort)

4. **Testing Dataset**: Can you provide 20-30 real user questions to test improvements?

---

## Conclusion

The current AI agent has a solid foundation but lacks critical context needed for reliable query generation. The proposed enhancements address the root causes:

**Root Causes:**
1. ❌ LLM doesn't know valid column values → Hallucination
2. ❌ LLM doesn't know table relationships → Failed joins
3. ❌ LLM doesn't learn from successful queries → Repeated errors
4. ❌ Users don't know what's possible → Poor questions

**Solutions:**
1. ✅ Data profiles with value ranges
2. ✅ ER diagram with join paths
3. ✅ Query history cache with RAG
4. ✅ Clarification questions + better errors

**Expected Outcome:**
- Query success rate: 40% → 95%
- User satisfaction: Moderate → Excellent
- Multi-table queries: Mostly fail → Mostly succeed

The roadmap prioritizes quick wins (Phase 1) that deliver immediate 75% improvement, followed by foundational enhancements that enable complex queries.

**Ready to start with Phase 1?**
