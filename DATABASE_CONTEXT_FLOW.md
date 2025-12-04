# Database Context Flow Documentation

This document explains how database schema context is provided to the database query agent.

## Overview

The database agent receives comprehensive schema information through a multi-layered approach:
1. **Embedded Schema** - Pre-extracted schema information stored in Python constants
2. **MySQL Catalog** - Dynamically loads table schemas and enhances them with embedded data
3. **Agent System Prompt** - Combines schema context with business rules and patterns

---

## 1. Embedded Schema (`src/data_processing/embedded_schema.py`)

### Purpose
Contains pre-extracted database schema information that doesn't require runtime database queries.

### Contents

#### Schema Structure
- **FOREIGN_KEYS**: List of foreign key relationships
  - Includes both explicit FKs from SQL structure and inferred FKs from KPI queries
  - Example: `patient_visit_cpt.visit_id -> patient_visit.id`

- **TABLE_RELATIONSHIPS**: Dict mapping tables to related tables
  - Example: `patient_visit -> ["patient_visit_cpt", "visit_status"]`

- **COLUMN_COMMENTS**: Dict of (table, column) -> comment
- **TABLE_COMMENTS**: Dict of table -> comment
- **INFERRED_RELATIONSHIPS**: Relationships inferred from views

#### Data Patterns
- **SAMPLE_VALUES**: Sample data values for key columns (by table and column index)
- **DATA_PATTERNS**: Common data patterns, value ranges, formats

#### Query Patterns (from KPI queries)
- **COMMON_JOINS**: Common join patterns per table
  ```python
  "patient_visit": [
    {"target_table": "patient_visit_cpt", "condition": "pv.id = pvc.visit_id AND ISNULL(pvc.deleted_at)"}
  ]
  ```

- **COMMON_FILTERS**: Standard WHERE clause filters per table
  ```python
  "patient_visit": ["deleted_at IS NULL", "status != 12"]
  ```

- **COMMON_CALCULATIONS**: Standard calculation formulas
  ```python
  ["Charges: SUM(COALESCE(pvc.unit, 0) * COALESCE(pvc.cpt_amount, 0))"]
  ```

- **COMMON_EXCLUSIONS**: Standard exclusion patterns
  ```python
  {"cpt_codes_exclude": "NOT IN (\"00001\", \"00002\", \"00003\", \"90221\", \"90222\")"}
  ```

- **COMMON_DATE_FILTERS**: Standard date filtering patterns
- **JOIN_CHAINS**: Common table join sequences
- **TABLE_USAGE_FREQUENCY**: Frequency of table usage in queries

---

## 2. MySQL Catalog (`src/data_processing/mysql_catalog.py`)

### Purpose
Manages MySQL connections and provides enhanced schema context by combining:
- Runtime schema information from `INFORMATION_SCHEMA`
- Embedded schema constants from `embedded_schema.py`

### Key Methods

#### `_load_schema_cache()`
- Queries `INFORMATION_SCHEMA.COLUMNS` to get table/column definitions
- Caches schema information (columns, types, nullable, keys, defaults)
- Called automatically on first query execution

#### `get_enhanced_schema_context(table_name: str) -> str`
Builds a formatted schema description for a table including:

1. **Table Name**
2. **Table Comment** (from embedded schema)
3. **Foreign Keys** (from embedded schema)
   - Format: `column -> references_table.references_column`
4. **Related Tables** (from embedded schema)
5. **Common Joins** (from embedded schema, per table)
   - Format: `target_table: join_condition`
6. **Common Filters** (from embedded schema, per table)
7. **Columns** (from INFORMATION_SCHEMA cache)
   - Column name, type, nullable, key type (PRI/MUL/UNI)
   - Column comments (from embedded schema)
   - Sample values for key columns (id, status, type, code, *_id)

### Example Output
```
Table: patient_visit
  Foreign Keys:
    - status -> visit_status.id
  Related Tables: patient_visit_cpt, visit_status
  Common Joins:
    - patient_visit_cpt: pv.id = pvc.visit_id AND ISNULL(pvc.deleted_at)
    - visit_status: pv.status = vs.id
  Common Filters: deleted_at IS NULL, status != 12
  Columns:
    - id (int(11)) - PRIMARY KEY - NOT NULL [Examples: 1, 2, 3]
    - visit_date (date) - NOT NULL
    - status (int(11)) - NOT NULL [Examples: 1, 2, 3]
    - deleted_at (datetime) - NULL
    ...
```

---

## 3. Database Query Agent (`src/agents/database_query_agent.py`)

### Purpose
LangGraph agent that converts natural language to SQL queries.

### Flow

#### Step 1: Load Schema (`load_schema_node`)
```python
for table_name in tables:
    schema_text = catalog.get_enhanced_schema_context(table_name)
    schema_parts.append(schema_text)

state["schema_context"] = "\n".join(schema_parts)
```

- Iterates through all tables
- Calls `get_enhanced_schema_context()` for each table
- Combines all table schemas into a single `schema_context` string
- Stores in agent state

#### Step 2: Generate SQL (`generate_sql_node`)
Builds system prompt that includes:

1. **Schema Context** (from `state['schema_context']`)
   - All tables with their enhanced schema information
   - Foreign keys, relationships, common joins, filters per table

2. **Hardcoded Business Rules** (in system prompt)
   - Common calculations (from COMMON_CALCULATIONS)
   - Common exclusions (from COMMON_EXCLUSIONS)
   - Common date filters (from COMMON_DATE_FILTERS)
   - Join chains (from JOIN_CHAINS)

3. **MySQL-Specific Syntax Rules**
   - Date functions, WHERE clause rules, GROUP BY rules, JOIN syntax

4. **Error Guidance** (on retry)
   - Common MySQL errors and how to fix them

### System Prompt Structure
```
You are a MySQL SQL expert...

Available tables and schemas:
{schema_context}  <-- All tables with enhanced context

IMPORTANT RULES:
1. Table/Column Names: Use EXACT names from schema
2. Foreign Keys: Use FK relationships for JOINs
3. Related Tables: Use for identifying joinable tables
...

COMMON BUSINESS RULES (from KPI queries):
- Always exclude deleted records: ISNULL(table.deleted_at)
- Exclude cancelled visits: pv.status != 12
- Exclude test CPT codes: cc.cpt_code NOT IN (...)
- Common Calculations:
  * Charges: SUM(COALESCE(pvc.unit, 0) * COALESCE(pvc.cpt_amount, 0))
  ...
- Common Date Filters:
  * Current month: DATE(pv.visit_date) >= DATE_FORMAT(CURDATE(), '%Y-%m-01') ...
  ...
```

---

## Complete Flow Diagram

```
┌─────────────────────────────────────┐
│  embedded_schema.py                │
│  - FOREIGN_KEYS                     │
│  - TABLE_RELATIONSHIPS              │
│  - COMMON_JOINS                     │
│  - COMMON_FILTERS                   │
│  - COMMON_CALCULATIONS              │
│  - COMMON_EXCLUSIONS                │
│  - COMMON_DATE_FILTERS              │
│  - JOIN_CHAINS                      │
│  - SAMPLE_VALUES                    │
│  - COLUMN_COMMENTS                  │
└──────────────┬──────────────────────┘
               │ Import
               ▼
┌─────────────────────────────────────┐
│  mysql_catalog.py                   │
│                                      │
│  _load_schema_cache()               │
│  └─> Queries INFORMATION_SCHEMA     │
│      └─> Caches table/column info    │
│                                      │
│  get_enhanced_schema_context()      │
│  ├─> Gets columns from cache        │
│  ├─> Adds FOREIGN_KEYS              │
│  ├─> Adds TABLE_RELATIONSHIPS       │
│  ├─> Adds COMMON_JOINS (per table) │
│  ├─> Adds COMMON_FILTERS (per table)│
│  ├─> Adds column comments           │
│  └─> Adds sample values             │
└──────────────┬──────────────────────┘
               │ Called for each table
               ▼
┌─────────────────────────────────────┐
│  database_query_agent.py             │
│                                      │
│  load_schema_node()                  │
│  └─> For each table:                 │
│      └─> catalog.get_enhanced_       │
│          schema_context(table)       │
│      └─> Combine into               │
│          schema_context string      │
│                                      │
│  generate_sql_node()                 │
│  └─> Build system prompt:            │
│      ├─> Include schema_context      │
│      ├─> Include COMMON_CALCULATIONS │
│      ├─> Include COMMON_EXCLUSIONS   │
│      ├─> Include COMMON_DATE_FILTERS │
│      └─> Include JOIN_CHAINS         │
│      └─> Send to LLM                │
└─────────────────────────────────────┘
```

---

## Key Points

1. **Two Sources of Information**:
   - **Runtime**: Table/column definitions from `INFORMATION_SCHEMA`
   - **Embedded**: Relationships, patterns, business rules from `embedded_schema.py`

2. **Per-Table Context**:
   - Foreign keys, related tables, common joins, and filters are shown per table
   - Makes it easy for LLM to understand relationships for each table

3. **Global Patterns**:
   - Common calculations, exclusions, date filters, and join chains are shown globally
   - These apply across multiple tables/queries

4. **Sample Values**:
   - Key columns (id, status, type, code, *_id) show example values
   - Helps LLM understand data formats and common values

5. **Business Rules**:
   - Hardcoded in system prompt based on KPI query analysis
   - Ensures queries follow production patterns (exclude deleted, exclude test codes, etc.)

---

## Potential Improvements

1. **Consistency**: Currently COMMON_JOINS and COMMON_FILTERS are in schema context (per table), but COMMON_CALCULATIONS, COMMON_EXCLUSIONS, COMMON_DATE_FILTERS, and JOIN_CHAINS are hardcoded in the prompt. Consider moving all to schema context or all to prompt.

2. **Selective Loading**: Currently loads schema for ALL tables. Could optimize to only load relevant tables based on the question.

3. **Dynamic Updates**: Embedded schema is static. Could add mechanism to update it when database structure changes.

