"""
Schema parser - Extract table structure information from MySQL/PG/Oracle schema SQL files.

Features:
  - Parse CREATE TABLE statements, extract table names, column names, column types, primary keys, foreign keys
  - Support DDL parsing for MySQL, PostgreSQL, Oracle dialects
  - Build dependency graph between tables (topological sort)
  - Generate LLM-readable schema summary text
"""

import re
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class ColumnInfo:
    """Column information."""
    name: str
    data_type: str                     # Original type declaration (e.g., INT, VARCHAR(255), TEXT, DECIMAL(10,2))
    is_nullable: bool = True
    is_primary_key: bool = False
    is_auto_increment: bool = False
    is_unique: bool = False
    default_value: Optional[str] = None
    max_length: Optional[int] = None   # N in VARCHAR(N)
    allowed_values: Optional[List[str]] = None  # Allowed values list for ENUM/CHECK IN
    
    def base_type(self) -> str:
        """Get base type (without length declaration)."""
        return re.split(r'[\(\s]', self.data_type.upper())[0]

    def full_base_type(self) -> str:
        """Get full base type (without length declaration, preserving WITH and other modifiers).

        For example:
          TIMESTAMP WITH TIME ZONE(6)  -> TIMESTAMP WITH TIME ZONE
          VARCHAR(255)                 -> VARCHAR
          DECIMAL(10,2)                -> DECIMAL
          DOUBLE PRECISION             -> DOUBLE PRECISION
        """
        return re.split(r'\(', self.data_type.strip().upper())[0].strip()
    
    def is_string_type(self) -> bool:
        bt = self.base_type()
        return bt in ('VARCHAR', 'VARCHAR2', 'CHAR', 'TEXT', 'LONGTEXT', 'MEDIUMTEXT',
                       'TINYTEXT', 'CLOB', 'NVARCHAR', 'NCHAR', 'NCLOB')
    
    def is_numeric_type(self) -> bool:
        bt = self.base_type()
        if bt == 'NUMBER':
            return True
        return bt in ('INT', 'INTEGER', 'BIGINT', 'SMALLINT', 'TINYINT', 'MEDIUMINT',
                       'DECIMAL', 'NUMERIC', 'FLOAT', 'DOUBLE', 'REAL',
                       'BINARY_FLOAT', 'BINARY_DOUBLE')
    
    def is_integer_type(self) -> bool:
        bt = self.base_type()
        if bt == 'NUMBER':
            # NUMBER(N) without decimal places -> integer; NUMBER(N,M) M>0 -> decimal
            m = re.search(r'NUMBER\((\d+)(?:,\s*(\d+))?\)', self.data_type.upper())
            if m:
                decimal_places = m.group(2)
                return decimal_places is None or int(decimal_places) == 0
            return True  # Plain NUMBER treated as integer
        return bt in ('INT', 'INTEGER', 'BIGINT', 'SMALLINT', 'TINYINT', 'MEDIUMINT')
    
    def is_decimal_type(self) -> bool:
        bt = self.base_type()
        if bt == 'NUMBER':
            m = re.search(r'NUMBER\(\d+,\s*(\d+)\)', self.data_type.upper())
            if m and int(m.group(1)) > 0:
                return True
            return False
        return bt in ('DECIMAL', 'NUMERIC', 'FLOAT', 'DOUBLE', 'REAL',
                       'BINARY_FLOAT', 'BINARY_DOUBLE')
    
    def is_date_type(self) -> bool:
        bt = self.base_type()
        fbt = self.full_base_type()
        return bt in ('DATE', 'DATETIME', 'TIMESTAMP', 'TIME', 'YEAR',
                       'TIMESTAMPTZ') or \
               fbt in ('TIMESTAMP WITH TIME ZONE',
                       'TIMESTAMP WITH LOCAL TIME ZONE')
    
    def is_boolean_type(self) -> bool:
        bt = self.base_type()
        if bt == 'NUMBER':
            # Oracle: NUMBER(1) used as BOOL
            return 'NUMBER(1)' in self.data_type.upper()
        return bt in ('BOOLEAN', 'BOOL', 'TINYINT')  # MySQL: TINYINT(1) used as BOOL
    
    def is_json_type(self) -> bool:
        bt = self.base_type()
        return bt in ('JSON', 'JSONB')
    
    def is_enum_type(self) -> bool:
        """Whether it is an ENUM type (MySQL ENUM or PG CREATE TYPE ... AS ENUM or has allowed_values)."""
        bt = self.base_type()
        return bt == 'ENUM' or self.allowed_values is not None
    
    def is_clob_type(self) -> bool:
        """Oracle CLOB type (may be Oracle mapping of TEXT or JSON)."""
        bt = self.base_type()
        return bt in ('CLOB', 'NCLOB')
    
    def is_blob_type(self) -> bool:
        """Binary large object type (including fixed-length binary types like VARBINARY/BINARY)."""
        bt = self.base_type()
        return bt in ('BLOB', 'BYTEA', 'LONGBLOB', 'MEDIUMBLOB', 'TINYBLOB', 'VARBINARY', 'BINARY')


@dataclass
class ForeignKeyInfo:
    """Foreign key information."""
    constraint_name: Optional[str]
    columns: List[str]                  # Local columns
    ref_table: str                      # Referenced table name
    ref_columns: List[str]              # Referenced columns
    on_update: Optional[str] = None
    on_delete: Optional[str] = None


@dataclass
class TableInfo:
    """Table information."""
    name: str
    columns: List[ColumnInfo] = field(default_factory=list)
    primary_key: List[str] = field(default_factory=list)
    foreign_keys: List[ForeignKeyInfo] = field(default_factory=list)
    
    def get_column(self, col_name: str) -> Optional[ColumnInfo]:
        """Get column by name."""
        for col in self.columns:
            if col.name.lower() == col_name.lower():
                return col
        return None
    
    def string_columns(self) -> List[ColumnInfo]:
        return [c for c in self.columns if c.is_string_type()]
    
    def numeric_columns(self) -> List[ColumnInfo]:
        return [c for c in self.columns if c.is_numeric_type()]
    
    def integer_columns(self) -> List[ColumnInfo]:
        return [c for c in self.columns if c.is_integer_type()]
    
    def decimal_columns(self) -> List[ColumnInfo]:
        return [c for c in self.columns if c.is_decimal_type()]
    
    def date_columns(self) -> List[ColumnInfo]:
        return [c for c in self.columns if c.is_date_type()]
    
    def boolean_columns(self) -> List[ColumnInfo]:
        return [c for c in self.columns if c.is_boolean_type()]
    
    def json_columns(self) -> List[ColumnInfo]:
        return [c for c in self.columns if c.is_json_type()]
    
    def enum_columns(self) -> List[ColumnInfo]:
        return [c for c in self.columns if c.is_enum_type()]
    
    def nullable_columns(self) -> List[ColumnInfo]:
        return [c for c in self.columns if c.is_nullable and not c.is_primary_key]
    
    def has_self_reference(self) -> bool:
        """Whether it has self-referencing foreign key (tree structure)."""
        for fk in self.foreign_keys:
            if fk.ref_table.lower() == self.name.lower():
                return True
        return False
    
    def referenced_tables(self) -> List[str]:
        """Get other referenced table names."""
        return list(set(fk.ref_table for fk in self.foreign_keys 
                       if fk.ref_table.lower() != self.name.lower()))


@dataclass
class SchemaInfo:
    """Database schema information."""
    database_name: str
    tables: List[TableInfo] = field(default_factory=list)
    
    def get_table(self, table_name: str) -> Optional[TableInfo]:
        for t in self.tables:
            if t.name.lower() == table_name.lower():
                return t
        return None
    
    def table_names(self) -> List[str]:
        return [t.name for t in self.tables]
    
    def topological_order(self) -> List[str]:
        """
        Return topological sort order of tables (depended tables first).
        Used to determine data insertion order: insert referenced tables first, then referencing tables.
        """
        # Build adjacency list
        in_degree = {t.name: 0 for t in self.tables}
        graph = {t.name: [] for t in self.tables}
        
        for table in self.tables:
            for ref_table in table.referenced_tables():
                if ref_table in graph:
                    graph[ref_table].append(table.name)
                    in_degree[table.name] += 1
        
        # Kahn's algorithm
        queue = [t for t in in_degree if in_degree[t] == 0]
        result = []
        
        while queue:
            node = queue.pop(0)
            result.append(node)
            for neighbor in graph.get(node, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # If there are cycles, append remaining tables at the end
        remaining = [t.name for t in self.tables if t.name not in result]
        result.extend(remaining)
        
        return result
    
    def summary_for_llm(self) -> str:
        """
        Generate LLM-readable schema summary text.
        Includes table names, column names, column types, primary keys, foreign key info.
        """
        lines = [f"Database: {self.database_name}"]
        lines.append(f"Tables: {len(self.tables)}")
        lines.append("")
        
        for table in self.tables:
            lines.append(f"Table: {table.name}")
            lines.append(f"  Columns ({len(table.columns)}):")
            for col in table.columns:
                flags = []
                if col.is_primary_key:
                    flags.append("PK")
                if col.is_auto_increment:
                    flags.append("AUTO_INC")
                if not col.is_nullable:
                    flags.append("NOT NULL")
                if col.default_value:
                    flags.append(f"DEFAULT={col.default_value}")
                flag_str = f" [{', '.join(flags)}]" if flags else ""
                lines.append(f"    - {col.name}: {col.data_type}{flag_str}")
            
            if table.primary_key:
                lines.append(f"  Primary Key: ({', '.join(table.primary_key)})")
            
            if table.foreign_keys:
                lines.append(f"  Foreign Keys:")
                for fk in table.foreign_keys:
                    lines.append(f"    - ({', '.join(fk.columns)}) -> {fk.ref_table}({', '.join(fk.ref_columns)})")
            
            # Column type summary
            type_summary = []
            if table.string_columns():
                type_summary.append(f"STRING: {', '.join(c.name for c in table.string_columns())}")
            if table.integer_columns():
                type_summary.append(f"INTEGER: {', '.join(c.name for c in table.integer_columns())}")
            if table.decimal_columns():
                type_summary.append(f"DECIMAL: {', '.join(c.name for c in table.decimal_columns())}")
            if table.date_columns():
                type_summary.append(f"DATE: {', '.join(c.name for c in table.date_columns())}")
            if table.boolean_columns():
                type_summary.append(f"BOOLEAN: {', '.join(c.name for c in table.boolean_columns())}")
            if table.json_columns():
                type_summary.append(f"JSON: {', '.join(c.name for c in table.json_columns())}")
            if table.enum_columns():
                for ec in table.enum_columns():
                    vals = ec.allowed_values or []
                    type_summary.append(
                        f"ENUM: {ec.name} — allowed values: {vals}"
                    )
            if table.nullable_columns():
                type_summary.append(f"NULLABLE: {', '.join(c.name for c in table.nullable_columns())}")
            if table.has_self_reference():
                type_summary.append("SELF_REFERENCE (tree structure)")
            
            if type_summary:
                lines.append(f"  Column Types Summary:")
                for ts in type_summary:
                    lines.append(f"    {ts}")
            
            lines.append("")
        
        return "\n".join(lines)


@dataclass
class TripleSchemaInfo:
    """
    Tri-dialect Schema info - holds MySQL, PostgreSQL, Oracle DDL and parsed results simultaneously.

    During data synthesis, LLM needs to see schemas from multiple dialects simultaneously.
    InsertFormatter needs each dialect's type info for type-aware value formatting.
    """
    database_name: str
    mysql_schema: Optional[SchemaInfo] = None
    pg_schema: Optional[SchemaInfo] = None
    oracle_schema: Optional[SchemaInfo] = None
    mysql_ddl: str = ""       # Raw MySQL DDL text
    pg_ddl: str = ""          # Raw PostgreSQL DDL text
    oracle_ddl: str = ""      # Raw Oracle DDL text
    
    @property
    def tables(self) -> List[TableInfo]:
        """Compatibility interface: return MySQL schema's table list (priority)."""
        if self.mysql_schema:
            return self.mysql_schema.tables
        if self.pg_schema:
            return self.pg_schema.tables
        if self.oracle_schema:
            return self.oracle_schema.tables
        return []
    
    def get_table(self, table_name: str) -> Optional[TableInfo]:
        """Compatibility interface."""
        if self.mysql_schema:
            return self.mysql_schema.get_table(table_name)
        if self.pg_schema:
            return self.pg_schema.get_table(table_name)
        if self.oracle_schema:
            return self.oracle_schema.get_table(table_name)
        return None
    
    def table_names(self) -> List[str]:
        """Compatibility interface."""
        if self.mysql_schema:
            return self.mysql_schema.table_names()
        if self.pg_schema:
            return self.pg_schema.table_names()
        if self.oracle_schema:
            return self.oracle_schema.table_names()
        return []
    
    def topological_order(self) -> List[str]:
        """Compatibility interface."""
        if self.mysql_schema:
            return self.mysql_schema.topological_order()
        if self.pg_schema:
            return self.pg_schema.topological_order()
        if self.oracle_schema:
            return self.oracle_schema.topological_order()
        return []
    
    def summary_for_llm(self) -> str:
        """
        Generate LLM-readable tri-dialect schema summary.
        """
        lines = [f"Database: {self.database_name}"]
        
        # MySQL DDL
        lines.append("")
        lines.append("=" * 50)
        lines.append("MySQL DDL (raw CREATE TABLE statements)")
        lines.append("=" * 50)
        if self.mysql_ddl:
            lines.append(self.mysql_ddl.strip())
        else:
            lines.append("(MySQL DDL not available)")
        
        # PostgreSQL DDL
        lines.append("")
        lines.append("=" * 50)
        lines.append("PostgreSQL DDL (raw CREATE TABLE statements)")
        lines.append("=" * 50)
        if self.pg_ddl:
            lines.append(self.pg_ddl.strip())
        else:
            lines.append("(PostgreSQL DDL not available)")
        
        # Oracle DDL
        lines.append("")
        lines.append("=" * 50)
        lines.append("Oracle DDL (raw CREATE TABLE statements)")
        lines.append("=" * 50)
        if self.oracle_ddl:
            lines.append(self.oracle_ddl.strip())
        else:
            lines.append("(Oracle DDL not available)")
        
        # Parsed structured summary
        primary_schema = self.mysql_schema or self.pg_schema or self.oracle_schema
        if primary_schema:
            lines.append("")
            lines.append("=" * 50)
            lines.append("Parsed structured summary")
            lines.append("=" * 50)
            lines.append(f"Tables: {len(primary_schema.tables)}")
            lines.append("")
            
            for table in primary_schema.tables:
                lines.append(f"Table: {table.name}")
                
                # Column type summary
                type_summary = []
                if table.string_columns():
                    type_summary.append(f"STRING: {', '.join(c.name for c in table.string_columns())}")
                if table.integer_columns():
                    type_summary.append(f"INTEGER: {', '.join(c.name for c in table.integer_columns())}")
                if table.decimal_columns():
                    type_summary.append(f"DECIMAL: {', '.join(c.name for c in table.decimal_columns())}")
                if table.date_columns():
                    type_summary.append(f"DATE: {', '.join(c.name for c in table.date_columns())}")
                if table.boolean_columns():
                    type_summary.append(f"BOOLEAN: {', '.join(c.name for c in table.boolean_columns())}")
                if table.json_columns():
                    type_summary.append(f"JSON: {', '.join(c.name for c in table.json_columns())}")
                if table.enum_columns():
                    for ec in table.enum_columns():
                        vals = ec.allowed_values or []
                        type_summary.append(
                            f"ENUM: {ec.name} — allowed values: {vals}"
                        )
                if table.nullable_columns():
                    type_summary.append(f"NULLABLE: {', '.join(c.name for c in table.nullable_columns())}")
                if table.has_self_reference():
                    type_summary.append("SELF_REFERENCE (tree structure)")
                
                if type_summary:
                    for ts in type_summary:
                        lines.append(f"  {ts}")
                
                if table.foreign_keys:
                    for fk in table.foreign_keys:
                        lines.append(f"  FK: ({', '.join(fk.columns)}) -> {fk.ref_table}({', '.join(fk.ref_columns)})")
                
                lines.append("")
        
        return "\n".join(lines)


# Backward-compatible alias
DualSchemaInfo = TripleSchemaInfo


class SchemaParser:
    """
    Schema SQL file parser.

    Supports parsing CREATE TABLE statements for MySQL, PostgreSQL, and Oracle.
    """
    
    def __init__(self):
        pass
    
    def parse_file(self, schema_file: str, database_name: str = None) -> SchemaInfo:
        """
        Parse schema SQL file.

        Args:
            schema_file: SQL file path
            database_name: Database name (optional, inferred from file path by default)

        Returns:
            SchemaInfo object
        """
        if database_name is None:
            # Infer from path: database/books/books_schema_mysql.sql -> books
            database_name = Path(schema_file).parent.name
        
        with open(schema_file, 'r', encoding='utf-8') as f:
            sql_content = f.read()
        
        return self.parse_sql(sql_content, database_name)
    
    def parse_sql(self, sql_content: str, database_name: str) -> SchemaInfo:
        """
        Parse SQL text content.

        Args:
            sql_content: SQL text
            database_name: Database name

        Returns:
            SchemaInfo object
        """
        schema = SchemaInfo(database_name=database_name)
        
        # Preprocess: extract PG CREATE TYPE ... AS ENUM definitions
        # Mapping: {type_name_lower: [allowed_values]}
        enum_type_map: Dict[str, List[str]] = {}
        create_type_pattern = re.compile(
            r'CREATE\s+TYPE\s+[`"]?(\w+)[`"]?\s+AS\s+ENUM\s*\(([^)]+)\)',
            re.IGNORECASE
        )
        for ct_match in create_type_pattern.finditer(sql_content):
            type_name = ct_match.group(1).lower()
            values_str = ct_match.group(2)
            values = re.findall(r"'([^']*)'", values_str)
            if values:
                enum_type_map[type_name] = values
        
        # Extract all CREATE TABLE statements
        # Match pattern: CREATE TABLE table_name ( ... ) ...;
        pattern = re.compile(
            r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?'
            r'[`"]?(\w+)[`"]?\s*\((.*?)\)\s*'
            r'(?:ENGINE\s*=\s*\w+)?'
            r'(?:\s+DEFAULT\s+(?:CHARSET|CHARACTER\s+SET)\s*=?\s*\w+)?'
            r'(?:\s+CHARSET\s*=?\s*\w+)?'
            r'(?:\s+CHARACTER\s+SET\s*=?\s*\w+)?'
            r'(?:\s+COLLATE\s*=?\s*\w+)?'
            r'\s*;',
            re.IGNORECASE | re.DOTALL
        )
        
        for match in pattern.finditer(sql_content):
            table_name = match.group(1)
            body = match.group(2)
            
            table = self._parse_table_body(table_name, body)
            
            # Post-process: map PG ENUM TYPE to corresponding columns
            if enum_type_map:
                for col in table.columns:
                    col_type_lower = col.data_type.strip().lower()
                    if col_type_lower in enum_type_map:
                        col.allowed_values = enum_type_map[col_type_lower]
            
            schema.tables.append(table)
        
        return schema
    
    def _parse_table_body(self, table_name: str, body: str) -> TableInfo:
        """Parse the inner content of CREATE TABLE."""
        table = TableInfo(name=table_name)
        
        # Split by comma, but handle nested parentheses
        parts = self._split_by_comma(body)
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
            
            upper_part = part.upper().strip()
            
            # Determine if it's a constraint or column definition
            if upper_part.startswith('PRIMARY KEY'):
                # PRIMARY KEY (col1, col2)
                pk_cols = self._extract_column_list(part)
                table.primary_key = pk_cols
                for col_name in pk_cols:
                    col = table.get_column(col_name)
                    if col:
                        col.is_primary_key = True
                        col.is_nullable = False
            
            elif upper_part.startswith('FOREIGN KEY') or upper_part.startswith('CONSTRAINT'):
                # Check if it's a CONSTRAINT ... PRIMARY KEY form named primary key constraint
                if 'PRIMARY KEY' in upper_part and 'FOREIGN KEY' not in upper_part:
                    pk_cols = self._extract_column_list(part)
                    if pk_cols:
                        table.primary_key = pk_cols
                        for col_name in pk_cols:
                            col = table.get_column(col_name)
                            if col:
                                col.is_primary_key = True
                                col.is_nullable = False
                elif 'CHECK' in upper_part and 'FOREIGN KEY' not in upper_part:
                    # CONSTRAINT ... CHECK (col IN ('a', 'b', 'c'))
                    check_in_match = re.search(
                        r'CHECK\s*\(\s*[`"]?(\w+)[`"]?\s+IN\s*\(([^)]+)\)',
                        part, re.IGNORECASE
                    )
                    if check_in_match:
                        col_name = check_in_match.group(1)
                        values_str = check_in_match.group(2)
                        values = re.findall(r"'([^']*)'", values_str)
                        if values:
                            col = table.get_column(col_name)
                            if col:
                                col.allowed_values = values
                else:
                    fk = self._parse_foreign_key(part)
                    if fk:
                        table.foreign_keys.append(fk)
            
            elif upper_part.startswith('UNIQUE') or re.match(r'KEY[\s(]|KEY$', upper_part) or upper_part.startswith('INDEX'):
                # Table-level UNIQUE constraint: mark corresponding columns
                if upper_part.startswith('UNIQUE'):
                    unique_cols_match = re.search(r'\(([^)]+)\)', part)
                    if unique_cols_match:
                        unique_col_names = [
                            c.strip().strip('`"').lower()
                            for c in unique_cols_match.group(1).split(',')
                        ]
                        # Only mark single-column UNIQUE (multi-column composite UNIQUE not suitable for per-column dedup)
                        if len(unique_col_names) == 1:
                            for col in table.columns:
                                if col.name.lower() == unique_col_names[0]:
                                    col.is_unique = True
            
            elif re.match(r'CHECK[\s(]|CHECK$', upper_part):
                # CHECK constraint: parse CHECK(col IN ('a', 'b', 'c')) form enum constraint
                check_in_match = re.search(
                    r'CHECK\s*\(\s*[`"]?(\w+)[`"]?\s+IN\s*\(([^)]+)\)',
                    part, re.IGNORECASE
                )
                if check_in_match:
                    col_name = check_in_match.group(1)
                    values_str = check_in_match.group(2)
                    values = re.findall(r"'([^']*)'", values_str)
                    if values:
                        col = table.get_column(col_name)
                        if col:
                            col.allowed_values = values
            
            else:
                # Column definition
                col = self._parse_column(part)
                if col:
                    table.columns.append(col)
                    if col.is_primary_key:
                        if col.name not in table.primary_key:
                            table.primary_key.append(col.name)
        
        return table
    
    def _split_by_comma(self, text: str) -> List[str]:
        """Split by comma, handling nested parentheses, quotes, and commas inside comments.
        
        MySQL COMMENT '...' may contain commas, must skip commas inside quotes.
        SQL comments /* ... */ may contain commas and single quotes (e.g., Customer's), must skip comment content.
        """
        # Preprocess: remove /* ... */ block comments to avoid quotes inside comments interfering with parsing
        cleaned = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
        # Preprocess: remove -- ... line comments (to end of line)
        cleaned = re.sub(r'--[^\n]*', '', cleaned)
        
        parts = []
        depth = 0
        current = []
        in_quote = None  # Current quote character (' or "), None means not inside quotes
        i = 0
        
        while i < len(cleaned):
            char = cleaned[i]
            if in_quote:
                current.append(char)
                if char == '\\' and i + 1 < len(cleaned):
                    # Backslash escape (e.g., MySQL \'): skip next character
                    current.append(cleaned[i + 1])
                    i += 2
                    continue
                elif char == in_quote:
                    # Could be closing quote or doubled quote (e.g., SQL standard '' or "")
                    if i + 1 < len(cleaned) and cleaned[i + 1] == in_quote:
                        # Doubled quote escape
                        current.append(cleaned[i + 1])
                        i += 2
                        continue
                    else:
                        in_quote = None  # Close quote
            elif char in ("'", '"'):
                in_quote = char
                current.append(char)
            elif char == '(':
                depth += 1
                current.append(char)
            elif char == ')':
                depth -= 1
                current.append(char)
            elif char == ',' and depth == 0:
                parts.append(''.join(current))
                current = []
            else:
                current.append(char)
            i += 1
        
        if current:
            parts.append(''.join(current))
        
        return parts
    
    def _parse_column(self, col_def: str) -> Optional[ColumnInfo]:
        """Parse column definition."""
        col_def = col_def.strip()
        if not col_def:
            return None
        
        # Priority match: backtick/double-quote wrapped column names (supports spaces and special characters)
        # e.g., `Academic Year` TEXT NULL or "School Name" TEXT NULL
        # Type name may also be quoted: "frame_geometry" "frame_geometry_enum" NOT NULL
        quoted_pattern = re.compile(
            r'[`"]([^`"]+)[`"]\s+'
            r'(?:[`"]([^`"]+)[`"]|(\w+(?:\([^)]*\))?))'  # Type: quoted or unquoted
            r'(.*)',                  # Rest
            re.IGNORECASE
        )

        # Unquoted plain column name
        plain_pattern = re.compile(
            r'(\w+)\s+'
            r'(?:[`"]([^`"]+)[`"]|(\w+(?:\([^)]*\))?))'  # Type: quoted or unquoted
            r'(.*)',                  # Rest
            re.IGNORECASE
        )
        
        m = quoted_pattern.match(col_def)
        if not m:
            m = plain_pattern.match(col_def)
        if not m:
            return None
        
        name = m.group(1)
        # Type name: group(2) is quoted match, group(3) is unquoted match
        data_type = (m.group(2) or m.group(3) or '').strip()
        rest = m.group(4).strip()

        # Post-process: consume type modifiers from rest prefix, append to data_type
        # Supports: WITH TIME ZONE, WITH LOCAL TIME ZONE, []
        tz_match = re.match(r'\s*WITH\s+(LOCAL\s+)?TIME\s+ZONE\b', rest, re.IGNORECASE)
        if tz_match:
            data_type += ' ' + tz_match.group(0).strip()
            rest = rest[tz_match.end():].strip()

        arr_match = re.match(r'(\[\])+', rest)
        if arr_match:
            data_type += arr_match.group(0)
            rest = rest[arr_match.end():].strip()
        
        rest_upper = rest.upper()
        
        # Skip cases where SQL keywords are misidentified as column names
        if name.upper() in ('PRIMARY', 'FOREIGN', 'CONSTRAINT', 'UNIQUE', 'KEY', 
                            'INDEX', 'CHECK'):
            return None
        
        col = ColumnInfo(name=name, data_type=data_type)
        
        # Parse MySQL ENUM type: ENUM('val1', 'val2', ...)
        if data_type.upper().startswith('ENUM('):
            enum_match = re.findall(r"'([^']*)'", data_type)
            if enum_match:
                col.allowed_values = enum_match
        
        # Parse modifiers
        if 'NOT NULL' in rest_upper:
            col.is_nullable = False
        
        if 'AUTO_INCREMENT' in rest_upper or 'SERIAL' in rest_upper.replace('BIGSERIAL', '').replace('SMALLSERIAL', ''):
            col.is_auto_increment = True
        
        if 'BIGSERIAL' in rest_upper or 'SERIAL' in rest_upper:
            col.is_auto_increment = True
            col.data_type = 'INTEGER'  # SERIAL is essentially INTEGER + AUTO_INCREMENT
        
        if 'PRIMARY KEY' in rest_upper:
            col.is_primary_key = True
            col.is_nullable = False
        
        if 'UNIQUE' in rest_upper:
            col.is_unique = True
        
        # Parse max_length: VARCHAR(N), VARCHAR2(N), CHAR(N)
        length_match = re.search(r'(?:VARCHAR2?|CHAR)\((\d+)\)', data_type, re.IGNORECASE)
        if length_match:
            col.max_length = int(length_match.group(1))
        
        # Parse DEFAULT value
        default_match = re.search(r'DEFAULT\s+(\S+)', rest)
        if default_match:
            col.default_value = default_match.group(1)
            if col.default_value == 'NULL':
                col.is_nullable = True
        
        return col
    
    def _parse_foreign_key(self, fk_def: str) -> Optional[ForeignKeyInfo]:
        """Parse foreign key definition."""
        # CONSTRAINT name FOREIGN KEY (cols) REFERENCES table(cols) [ON UPDATE ...] [ON DELETE ...]
        # FOREIGN KEY (cols) REFERENCES table(cols) [ON UPDATE ...] [ON DELETE ...]
        
        constraint_name = None
        constraint_match = re.match(r'CONSTRAINT\s+[`"]?(\w+)[`"]?\s+', fk_def, re.IGNORECASE)
        if constraint_match:
            constraint_name = constraint_match.group(1)
        
        fk_match = re.search(
            r'FOREIGN\s+KEY\s*\(([^)]+)\)\s*REFERENCES\s+[`"]?(\w+)[`"]?\s*\(([^)]+)\)',
            fk_def,
            re.IGNORECASE
        )
        
        if not fk_match:
            return None
        
        columns = [c.strip().strip('`"') for c in fk_match.group(1).split(',')]
        ref_table = fk_match.group(2)
        ref_columns = [c.strip().strip('`"') for c in fk_match.group(3).split(',')]
        
        on_update = None
        on_delete = None
        
        on_update_match = re.search(r'ON\s+UPDATE\s+(\w+(?:\s+\w+)?)', fk_def, re.IGNORECASE)
        if on_update_match:
            on_update = on_update_match.group(1)
        
        on_delete_match = re.search(r'ON\s+DELETE\s+(\w+(?:\s+\w+)?)', fk_def, re.IGNORECASE)
        if on_delete_match:
            on_delete = on_delete_match.group(1)
        
        return ForeignKeyInfo(
            constraint_name=constraint_name,
            columns=columns,
            ref_table=ref_table,
            ref_columns=ref_columns,
            on_update=on_update,
            on_delete=on_delete,
        )
    
    def _extract_column_list(self, text: str) -> List[str]:
        """Extract column name list from PRIMARY KEY (col1, col2)."""
        m = re.search(r'\(([^)]+)\)', text)
        if not m:
            return []
        return [c.strip().strip('`"') for c in m.group(1).split(',')]


def parse_all_schemas(database_dir: str, dialect: str = 'mysql') -> Dict[str, SchemaInfo]:
    """
    Parse all database schema files (single dialect mode).

    Args:
        database_dir: Database directory (contains subdirectories, e.g., database/books/)
        dialect: 'mysql' or 'pg'

    Returns:
        {database_name: SchemaInfo}
    """
    parser = SchemaParser()
    schemas = {}
    
    database_path = Path(database_dir)
    
    for subdir in sorted(database_path.iterdir()):
        if not subdir.is_dir():
            continue
        
        db_name = subdir.name
        schema_file = subdir / f"{db_name}_schema_{dialect}.sql"
        
        if not schema_file.exists():
            continue
        
        try:
            schema = parser.parse_file(str(schema_file), db_name)
            schemas[db_name] = schema
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Failed to parse {db_name} schema: {e}")
    
    return schemas


def parse_all_dual_schemas(database_dir: str) -> Dict[str, TripleSchemaInfo]:
    """
    Parse all database MySQL + PostgreSQL + Oracle tri-dialect schema files.

    Reads both raw DDL text and parsed structured info simultaneously.

    Args:
        database_dir: Database directory (contains subdirectories, e.g., database/books/)

    Returns:
        {database_name: TripleSchemaInfo}
    """
    return parse_all_triple_schemas(database_dir)


def parse_all_triple_schemas(database_dir: str) -> Dict[str, TripleSchemaInfo]:
    """
    Parse all database MySQL + PostgreSQL + Oracle tri-dialect schema files.

    Reads both raw DDL text and parsed structured info simultaneously,
    used for data synthesis and INSERT formatting where each dialect's type info is needed.

    Args:
        database_dir: Database directory (contains subdirectories, e.g., database/dev/books/)

    Returns:
        {database_name: TripleSchemaInfo}
    """
    parser = SchemaParser()
    schemas = {}
    
    database_path = Path(database_dir)
    
    for subdir in sorted(database_path.iterdir()):
        if not subdir.is_dir():
            continue
        
        db_name = subdir.name
        mysql_file = subdir / f"{db_name}_schema_mysql.sql"
        pg_file = subdir / f"{db_name}_schema_pg.sql"
        oracle_file = subdir / f"{db_name}_schema_oracle.sql"
        
        # At least one schema file needed
        if not mysql_file.exists() and not pg_file.exists() and not oracle_file.exists():
            continue
        
        triple = TripleSchemaInfo(database_name=db_name)
        
        # Parse MySQL schema
        if mysql_file.exists():
            try:
                with open(mysql_file, 'r', encoding='utf-8') as f:
                    triple.mysql_ddl = f.read()
                triple.mysql_schema = parser.parse_sql(triple.mysql_ddl, db_name)
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"Failed to parse {db_name} MySQL schema: {e}")
        
        # Parse PostgreSQL schema
        if pg_file.exists():
            try:
                with open(pg_file, 'r', encoding='utf-8') as f:
                    triple.pg_ddl = f.read()
                triple.pg_schema = parser.parse_sql(triple.pg_ddl, db_name)
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"Failed to parse {db_name} PG schema: {e}")
        
        # Parse Oracle schema
        if oracle_file.exists():
            try:
                with open(oracle_file, 'r', encoding='utf-8') as f:
                    triple.oracle_ddl = f.read()
                triple.oracle_schema = parser.parse_sql(triple.oracle_ddl, db_name)
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"Failed to parse {db_name} Oracle schema: {e}")
        
        # Cross-dialect allowed_values sync:
        # If a column in any dialect has allowed_values (ENUM/CHECK), sync to corresponding columns in other dialects
        _sync_allowed_values_across_dialects(triple)
        
        schemas[db_name] = triple
    
    return schemas


def _sync_allowed_values_across_dialects(triple: TripleSchemaInfo):
    """
    Cross-dialect sync of allowed_values:
    If a column in any of the three dialects has allowed_values (ENUM/CHECK), sync to corresponding columns in other dialects.
    
    Priority: Use the strictest (non-empty) allowed_values.
    If multiple dialects have different values, take the intersection (strictest).
    """
    all_schemas = []
    if triple.mysql_schema:
        all_schemas.append(triple.mysql_schema)
    if triple.pg_schema:
        all_schemas.append(triple.pg_schema)
    if triple.oracle_schema:
        all_schemas.append(triple.oracle_schema)
    
    if len(all_schemas) < 2:
        return
    
    # Collect all (table_name, col_name) -> allowed_values from all dialects
    enum_map: Dict[Tuple[str, str], List[str]] = {}
    
    for schema in all_schemas:
        for table in schema.tables:
            for col in table.columns:
                if col.allowed_values:
                    key = (table.name.lower(), col.name.lower())
                    if key not in enum_map:
                        enum_map[key] = col.allowed_values
                    else:
                        # Multiple dialects have allowed_values: take intersection (strictest)
                        existing = set(enum_map[key])
                        current = set(col.allowed_values)
                        intersection = existing & current
                        if intersection:
                            enum_map[key] = sorted(intersection)
                        # If intersection is empty, keep the original (do not overwrite)
    
    # Sync collected allowed_values to all dialects
    for schema in all_schemas:
        for table in schema.tables:
            for col in table.columns:
                key = (table.name.lower(), col.name.lower())
                if key in enum_map and not col.allowed_values:
                    col.allowed_values = enum_map[key]


# =============================================================================
# CLI test
# =============================================================================
if __name__ == "__main__":
    import sys
    
    project_root = Path(__file__).parent.parent.parent
    database_dir = project_root / "database"
    
    print("=" * 80)
    print("Schema Parser - Parse all databases")
    print("=" * 80)
    
    schemas = parse_all_schemas(str(database_dir), dialect='mysql')
    
    for db_name, schema in schemas.items():
        print(f"\n{'─' * 60}")
        print(f"📦 {db_name} ({len(schema.tables)} tables)")
        print(f"  Topological sort: {' -> '.join(schema.topological_order())}")
        for t in schema.tables:
            col_types = []
            if t.string_columns(): col_types.append(f"STR:{len(t.string_columns())}")
            if t.integer_columns(): col_types.append(f"INT:{len(t.integer_columns())}")
            if t.decimal_columns(): col_types.append(f"DEC:{len(t.decimal_columns())}")
            if t.date_columns(): col_types.append(f"DATE:{len(t.date_columns())}")
            if t.boolean_columns(): col_types.append(f"BOOL:{len(t.boolean_columns())}")
            if t.json_columns(): col_types.append(f"JSON:{len(t.json_columns())}")
            fk_str = f", FK→{', '.join(t.referenced_tables())}" if t.referenced_tables() else ""
            self_ref = " [SELF_REF]" if t.has_self_reference() else ""
            print(f"    {t.name}: {len(t.columns)} cols ({', '.join(col_types)}){fk_str}{self_ref}")
