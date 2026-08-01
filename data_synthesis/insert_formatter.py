"""
Tri-dialect INSERT formatter (Insert Formatter).

Formats in-memory {table: [row_dict, ...]} data into:
  - MySQL INSERT statements
  - PostgreSQL INSERT statements
  - Oracle INSERT statements

Handles type differences:
  - BOOLEAN: MySQL uses 1/0, PG uses TRUE/FALSE, Oracle uses 1/0 (NUMBER(1))
  - JSON: MySQL uses JSON literal, PG uses JSONB cast, Oracle stores CLOB plain string
  - NULL: consistent across all three
  - String escaping: MySQL backslash escaping, PG/Oracle standard SQL escaping ('' for ')
  - Batch INSERT: MySQL/PG multi-row VALUES, Oracle single-row INSERT
  - DATETIME vs TIMESTAMP: format consistent, Oracle uses TO_TIMESTAMP()
  - DECIMAL vs NUMERIC vs NUMBER: value format consistent
"""

import json
import math
import sys
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import date, datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data_synthesis.schema_parser import SchemaInfo, TableInfo, ColumnInfo
from utils.logging_config import get_logger

logger = get_logger(__name__)


class InsertFormatter:
    """
    Tri-dialect INSERT statement formatter.
    
    Formats the same in-memory data into INSERT SQL for three dialects.
    """
    
    # ISO 8601 datetime regex pattern (matches common LLM output formats)
    _ISO_DATETIME_RE = re.compile(
        r'^\d{4}-\d{2}-\d{2}'           # 2024-01-15
        r'[T ]\d{2}:\d{2}:\d{2}'        # T08:30:00
        r'(?:\.\d+)?'                    # .000 (optional milliseconds)
        r'(?:Z|[+-]\d{2}:?\d{2})?$'     # Z or +08:00 (optional timezone)
    )
    _ISO_DATE_RE = re.compile(r'^\d{4}-\d{2}-\d{2}$')
    _ISO_TIME_RE = re.compile(r'^\d{2}:\d{2}:\d{2}(?:\.\d+)?$')
    
    def __init__(self, batch_size: int = 50):
        """
        Args:
            batch_size: Number of rows per INSERT statement (avoid overly long single statements)
        """
        self.batch_size = batch_size
    
    # =========================================================================
    # Common: Schema-aware string value normalization
    # =========================================================================
    
    def _normalize_datetime_str(self, value_str: str) -> str:
        """
        Normalize ISO 8601 datetime string to standard 'YYYY-MM-DD HH:MM:SS' format.
        Applicable to MySQL and PG (both accept this format).
        
        Handles:
          - 2024-01-15T08:30:00.000Z     → 2024-01-15 08:30:00
          - 2024-01-15T08:30:00+08:00    → 2024-01-15 08:30:00
          - 2024-01-15T08:30:00.123456   → 2024-01-15 08:30:00
          - 2024-01-15 08:30:00          -> 2024-01-15 08:30:00 (already standard, unchanged)
        """
        s = value_str.strip()
        # Replace T separator with space
        s = s.replace('T', ' ')
        # Remove milliseconds (.000, .123456 etc.)
        s = re.sub(r'\.\d+', '', s)
        # Remove timezone suffix (Z, +08:00, -05:00 etc.)
        s = re.sub(r'[Zz]$', '', s)
        s = re.sub(r'[+-]\d{2}:?\d{2}$', '', s)
        return s.strip()
    
    def _is_datetime_string(self, value_str: str) -> bool:
        """Detect if string is a datetime format."""
        return bool(self._ISO_DATETIME_RE.match(value_str.strip()))
    
    def _is_date_string(self, value_str: str) -> bool:
        """Detect if string is a pure date format (YYYY-MM-DD)."""
        return bool(self._ISO_DATE_RE.match(value_str.strip()))
    
    def _is_time_string(self, value_str: str) -> bool:
        """Detect if string is a pure time format (HH:MM:SS)."""
        return bool(self._ISO_TIME_RE.match(value_str.strip()))
    
    def _normalize_boolean_str(self, value_str: str, dialect: str) -> Optional[str]:
        """
        Normalize boolean string to database-accepted format.
        
        Returns:
            Normalized value, or None if not a boolean.
        """
        lower = value_str.strip().lower()
        if lower in ('true', '1', 'yes', 'on'):
            if dialect == 'mysql':
                return '1'
            elif dialect == 'pg':
                return 'TRUE'
            else:  # oracle
                return '1'
        elif lower in ('false', '0', 'no', 'off'):
            if dialect == 'mysql':
                return '0'
            elif dialect == 'pg':
                return 'FALSE'
            else:  # oracle
                return '0'
        return None
    
    def _normalize_numeric_str(self, value_str: str) -> Optional[str]:
        """
        Remove quotes from numeric strings and output directly.
        
        Returns:
            Numeric literal, or None if not a valid number
        """
        s = value_str.strip()
        try:
            # Try to parse as number
            if '.' in s:
                f = float(s)
                # NaN / Infinity -> not a valid number
                if not math.isfinite(f):
                    return None
                if f == int(f) and abs(f) < 1e15:
                    return str(int(f))
                return f"{f:.6f}".rstrip('0').rstrip('.')
            else:
                return str(int(s))
        except (ValueError, OverflowError):
            return None
    
    def format_mysql(
        self,
        table_name: str,
        columns: List[str],
        rows: List[Dict],
        schema_table: Optional[TableInfo] = None,
    ) -> str:
        """
        Format MySQL INSERT statement.
        
        Args:
            table_name: Table name
            columns: Column name list (in order)
            rows: Data row list [{col: val, ...}, ...]
            schema_table: Table schema info (for type-aware value formatting)
        
        Returns:
            Complete MySQL INSERT SQL text
        """
        if not rows:
            return f"-- {table_name}: no data\n"
        
        statements = []
        statements.append(f"-- {table_name}: {len(rows)} rows")
        
        # Generate INSERT in batches
        for batch_start in range(0, len(rows), self.batch_size):
            batch = rows[batch_start:batch_start + self.batch_size]
            
            col_list = ", ".join(f"`{c}`" for c in columns)
            
            value_rows = []
            for row in batch:
                values = []
                for col_name in columns:
                    val = row.get(col_name)
                    col_info = schema_table.get_column(col_name) if schema_table else None
                    values.append(self._format_value_mysql(val, col_info))
                value_rows.append(f"  ({', '.join(values)})")
            
            stmt = f"INSERT INTO `{table_name}` ({col_list}) VALUES\n"
            stmt += ",\n".join(value_rows)
            stmt += ";"
            statements.append(stmt)
        
        return "\n\n".join(statements) + "\n"
    
    def format_pg(
        self,
        table_name: str,
        columns: List[str],
        rows: List[Dict],
        schema_table: Optional[TableInfo] = None,
    ) -> str:
        """
        Format PostgreSQL INSERT statement.
        """
        if not rows:
            return f"-- {table_name}: no data\n"
        
        statements = []
        statements.append(f"-- {table_name}: {len(rows)} rows")
        
        for batch_start in range(0, len(rows), self.batch_size):
            batch = rows[batch_start:batch_start + self.batch_size]
            
            col_list = ", ".join(f'"{c}"' for c in columns)
            
            value_rows = []
            for row in batch:
                values = []
                row_lower = {k.lower(): v for k, v in row.items()}
                for col_name in columns:
                    # Prefer exact match, then fallback to case-insensitive match
                    val = row.get(col_name)
                    if val is None and col_name.lower() in row_lower:
                        mapped_val = row_lower[col_name.lower()]
                        if mapped_val is not None:
                            val = mapped_val
                    col_info = schema_table.get_column(col_name) if schema_table else None
                    values.append(self._format_value_pg(val, col_info))
                value_rows.append(f"  ({', '.join(values)})")
            
            stmt = f'INSERT INTO "{table_name}" ({col_list}) VALUES\n'
            stmt += ",\n".join(value_rows)
            stmt += ";"
            statements.append(stmt)
        
        return "\n\n".join(statements) + "\n"
    
    def format_database_mysql(
        self,
        db_name: str,
        data: Dict[str, List[Dict]],
        schema: SchemaInfo,
        generation_order: List[str],
    ) -> str:
        """Format MySQL INSERT for entire database."""
        lines = [
            f"-- ====================================",
            f"-- MySQL INSERT script: {db_name}",
            f"-- auto-generated by Translation_BENCHMARK",
            f"-- ====================================",
            "",
            "SET NAMES utf8mb4;",
            "SET FOREIGN_KEY_CHECKS = 0;",
            "",
        ]
        
        for table_name in generation_order:
            rows = data.get(table_name, [])
            if not rows:
                continue
            
            table_info = schema.get_table(table_name)
            columns = [col.name for col in table_info.columns] if table_info else list(rows[0].keys())
            
            lines.append(self.format_mysql(table_name, columns, rows, table_info))
        
        lines.append("")
        lines.append("SET FOREIGN_KEY_CHECKS = 1;")
        lines.append("")
        
        return "\n".join(lines)
    
    def format_database_pg(
        self,
        db_name: str,
        data: Dict[str, List[Dict]],
        schema: SchemaInfo,
        generation_order: List[str],
    ) -> str:
        """Format PostgreSQL INSERT for entire database."""
        lines = [
            f"-- ====================================",
            f"-- PostgreSQL INSERT script: {db_name}",
            f"-- auto-generated by Translation_BENCHMARK",
            f"-- ====================================",
            "",
            "SET client_encoding = 'UTF8';",
            "",
        ]
        
        for table_name in generation_order:
            rows = data.get(table_name, [])
            if not rows:
                continue
            
            table_info = schema.get_table(table_name)
            columns = [col.name for col in table_info.columns] if table_info else list(rows[0].keys())
            
            lines.append(self.format_pg(table_name, columns, rows, table_info))
        
        # Reset sequences (IDENTITY/SERIAL)
        lines.append("")
        lines.append("-- Reset IDENTITY sequences")
        for table_name in generation_order:
            rows = data.get(table_name, [])
            if not rows:
                continue
            table_info = schema.get_table(table_name)
            if table_info:
                for col in table_info.columns:
                    if col.is_auto_increment:
                        max_val = max((r.get(col.name, 0) or 0) for r in rows)
                        lines.append(
                            f"SELECT setval(pg_get_serial_sequence('{table_name}', '{col.name}'), "
                            f"{max_val}, true);"
                        )
        
        lines.append("")
        
        return "\n".join(lines)
    
    # =========================================================================
    # Value formatting (MySQL)
    # =========================================================================
    
    def _format_value_mysql(self, value: Any, col_info: Optional[ColumnInfo] = None) -> str:
        """Format Python value as MySQL INSERT value."""
        if value is None:
            return "NULL"
        
        # BOOLEAN
        if isinstance(value, bool):
            return "1" if value else "0"
        
        # Number
        if isinstance(value, (int, float)):
            if isinstance(value, float):
                # NaN / Infinity → NULL
                if not math.isfinite(value):
                    return "NULL"
                # Avoid scientific notation
                if value == int(value) and abs(value) < 1e15:
                    return str(int(value))
                return f"{value:.6f}".rstrip('0').rstrip('.')
            return str(value)
        
        # Date (Python datetime object)
        if isinstance(value, datetime):
            return f"'{value.strftime('%Y-%m-%d %H:%M:%S')}'"
        if isinstance(value, date):
            return f"'{value.strftime('%Y-%m-%d')}'"
        
        # String - Schema-aware normalization
        value_str = str(value)
        
        if col_info:
            # 1. Date/time type columns: normalize ISO 8601 format
            if col_info.is_date_type():
                if self._is_datetime_string(value_str):
                    normalized = self._normalize_datetime_str(value_str)
                    return f"'{self._escape_mysql(normalized)}'"
                elif self._is_date_string(value_str):
                    return f"'{self._escape_mysql(value_str.strip())}'"
                elif self._is_time_string(value_str):
                    # Remove milliseconds
                    t = re.sub(r'\.\d+', '', value_str.strip())
                    # Fallback: DATETIME/TIMESTAMP column receives pure time value -> prepend date prefix
                    bt = col_info.base_type()
                    if bt in ('DATETIME', 'TIMESTAMP', 'TIMESTAMPTZ', 'DATE'):
                        t = f"2024-01-01 {t}"
                    return f"'{self._escape_mysql(t)}'"
            
            # 2. Boolean type columns: string true/false -> 1/0
            if col_info.is_boolean_type():
                bool_val = self._normalize_boolean_str(value_str, 'mysql')
                if bool_val is not None:
                    return bool_val
            
            # 3. Numeric type columns: string number -> remove quotes
            if col_info.is_numeric_type() and not col_info.is_boolean_type():
                num_val = self._normalize_numeric_str(value_str)
                if num_val is not None:
                    return num_val
        else:
            # Without schema info, also try to detect obvious ISO datetime format
            if self._is_datetime_string(value_str):
                normalized = self._normalize_datetime_str(value_str)
                return f"'{self._escape_mysql(normalized)}'"
        
        # BLOB / binary type: hex literal without quotes
        if col_info and col_info.is_blob_type():
            if value_str.startswith('0x') or value_str.startswith('0X'):
                return value_str  # MySQL: 0x89504E47 without quotes, directly as hex literal
            # Non-hex format binary data, use UNHEX function
            return f"UNHEX('{self._escape_mysql(value_str)}')"

        # JSON detection
        if col_info and col_info.is_json_type():
            # MySQL JSON does not need special cast
            return f"'{self._escape_mysql(value_str)}'"
        
        # Plain string
        return f"'{self._escape_mysql(value_str)}'"
    
    def _escape_mysql(self, s: str) -> str:
        """MySQL string escaping."""
        s = s.replace("\\", "\\\\")
        s = s.replace("'", "\\'")
        s = s.replace('"', '\\"')
        s = s.replace("\n", "\\n")
        s = s.replace("\r", "\\r")
        s = s.replace("\t", "\\t")
        s = s.replace("\0", "\\0")
        return s
    
    # =========================================================================
    # Value formatting (PostgreSQL)
    # =========================================================================
    
    def _format_value_pg(self, value: Any, col_info: Optional[ColumnInfo] = None) -> str:
        """Format Python value as PostgreSQL INSERT value."""
        if value is None:
            return "NULL"
        
        # BOOLEAN (Python bool)
        if isinstance(value, bool):
            return "TRUE" if value else "FALSE"
        
        # Number - but if column type is BOOLEAN，1/0 → TRUE/FALSE
        if isinstance(value, (int, float)):
            if col_info and col_info.is_boolean_type():
                if isinstance(value, int):
                    return "TRUE" if value else "FALSE"
                # Also handle float boolean values
                return "TRUE" if value else "FALSE"
            if isinstance(value, float):
                # NaN / Infinity → NULL
                if not math.isfinite(value):
                    return "NULL"
                if value == int(value) and abs(value) < 1e15:
                    return str(int(value))
                return f"{value:.6f}".rstrip('0').rstrip('.')
            return str(value)
        
        # Date (Python datetime object)
        if isinstance(value, datetime):
            return f"'{value.strftime('%Y-%m-%d %H:%M:%S')}'"
        if isinstance(value, date):
            return f"'{value.strftime('%Y-%m-%d')}'"
        
        # String - Schema-aware normalization
        value_str = str(value)
        
        if col_info:
            # 1. Date/time type columns: normalize ISO 8601 format
            if col_info.is_date_type():
                if self._is_datetime_string(value_str):
                    normalized = self._normalize_datetime_str(value_str)
                    return f"'{self._escape_pg(normalized)}'"
                elif self._is_date_string(value_str):
                    return f"'{self._escape_pg(value_str.strip())}'"
                elif self._is_time_string(value_str):
                    # Remove milliseconds
                    t = re.sub(r'\.\d+', '', value_str.strip())
                    # Fallback: DATETIME/TIMESTAMP column receives pure time value -> prepend date prefix
                    bt = col_info.base_type()
                    if bt in ('DATETIME', 'TIMESTAMP', 'TIMESTAMPTZ', 'DATE'):
                        t = f"2024-01-01 {t}"
                    return f"'{self._escape_pg(t)}'"
            
            # 2. Boolean type columns: string true/false -> TRUE/FALSE
            if col_info.is_boolean_type():
                bool_val = self._normalize_boolean_str(value_str, 'pg')
                if bool_val is not None:
                    return bool_val
            
            # 3. Numeric type columns: string number -> remove quotes
            if col_info.is_numeric_type() and not col_info.is_boolean_type():
                num_val = self._normalize_numeric_str(value_str)
                if num_val is not None:
                    return num_val
        else:
            # Without schema info, also try to detect obvious ISO datetime format
            if self._is_datetime_string(value_str):
                normalized = self._normalize_datetime_str(value_str)
                return f"'{self._escape_pg(normalized)}'"
        
        # JSON detection -> add ::jsonb cast
        if col_info and col_info.is_json_type():
            return f"'{self._escape_pg(value_str)}'::jsonb"
        
        # BLOB / BYTEA type: hex string needs PG bytea hex format
        if col_info and col_info.is_blob_type():
            if value_str.startswith('0x') or value_str.startswith('0X'):
                # PG BYTEA hex format: E'\\x89504E47'
                hex_content = value_str[2:]
                return f"E'\\\\x{hex_content}'"
            # Non-hex format, store as plain string
            return f"'{self._escape_pg(value_str)}'"
        
        # ARRAY detection
        if col_info and "[]" in col_info.data_type.upper():
            # Convert JSON array -> PG ARRAY syntax
            # Extract array element type for type cast (e.g., TEXT[] -> ::TEXT[])
            array_base_type = re.sub(r'\[\]$', '', col_info.data_type.upper())
            try:
                arr = json.loads(value_str)
                if isinstance(arr, list):
                    if arr:
                        elements = ", ".join(f"'{self._escape_pg(str(e))}'" for e in arr)
                        return f"ARRAY[{elements}]::{array_base_type}[]"
                    else:
                        # Empty array must have explicit type cast, otherwise PG reports "cannot determine type of empty array"
                        return f"ARRAY[]::{array_base_type}[]"
                elif isinstance(arr, dict):
                    # JSON object -> store as single-element text array (MySQL JSON -> PG TEXT[] compatibility)
                    return f"ARRAY['{self._escape_pg(json.dumps(arr, ensure_ascii=False))}']::{array_base_type}[]"
            except (json.JSONDecodeError, TypeError):
                pass
        
        # Plain string (PG uses standard SQL escaping: double single-quote for single-quote)
        return f"'{self._escape_pg(value_str)}'"
    
    def _escape_pg(self, s: str) -> str:
        """PostgreSQL string escaping (standard SQL method)."""
        # PG standard mode: single quotes escaped with ''
        s = s.replace("'", "''")
        # Remove \r to avoid cross-platform inconsistency
        s = s.replace("\r", "")
        return s
    
    # =========================================================================
    # Single table formatting (Oracle)
    # =========================================================================
    
    def format_oracle(
        self,
        table_name: str,
        columns: List[str],
        rows: List[Dict],
        schema_table: Optional[TableInfo] = None,
    ) -> str:
        """
        Format Oracle INSERT statement (single-row INSERT).
        
        Oracle does not support multi-row VALUES syntax; each INSERT inserts one row.
        """
        if not rows:
            return f"-- {table_name}: no data\n"
        
        statements = []
        statements.append(f"-- {table_name}: {len(rows)} rows")
        
        col_list = ", ".join(f'"{c}"' for c in columns)
        
        # Build case-insensitive row lookup mapping (Oracle column names may be lowercase, data row keys preserve MySQL case)
        row_key_map = {}
        for row in rows:
            row_key_map[id(row)] = {k.lower(): (k, v) for k, v in row.items()}

        for row in rows:
            values = []
            key_map = row_key_map[id(row)]
            for col_name in columns:
                # Prefer exact match, then fallback to case-insensitive match
                val = row.get(col_name)
                if val is None and col_name.lower() in key_map:
                    _, mapped_val = key_map[col_name.lower()]
                    if mapped_val is not None:
                        val = mapped_val
                col_info = schema_table.get_column(col_name) if schema_table else None
                values.append(self._format_value_oracle(val, col_info))
            
            stmt = f'INSERT INTO "{table_name}" ({col_list}) VALUES ({", ".join(values)});'
            statements.append(stmt)
        
        return "\n".join(statements) + "\n"
    
    def format_database_oracle(
        self,
        db_name: str,
        data: Dict[str, List[Dict]],
        schema: SchemaInfo,
        generation_order: List[str],
    ) -> str:
        """Format Oracle INSERT for entire database."""
        lines = [
            f"-- ====================================",
            f"-- Oracle INSERT script: {db_name}",
            f"-- auto-generated by Translation_BENCHMARK",
            f"-- ====================================",
            "",
            "-- Disable foreign key constraint checks",
            "BEGIN",
            "  FOR c IN (SELECT table_name, constraint_name FROM user_constraints WHERE constraint_type = 'R') LOOP",
            "    EXECUTE IMMEDIATE 'ALTER TABLE \"' || c.table_name || '\" DISABLE CONSTRAINT \"' || c.constraint_name || '\"';",
            "  END LOOP;",
            "END;",
            "/",
            "",
        ]
        
        for table_name in generation_order:
            rows = data.get(table_name, [])
            if not rows:
                continue
            
            table_info = schema.get_table(table_name)
            columns = [col.name for col in table_info.columns] if table_info else list(rows[0].keys())
            
            lines.append(self.format_oracle(table_name, columns, rows, table_info))
        
        # Restore foreign key constraints
        lines.append("")
        lines.append("-- Restore foreign key constraint checks")
        lines.append("BEGIN")
        lines.append("  FOR c IN (SELECT table_name, constraint_name FROM user_constraints WHERE constraint_type = 'R') LOOP")
        lines.append("    EXECUTE IMMEDIATE 'ALTER TABLE \"' || c.table_name || '\" ENABLE CONSTRAINT \"' || c.constraint_name || '\"';")
        lines.append("  END LOOP;")
        lines.append("END;")
        lines.append("/")
        lines.append("")
        lines.append("COMMIT;")
        lines.append("")
        
        return "\n".join(lines)
    
    # =========================================================================
    # Value formatting (Oracle)
    # =========================================================================
    
    def _format_value_oracle(self, value: Any, col_info: Optional[ColumnInfo] = None) -> str:
        """Format Python value as Oracle INSERT value."""
        if value is None:
            return "NULL"
        
        # BOOLEAN -> Oracle uses NUMBER(1): 1/0
        if isinstance(value, bool):
            return "1" if value else "0"
        
        # Number
        if isinstance(value, (int, float)):
            if isinstance(value, float):
                # NaN / Infinity → NULL
                if not math.isfinite(value):
                    return "NULL"
                if value == int(value) and abs(value) < 1e15:
                    return str(int(value))
                return f"{value:.6f}".rstrip('0').rstrip('.')
            return str(value)
        
        # Date
        if isinstance(value, datetime):
            return f"TO_TIMESTAMP('{value.strftime('%Y-%m-%d %H:%M:%S')}', 'YYYY-MM-DD HH24:MI:SS')"
        if isinstance(value, date):
            return f"TO_DATE('{value.strftime('%Y-%m-%d')}', 'YYYY-MM-DD')"
        
        # String
        value_str = str(value)
        
        # Date/time string detection (LLM-generated data is typically string-format dates)
        if col_info:
            fbt = col_info.full_base_type()
            bt = col_info.base_type()

            # Fallback: TIMESTAMP/DATE column receives pure time value -> prepend date prefix
            if bt in ('TIMESTAMP', 'DATE', 'TIMESTAMPTZ') or fbt in (
                'TIMESTAMP WITH TIME ZONE', 'TIMESTAMP WITH LOCAL TIME ZONE'
            ):
                if self._is_time_string(value_str):
                    t = re.sub(r'\.\d+', '', value_str.strip())
                    value_str = f"2024-01-01 {t}"

            if fbt in ('TIMESTAMP WITH TIME ZONE', 'TIMESTAMP WITH LOCAL TIME ZONE'):
                return self._format_oracle_timestamp_tz(value_str)
            if bt == 'TIMESTAMP':
                return self._format_oracle_timestamp(value_str)
            if bt == 'DATE':
                # First normalize ISO 8601 format (T separator, milliseconds, timezone)
                normalized = self._normalize_datetime_str(value_str) if self._is_datetime_string(value_str) else value_str
                # Oracle DATE includes time part
                if ' ' in normalized:
                    return f"TO_DATE('{self._escape_oracle(normalized)}', 'YYYY-MM-DD HH24:MI:SS')"
                return f"TO_DATE('{self._escape_oracle(normalized)}', 'YYYY-MM-DD')"
            
            # Boolean type columns: string true/false -> 1/0
            if col_info.is_boolean_type():
                bool_val = self._normalize_boolean_str(value_str, 'oracle')
                if bool_val is not None:
                    return bool_val
            
            # Numeric type columns: string number -> remove quotes
            if col_info.is_numeric_type() and not col_info.is_boolean_type():
                num_val = self._normalize_numeric_str(value_str)
                if num_val is not None:
                    return num_val
        
        # BLOB type: hex string needs HEXTORAW() conversion
        if col_info and col_info.is_blob_type():
            if value_str.startswith('0x') or value_str.startswith('0X'):
                return f"HEXTORAW('{value_str[2:]}')"
            # Non-hex format binary data, try to store directly
            return f"UTL_RAW.CAST_TO_RAW('{self._escape_oracle(value_str)}')"

        # CLOB type (Oracle equivalent of MySQL JSON/LONGTEXT and PG JSONB/TEXT)
        # Use string literal directly, Oracle will auto-convert to CLOB
        
        # Plain string (Oracle uses standard SQL escaping: double single-quote for single-quote)
        return f"'{self._escape_oracle(value_str)}'"
    
    def _format_oracle_timestamp_tz(self, value_str: str) -> str:
        """Format time string as Oracle TO_TIMESTAMP_TZ call.

        Auto-detect value format and select matching Oracle format string:
          - 2024-01-15T08:30:00-05:00  → YYYY-MM-DD"T"HH24:MI:SSTZH:TZM
          - 2024-01-15 08:30:00+00:00  → YYYY-MM-DD HH24:MI:SSTZH:TZM
          - 2024-01-15T08:30:00        → YYYY-MM-DD"T"HH24:MI:SS
          - 2024-01-15 08:30:00        → YYYY-MM-DD HH24:MI:SS (no timezone, Oracle uses session timezone)
        """
        s = value_str.strip()
        # Remove milliseconds (.000, .123456 etc.)
        s = re.sub(r'\.\d+', '', s)
        # Remove trailing Z (UTC indicator) - remove if no +/- timezone offset
        if s.endswith('Z') or s.endswith('z'):
            s = s[:-1] + '+00:00'
        
        has_t = 'T' in s and s.index('T') == 10
        has_tz = bool(re.search(r'[+-]\d{2}:\d{2}$', s))

        if has_t and has_tz:
            fmt = 'YYYY-MM-DD"T"HH24:MI:SSTZH:TZM'
        elif has_tz:
            fmt = 'YYYY-MM-DD HH24:MI:SSTZH:TZM'
        elif has_t:
            fmt = 'YYYY-MM-DD"T"HH24:MI:SS'
        else:
            fmt = 'YYYY-MM-DD HH24:MI:SS'

        return f"TO_TIMESTAMP_TZ('{self._escape_oracle(s)}', '{fmt}')"

    def _format_oracle_timestamp(self, value_str: str) -> str:
        """Format time string as Oracle TO_TIMESTAMP call.

        Auto-detect whether value contains T separator:
          - 2024-01-15T08:30:00  → YYYY-MM-DD"T"HH24:MI:SS
          - 2024-01-15 08:30:00  → YYYY-MM-DD HH24:MI:SS
        """
        s = value_str.strip()
        # Remove milliseconds and timezone suffix (LLM may generate .000Z format)
        s = re.sub(r'\.\d+', '', s)
        s = re.sub(r'[Zz]$', '', s)
        s = re.sub(r'[+-]\d{2}:?\d{2}$', '', s)
        
        has_t = 'T' in s and s.index('T') == 10

        if has_t:
            fmt = 'YYYY-MM-DD"T"HH24:MI:SS'
        else:
            fmt = 'YYYY-MM-DD HH24:MI:SS'

        return f"TO_TIMESTAMP('{self._escape_oracle(s)}', '{fmt}')"

    def _escape_oracle(self, s: str) -> str:
        """Oracle string escaping (standard SQL method, same as PG)."""
        s = s.replace("'", "''")
        # Remove \r to avoid cross-platform inconsistency
        s = s.replace("\r", "")
        return s
    
    # =========================================================================
    # File output
    # =========================================================================
    
    def save_mysql(
        self,
        db_name: str,
        data: Dict[str, List[Dict]],
        schema: SchemaInfo,
        generation_order: List[str],
        output_dir: str,
    ) -> str:
        """Save MySQL INSERT file."""
        sql = self.format_database_mysql(db_name, data, schema, generation_order)
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        filepath = output_path / f"{db_name}_data_mysql.sql"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(sql)
        
        logger.info(f"  MySQL INSERT saved: {filepath}")
        return str(filepath)
    
    def save_pg(
        self,
        db_name: str,
        data: Dict[str, List[Dict]],
        schema: SchemaInfo,
        generation_order: List[str],
        output_dir: str,
    ) -> str:
        """Save PostgreSQL INSERT file."""
        sql = self.format_database_pg(db_name, data, schema, generation_order)
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        filepath = output_path / f"{db_name}_data_pg.sql"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(sql)
        
        logger.info(f"  PG INSERT saved: {filepath}")
        return str(filepath)
    
    def save_both(
        self,
        db_name: str,
        data: Dict[str, List[Dict]],
        mysql_schema: SchemaInfo,
        pg_schema: Optional[SchemaInfo],
        generation_order: List[str],
        output_dir: str,
    ) -> Dict[str, str]:
        """Save both MySQL and PG INSERT files (backward compatible)."""
        mysql_path = self.save_mysql(db_name, data, mysql_schema, generation_order, output_dir)
        
        pg_schema_to_use = pg_schema or mysql_schema
        pg_path = self.save_pg(db_name, data, pg_schema_to_use, generation_order, output_dir)
        
        return {"mysql": mysql_path, "pg": pg_path}
    
    def save_oracle(
        self,
        db_name: str,
        data: Dict[str, List[Dict]],
        schema: SchemaInfo,
        generation_order: List[str],
        output_dir: str,
    ) -> str:
        """Save Oracle INSERT file."""
        sql = self.format_database_oracle(db_name, data, schema, generation_order)
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        filepath = output_path / f"{db_name}_data_oracle.sql"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(sql)
        
        logger.info(f"  Oracle INSERT saved: {filepath}")
        return str(filepath)
    
    def save_all(
        self,
        db_name: str,
        data: Dict[str, List[Dict]],
        mysql_schema: SchemaInfo,
        pg_schema: Optional[SchemaInfo],
        oracle_schema: Optional[SchemaInfo],
        generation_order: List[str],
        output_dir: str,
    ) -> Dict[str, str]:
        """Save MySQL, PG, Oracle tri-dialect INSERT files simultaneously."""
        mysql_path = self.save_mysql(db_name, data, mysql_schema, generation_order, output_dir)
        
        pg_schema_to_use = pg_schema or mysql_schema
        pg_path = self.save_pg(db_name, data, pg_schema_to_use, generation_order, output_dir)
        
        oracle_schema_to_use = oracle_schema or mysql_schema
        oracle_path = self.save_oracle(db_name, data, oracle_schema_to_use, generation_order, output_dir)
        
        return {"mysql": mysql_path, "pg": pg_path, "oracle": oracle_path}
