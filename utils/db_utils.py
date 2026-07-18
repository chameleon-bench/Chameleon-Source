"""
Database utility module.

Provides unified database connection, query, and execution functionality.
Supports MySQL, PostgreSQL, and Oracle.
"""

import pymysql
import psycopg2
try:
    import oracledb as cx_Oracle  # Prefer the new oracledb package
except ImportError:
    try:
        import cx_Oracle
    except ImportError:
        cx_Oracle = None
from contextlib import contextmanager
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class DatabaseConfig:
    """Database configuration."""
    host: str
    port: int
    user: str
    password: str
    database: Optional[str] = None


class DatabaseManager:
    """Database manager."""

    def __init__(self, config: DatabaseConfig, db_type: str = 'mysql'):
        """
        Initialize the database manager.

        Args:
            config: Database configuration
            db_type: Database type 'mysql', 'postgresql', or 'oracle'
        """
        self.config = config
        self.db_type = db_type.lower()
        self._connection = None

    def _connect(self, database: str = None, autocommit: bool = False):
        """Establish a connection."""
        db = database or self.config.database

        if self.db_type == 'mysql':
            conn = pymysql.connect(
                host=self.config.host,
                port=self.config.port,
                user=self.config.user,
                password=self.config.password,
                database=db,
                charset='utf8mb4',
                cursorclass=pymysql.cursors.DictCursor,
                autocommit=autocommit,
                read_timeout=30,
                write_timeout=30
            )
            return conn
        elif self.db_type == 'postgresql':
            return psycopg2.connect(
                host=self.config.host,
                port=self.config.port,
                user=self.config.user,
                password=self.config.password,
                database=db or 'postgres',
                connect_timeout=10,
                options='-c statement_timeout=30000'
            )
        elif self.db_type == 'oracle':
            if cx_Oracle is None:
                raise ImportError("Oracle driver not installed (oracledb or cx_Oracle)")

            # Oracle: connect using a fixed service_name, then switch CURRENT_SCHEMA
            # db_name is the database name (e.g., academic_research_and_evaluation),
            # which in Oracle corresponds to a schema/user (uppercased, truncated to 30 chars)
            service_name = getattr(self.config, 'service_name', None) or 'XE'
            dsn = f"{self.config.host}:{self.config.port}/{service_name}"
            conn = cx_Oracle.connect(
                user=self.config.user,
                password=self.config.password,
                dsn=dsn,
            )
            # Oracle query timeout is set via connection property (not a connect parameter)
            conn.call_timeout = 30000

            # Set CURRENT_SCHEMA so SQL can be written without schema prefix
            if db:
                # Oracle username: uppercase, truncated to 30 chars (Oracle identifier limit)
                schema_name = db.upper()[:30]
                with conn.cursor() as cur:
                    cur.execute(f'ALTER SESSION SET CURRENT_SCHEMA = {schema_name}')

            return conn
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}")

    @contextmanager
    def get_connection(self, database: str = None, autocommit: bool = False):
        """
        Get a database connection (context manager).

        Usage example:
            with db_manager.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT * FROM table")
        """
        conn = None
        try:
            conn = self._connect(database, autocommit=autocommit)
            yield conn
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise
        finally:
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass

    def execute(self, sql: str, params: tuple = None, database: str = None) -> bool:
        """
        Execute a SQL statement (no return result).

        Args:
            sql: SQL statement
            params: Parameters
            database: Database name

        Returns:
            Whether execution succeeded
        """
        try:
            with self.get_connection(database, autocommit=True) as conn:
                if self.db_type == 'postgresql':
                    conn.autocommit = True

                with conn.cursor() as cursor:
                    cursor.execute(sql, params)
                return True
        except Exception as e:
            logger.error(f"SQL execution failed: {sql}, error: {e}")
            return False

    def execute_many(self, sql_statements: List[str], database: str = None) -> Tuple[int, int]:
        """
        Execute multiple SQL statements.

        Args:
            sql_statements: List of SQL statements
            database: Database name

        Returns:
            (success_count, fail_count)
        """
        success_count = 0
        fail_count = 0

        try:
            with self.get_connection(database, autocommit=True) as conn:
                if self.db_type == 'postgresql':
                    conn.autocommit = True

                with conn.cursor() as cursor:
                    for sql in sql_statements:
                        sql = sql.strip()
                        if not sql:
                            continue
                        try:
                            cursor.execute(sql)
                            success_count += 1
                        except Exception as e:
                            logger.warning(f"SQL execution failed: {sql[:100]}..., error: {e}")
                            fail_count += 1

        except Exception as e:
            logger.error(f"Batch SQL execution failed: {e}")

        return success_count, fail_count

    def query(self, sql: str, params: tuple = None, database: str = None) -> List[Dict[str, Any]]:
        """
        Execute a query SQL.

        Args:
            sql: SQL statement
            params: Parameters
            database: Database name

        Returns:
            List of query results
        """
        try:
            with self.get_connection(database) as conn:
                with conn.cursor() as cursor:
                    cursor.execute(sql, params)

                    if self.db_type == 'mysql':
                        return cursor.fetchall()
                    else:
                        columns = [desc[0] for desc in cursor.description] if cursor.description else []
                        rows = cursor.fetchall()
                        return [dict(zip(columns, row)) for row in rows]
        except Exception as e:
            logger.error(f"Query failed: {sql}, error: {e}")
            return []

    def query_one(self, sql: str, params: tuple = None, database: str = None) -> Optional[Dict[str, Any]]:
        """
        Execute a query SQL, return a single result.

        Args:
            sql: SQL statement
            params: Parameters
            database: Database name

        Returns:
            Single result or None
        """
        results = self.query(sql, params, database)
        return results[0] if results else None

    def database_exists(self, db_name: str) -> bool:
        """
        Check if a database exists.

        Args:
            db_name: Database name

        Returns:
            Whether it exists
        """
        try:
            if self.db_type == 'mysql':
                sql = "SELECT SCHEMA_NAME FROM INFORMATION_SCHEMA.SCHEMATA WHERE SCHEMA_NAME = %s"
                result = self.query_one(sql, (db_name,))
            else:
                sql = "SELECT datname FROM pg_database WHERE datname = %s"
                result = self.query_one(sql, (db_name,), database='postgres')

            return result is not None
        except Exception as e:
            logger.error(f"Failed to check database existence: {e}")
            return False

    def create_database(self, db_name: str, charset: str = 'utf8mb4') -> bool:
        """
        Create a database.

        Args:
            db_name: Database name
            charset: Character set (MySQL only)

        Returns:
            Whether creation succeeded
        """
        try:
            if self.db_type == 'mysql':
                sql = f"CREATE DATABASE IF NOT EXISTS `{db_name}` CHARACTER SET {charset} COLLATE {charset}_unicode_ci"
                return self.execute(sql)
            else:
                # PostgreSQL requires creating a new database in the postgres database
                # Use double quotes around the database name to support special characters
                safe_name = db_name.replace('"', '""')
                with self.get_connection('postgres', autocommit=True) as conn:
                    conn.autocommit = True
                    with conn.cursor() as cursor:
                        try:
                            cursor.execute(f'CREATE DATABASE "{safe_name}" ENCODING \'UTF8\'')
                            return True
                        except psycopg2.errors.DuplicateDatabase:
                            logger.warning(f"Database {db_name} already exists")
                            return True
        except Exception as e:
            logger.error(f"Failed to create database: {e}")
            return False

    def drop_database(self, db_name: str) -> bool:
        """
        Drop a database.

        Args:
            db_name: Database name

        Returns:
            Whether deletion succeeded
        """
        try:
            if self.db_type == 'mysql':
                sql = f"DROP DATABASE IF EXISTS `{db_name}`"
                return self.execute(sql)
            else:
                # PostgreSQL requires disconnecting all connections first
                safe_name = db_name.replace('"', '""')
                with self.get_connection('postgres', autocommit=True) as conn:
                    conn.autocommit = True
                    with conn.cursor() as cursor:
                        # Terminate all connections
                        cursor.execute(
                            "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = %s",
                            (db_name,)
                        )
                        cursor.execute(f'DROP DATABASE IF EXISTS "{safe_name}"')
                return True
        except Exception as e:
            logger.error(f"Failed to drop database: {e}")
            return False

    def execute_schema_file(self, schema_file: str, db_name: str) -> bool:
        """
        Execute a schema file.

        Args:
            schema_file: Schema file path
            db_name: Target database name

        Returns:
            Whether execution succeeded
        """
        try:
            with open(schema_file, 'r', encoding='utf-8') as f:
                schema_sql = f.read()

            # Split SQL statements
            statements = [s.strip() for s in schema_sql.split(';') if s.strip()]

            success_count, fail_count = self.execute_many(statements, db_name)

            logger.info(f"Schema execution complete: {success_count} succeeded, {fail_count} failed")
            return fail_count == 0

        except Exception as e:
            logger.error(f"Failed to execute schema file: {e}")
            return False

    def list_databases(self) -> List[str]:
        """
        List all databases.

        Returns:
            List of database names
        """
        try:
            if self.db_type == 'mysql':
                sql = "SHOW DATABASES"
                results = self.query(sql)
                return [row['Database'] for row in results]
            else:
                sql = "SELECT datname FROM pg_database WHERE datistemplate = false"
                results = self.query(sql, database='postgres')
                return [row['datname'] for row in results]
        except Exception as e:
            logger.error(f"Failed to list databases: {e}")
            return []

    def list_tables(self, db_name: str) -> List[str]:
        """
        List all tables in a database.

        Args:
            db_name: Database name

        Returns:
            List of table names
        """
        try:
            if self.db_type == 'mysql':
                sql = "SHOW TABLES"
                results = self.query(sql, database=db_name)
                key = f"Tables_in_{db_name}"
                return [row[key] for row in results if key in row]
            else:
                sql = "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
                results = self.query(sql, database=db_name)
                return [row['tablename'] for row in results]
        except Exception as e:
            logger.error(f"Failed to list tables: {e}")
            return []


def create_mysql_manager(host: str, port: int, user: str, password: str, database: str = None) -> DatabaseManager:
    """
    Create a MySQL manager.

    Args:
        host: Host address
        port: Port
        user: Username
        password: Password
        database: Database name

    Returns:
        DatabaseManager instance
    """
    config = DatabaseConfig(host=host, port=port, user=user, password=password, database=database)
    return DatabaseManager(config, db_type='mysql')


def create_postgresql_manager(host: str, port: int, user: str, password: str, database: str = None) -> DatabaseManager:
    """
    Create a PostgreSQL manager.

    Args:
        host: Host address
        port: Port
        user: Username
        password: Password
        database: Database name

    Returns:
        DatabaseManager instance
    """
    config = DatabaseConfig(host=host, port=port, user=user, password=password, database=database)
    return DatabaseManager(config, db_type='postgresql')
