"""SQLite connection management and migration runner.

Provides the :class:`Database` class for async SQLite access via ``aiosqlite``.
Migrations are discovered from ``db/migrations/*.sql`` in lexicographic order
and applied idempotently using a ``schema_migrations`` tracking table.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import aiosqlite

logger = logging.getLogger(__name__)

# Default path to the migrations directory, relative to this file.
_DEFAULT_MIGRATIONS_DIR = Path(__file__).parent / "migrations"

# SQL to create the migration tracking table.
_CREATE_MIGRATIONS_TABLE = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    migration_name  TEXT PRIMARY KEY,
    applied_at      TEXT NOT NULL   -- ISO 8601
);
"""


class Database:
    """Async SQLite database wrapper with integrated migration support.

    Usage (context manager)::

        async with Database(":memory:") as db:
            await db.execute("INSERT INTO ...")
            rows = await db.fetchall("SELECT ...")

    Usage (manual)::

        db = Database("path/to/fleet.db")
        await db.connect()
        try:
            ...
        finally:
            await db.close()
    """

    def __init__(
        self,
        db_path: str | Path = ":memory:",
        migrations_dir: str | Path | None = None,
    ) -> None:
        self._db_path = str(db_path)
        self._migrations_dir = (
            Path(migrations_dir) if migrations_dir is not None else _DEFAULT_MIGRATIONS_DIR
        )
        self._conn: aiosqlite.Connection | None = None

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Open the SQLite connection and run pending migrations."""
        if self._conn is not None:
            return  # already connected

        self._conn = await aiosqlite.connect(self._db_path)
        # Enable WAL mode for better concurrent read performance.
        await self._conn.execute("PRAGMA journal_mode=WAL")
        # Enforce foreign-key constraints.
        await self._conn.execute("PRAGMA foreign_keys=ON")
        await self._conn.commit()

        await self._run_migrations()

    async def close(self) -> None:
        """Close the SQLite connection."""
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    # ------------------------------------------------------------------
    # Async context manager support
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "Database":
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    async def execute(
        self,
        sql: str,
        parameters: Sequence[Any] | None = None,
    ) -> aiosqlite.Cursor:
        """Execute a single SQL statement and return the cursor.

        The caller is responsible for committing if the statement is a DML
        operation (INSERT / UPDATE / DELETE).  For convenience, use
        :meth:`execute_and_commit` when an immediate commit is desired.
        """
        self._require_connected()
        params = parameters if parameters is not None else ()
        return await self._conn.execute(sql, params)  # type: ignore[union-attr]

    async def execute_and_commit(
        self,
        sql: str,
        parameters: Sequence[Any] | None = None,
    ) -> aiosqlite.Cursor:
        """Execute a single SQL statement and immediately commit."""
        cursor = await self.execute(sql, parameters)
        await self._conn.commit()  # type: ignore[union-attr]
        return cursor

    async def fetchall(
        self,
        sql: str,
        parameters: Sequence[Any] | None = None,
    ) -> list[aiosqlite.Row]:
        """Execute a SELECT statement and return all rows."""
        cursor = await self.execute(sql, parameters)
        return await cursor.fetchall()

    async def fetchone(
        self,
        sql: str,
        parameters: Sequence[Any] | None = None,
    ) -> aiosqlite.Row | None:
        """Execute a SELECT statement and return the first row, or ``None``."""
        cursor = await self.execute(sql, parameters)
        return await cursor.fetchone()

    # ------------------------------------------------------------------
    # Transaction helpers
    # ------------------------------------------------------------------

    async def commit(self) -> None:
        """Commit the current transaction."""
        self._require_connected()
        await self._conn.commit()  # type: ignore[union-attr]

    async def rollback(self) -> None:
        """Roll back the current transaction."""
        self._require_connected()
        await self._conn.rollback()  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # Migration runner
    # ------------------------------------------------------------------

    async def _run_migrations(self) -> None:
        """Discover and apply pending migrations idempotently.

        Steps:
        1. Ensure the ``schema_migrations`` tracking table exists.
        2. Collect ``*.sql`` files from ``migrations_dir`` in lexicographic order.
        3. For each file not already recorded in ``schema_migrations``, execute
           the SQL and insert a tracking row — all within a single transaction.
        """
        self._require_connected()

        # Ensure the tracking table exists.
        await self._conn.executescript(_CREATE_MIGRATIONS_TABLE)  # type: ignore[union-attr]
        await self._conn.commit()  # type: ignore[union-attr]

        # Collect migration files in lexicographic order.
        migration_files = sorted(self._migrations_dir.glob("*.sql"))

        if not migration_files:
            logger.debug("No migration files found in %s", self._migrations_dir)
            return

        # Fetch already-applied migrations.
        applied: set[str] = set()
        async with self._conn.execute(  # type: ignore[union-attr]
            "SELECT migration_name FROM schema_migrations"
        ) as cursor:
            rows = await cursor.fetchall()
            applied = {row[0] for row in rows}

        for migration_path in migration_files:
            name = migration_path.name
            if name in applied:
                logger.debug("Migration already applied, skipping: %s", name)
                continue

            logger.info("Applying migration: %s", name)
            sql = migration_path.read_text(encoding="utf-8")

            try:
                # executescript implicitly commits any pending transaction and
                # runs the SQL outside of a user transaction, so we record the
                # migration name in a separate statement afterwards.
                await self._conn.executescript(sql)  # type: ignore[union-attr]

                # Record the migration as applied.
                from datetime import datetime, timezone

                applied_at = datetime.now(timezone.utc).isoformat()
                await self._conn.execute(  # type: ignore[union-attr]
                    "INSERT INTO schema_migrations (migration_name, applied_at) VALUES (?, ?)",
                    (name, applied_at),
                )
                await self._conn.commit()  # type: ignore[union-attr]
                logger.info("Migration applied successfully: %s", name)
            except Exception:
                logger.exception("Failed to apply migration: %s", name)
                raise

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_connected(self) -> None:
        """Raise ``RuntimeError`` if the database is not connected."""
        if self._conn is None:
            raise RuntimeError(
                "Database is not connected. Call connect() or use 'async with Database(...)' first."
            )

    @property
    def connection(self) -> aiosqlite.Connection:
        """Return the underlying ``aiosqlite`` connection (for advanced use)."""
        self._require_connected()
        return self._conn  # type: ignore[return-value]
