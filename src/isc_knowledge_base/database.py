"""PostgreSQL connections shared by the command-line tools."""

import os
from pathlib import Path

from dotenv import load_dotenv
import psycopg


def connect_postgres() -> psycopg.Connection:
    load_dotenv(Path.cwd() / ".env")
    return psycopg.connect(
        host=os.getenv("POSTGRES_HOST") or "127.0.0.1",
        port=os.getenv("POSTGRES_PORT") or "5432",
        dbname=os.getenv("POSTGRES_DB") or "isc_knowledge_base",
        user=os.getenv("POSTGRES_USER") or "isc_knowledge_base",
        password=os.getenv("POSTGRES_PASSWORD") or "isc_knowledge_base_dev",
        autocommit=True,
    )
