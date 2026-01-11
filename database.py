"""
Database layer - SQLite with SQLAlchemy-style queries
Uses sqlite3 stdlib to avoid additional dependencies
"""
import sqlite3
import json
from datetime import datetime
from pathlib import Path
from config import DATABASE_PATH
import logging

logger = logging.getLogger(__name__)

def get_db_connection():
    """Create connection to SQLite database"""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize database and create necessary tables"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Documents table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            doc_id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL UNIQUE,
            filepath TEXT NOT NULL,
            num_lines INTEGER DEFAULT 0,
            num_chars INTEGER DEFAULT 0,
            file_size INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Chunks table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id INTEGER NOT NULL,
            strategy TEXT NOT NULL,
            params_json TEXT,
            position INTEGER NOT NULL,
            text TEXT NOT NULL,
            len_chars INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
        )
    ''')
    
    # Indexes to improve query performance
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_chunks_strategy ON chunks(strategy)')
    
    conn.commit()
    conn.close()
    logger.info("Database initialized successfully")

def execute_query(query, params=(), fetch_one=False, fetch_all=False):
    """Helper function to execute queries"""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(query, params)
        if fetch_one:
            result = cursor.fetchone()
        elif fetch_all:
            result = cursor.fetchall()
        else:
            result = cursor.lastrowid
        conn.commit()
        return result
    except Exception as e:
        conn.rollback()
        logger.error(f"Database error: {e}")
        raise
    finally:
        conn.close()
