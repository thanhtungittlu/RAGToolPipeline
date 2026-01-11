"""
RAG Tool System Configuration
Supports reading from .env file or using default values
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).parent.absolute()

# Data directory - stores documents
DATA_DIR_STR = os.getenv('DATA_DIR', '').strip()
if DATA_DIR_STR:
    DATA_DIR = Path(DATA_DIR_STR).resolve()
else:
    DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Database
DATABASE_PATH_STR = os.getenv('DATABASE_PATH', '').strip()
if DATABASE_PATH_STR:
    DATABASE_PATH = Path(DATABASE_PATH_STR).resolve()
else:
    DATABASE_PATH = BASE_DIR / "rag_tool.db"

# Allowed file extensions
ALLOWED_EXTENSIONS = {'.md', '.txt'}

# Chunking defaults
DEFAULT_CHUNK_SIZE = int(os.getenv('DEFAULT_CHUNK_SIZE', '500'))
DEFAULT_CHUNK_OVERLAP = int(os.getenv('DEFAULT_CHUNK_OVERLAP', '50'))

# Ollama configuration
# Read from .env or use default values
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
OLLAMA_EMBEDDING_MODEL = os.getenv('OLLAMA_EMBEDDING_MODEL', 'nomic-embed-text')
OLLAMA_LLM_MODEL = os.getenv('OLLAMA_LLM_MODEL', 'llama3.2:3b')

# Backward compatibility
OLLAMA_MODEL = OLLAMA_EMBEDDING_MODEL
