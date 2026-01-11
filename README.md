# RAG Tool - Document & Chunking Pipeline

RAG Tool system designed with extensible architecture supporting multi-step pipeline.

## 🚀 Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Environment Configuration (Optional)

Create `.env` file from `.env.example` template:

```bash
cp .env.example .env
```

Then edit `.env` file according to your needs:
- `OLLAMA_BASE_URL`: Ollama server URL (default: http://localhost:11434)
- `OLLAMA_EMBEDDING_MODEL`: Embedding model for semantic chunking (default: nomic-embed-text)
- `OLLAMA_LLM_MODEL`: LLM model for later steps (default: llama3.2:3b)
- `DATABASE_PATH`: Database path (leave empty to use default)
- `DATA_DIR`: Directory to store documents (leave empty to use default: ./data)
- `LOG_FILE`: Log file path (leave empty to use default: ./logs/app.log)
- `DEFAULT_CHUNK_SIZE`: Default chunk size (default: 500)
- `DEFAULT_CHUNK_OVERLAP`: Default overlap (default: 50)

**Note:** If you don't create `.env` file, the system will use default values.

### 3. Run Application

```bash
python app.py
```

### 4. Access

Open browser and access: `http://localhost:5000`

## 📁 Project Structure

```
OverviewSystemRetrieval/
├── app.py                 # Flask application entry point
├── config.py              # System configuration
├── database.py            # Database layer (SQLite)
├── models.py              # Data models (Document, Chunk)
├── routes.py              # Flask routes and API endpoints
├── requirements.txt       # Python dependencies
├── README.md             # This documentation
├── data/                 # Directory to store documents
│   ├── ve_sinh_rang_mieng.md
│   ├── dieu_tri_chinh_nha.md
│   ├── dieu_tri_tuy_rang.md
│   ├── cay_ghep_rang.md
│   └── nha_khoa_phong_ngua.md
├── services/             # Business logic layer
│   ├── __init__.py
│   ├── document_service.py    # Document management
│   └── chunking_service.py     # Chunking strategies
├── templates/            # HTML templates
│   └── index.html       # Single-page UI
└── logs/                 # Log files
    └── app.log
```

## 🏗️ Architecture

The system is designed with **Clean Architecture** with the following layers:

### Storage Layer
- `database.py`: SQLite database management
- `models.py`: Data models (Document, Chunk)

### Services Layer
- `document_service.py`: Handle upload, list, discover documents
- `chunking_service.py`: Implement chunking strategies

### Routes Layer
- `routes.py`: Flask routes and API endpoints

### Templates Layer
- `templates/index.html`: Single-page HTML UI

## ✨ MVP Features

### Step 1: Document Management

1. **Upload Documents**
   - Upload `.md` or `.txt` files
   - Paste text and save as file

2. **Document Listing**
   - Display list of documents with information:
     - Filename
     - Number of lines
     - Number of characters
     - File size
   - Search by filename
   - Refresh/Rescan to discover new files

3. **Select Documents**
   - Select one or multiple documents using checkboxes
   - View document content (modal)
   - Move to Step 2 for chunking

### Step 2: Chunking

1. **Chunking Strategies**
   - **Fixed Size**: Split by fixed size with overlap
   - **Markdown Header**: Split by markdown headers (# ## ###)
   - **Recursive**: Recursive splitting by separators
   - **Paragraph-based**: Split by paragraphs
   - **Sliding Window**: Split with sliding window
   - **Semantic**: Split based on semantic similarity using embeddings

2. **Parameters**
   - Each strategy has its own parameters
   - UI automatically updates based on selected strategy

3. **Preview & Statistics**
   - Preview first 5-10 chunks
   - Statistics: total_chunks, avg_len, min_len, max_len
   - Filter by document
   - Expand/collapse to view full text

### Placeholder Steps (Coming Soon)

- Step 3: Embeddings
- Step 4: UMAP Visualization
- Step 5: Retrieval Test
- Step 6: RAGAS Evaluation

## 🔧 Configuration

Configuration can be changed in `config.py` or `.env` file:

- `DATA_DIR`: Directory to store documents
- `DATABASE_PATH`: Database path
- `ALLOWED_EXTENSIONS`: Allowed file extensions
- `DEFAULT_CHUNK_SIZE`: Default chunk size
- `DEFAULT_CHUNK_OVERLAP`: Default overlap

## 📊 Database Schema

### Documents Table
- `doc_id`: Primary key
- `filename`: File name
- `filepath`: File path
- `num_lines`: Number of lines
- `num_chars`: Number of characters
- `file_size`: File size (bytes)
- `created_at`: Creation timestamp
- `updated_at`: Update timestamp

### Chunks Table
- `chunk_id`: Primary key
- `doc_id`: Foreign key to documents
- `strategy`: Strategy name used
- `params_json`: Parameters in JSON format
- `position`: Chunk position in document
- `text`: Chunk content
- `len_chars`: Chunk length (characters)
- `created_at`: Creation timestamp

## 🎨 UI Features

- **Single Page Application**: All features on one page
- **Pipeline View**: Steps displayed from top to bottom
- **Responsive Design**: Beautiful, easy-to-use interface
- **Real-time Updates**: Data updates without reload
- **Modal View**: View document content in modal
- **Chunk Preview**: Preview chunks with expand/collapse

## 📝 API Endpoints

### Documents
- `GET /api/documents` - Get list of documents
- `POST /api/documents/upload` - Upload file
- `POST /api/documents/paste` - Paste text
- `POST /api/documents/discover` - Discover files
- `GET /api/documents/<doc_id>/content` - Get document content

### Chunking
- `GET /api/chunking/strategies` - Get list of strategies
- `POST /api/chunking/run` - Run chunking
- `GET /api/chunks` - Get chunks with pagination

## 🛠️ Development

### Adding New Chunking Strategy

1. Add method in `services/chunking_service.py`:
```python
@staticmethod
def new_strategy_chunk(text: str, param1: int, param2: str) -> List[str]:
    # Implementation
    pass
```

2. Add to `chunk_document()` method:
```python
elif strategy == 'new_strategy':
    param1 = params.get('param1', default_value)
    chunks_text = ChunkingService.new_strategy_chunk(content, param1, ...)
```

3. Add to API response in `routes.py`:
```python
'new_strategy': {
    'name': 'New Strategy',
    'description': 'Description',
    'params': {
        'param1': {'type': 'number', 'default': 100, 'label': 'Param 1'}
    }
}
```

### Adding New Step

1. Add HTML section in `templates/index.html`
2. Add service in `services/` if needed
3. Add routes in `routes.py`
4. Update UI JavaScript to handle new step

## 📄 License

MIT License

## 👥 Author

Senior RAG Engineer
