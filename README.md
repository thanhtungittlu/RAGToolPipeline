# RAG Tool - Document & Chunking Pipeline

Hệ thống RAG Tool được thiết kế theo kiến trúc mở rộng với pipeline nhiều bước. 
## 🚀 Hướng Dẫn Chạy

### 1. Cài Đặt Dependencies

```bash
pip install -r requirements.txt
```

### 2. Cấu Hình Môi Trường (Optional)

Tạo file `.env` từ file mẫu `.env.example`:

```bash
cp .env.example .env
```

Sau đó chỉnh sửa file `.env` theo nhu cầu của bạn:
- `OLLAMA_BASE_URL`: URL của Ollama server (mặc định: http://localhost:11434)
- `OLLAMA_EMBEDDING_MODEL`: Model embedding cho semantic chunking (mặc định: nomic-embed-text)
- `OLLAMA_LLM_MODEL`: Model LLM cho các step sau (mặc định: llama3.2:3b)
- `DATABASE_PATH`: Đường dẫn database (để trống để dùng mặc định)
- `DATA_DIR`: Thư mục lưu documents (để trống để dùng mặc định: ./data)
- `LOG_FILE`: Đường dẫn file log (để trống để dùng mặc định: ./logs/app.log)
- `DEFAULT_CHUNK_SIZE`: Kích thước chunk mặc định (mặc định: 500)
- `DEFAULT_CHUNK_OVERLAP`: Overlap mặc định (mặc định: 50)

**Lưu ý:** Nếu không tạo file `.env`, hệ thống sẽ sử dụng các giá trị mặc định.

### 3. Chạy Ứng Dụng

```bash
python app.py
```

### 4. Truy Cập

Mở browser và truy cập: `http://localhost:5000`

## 📁 Cấu Trúc Dự Án

```
OverviewSystemRetrieval/
├── app.py                 # Flask application entry point
├── config.py              # Cấu hình hệ thống
├── database.py            # Database layer (SQLite)
├── models.py              # Data models (Document, Chunk)
├── routes.py              # Flask routes và API endpoints
├── requirements.txt       # Python dependencies
├── README.md             # Tài liệu này
├── data/                 # Thư mục lưu trữ documents
│   ├── dental_hygiene.md
│   ├── orthodontic_treatment.md
│   ├── root_canal_treatment.md
│   ├── dental_implants.md
│   └── preventive_dentistry.md
├── services/             # Business logic layer
│   ├── __init__.py
│   ├── document_service.py    # Document management
│   └── chunking_service.py     # Chunking strategies
├── templates/            # HTML templates
│   └── index.html       # Single-page UI
└── logs/                 # Log files
    └── app.log
```

## 🏗️ Kiến Trúc

Hệ thống được thiết kế theo **Clean Architecture** với các lớp:

### Storage Layer
- `database.py`: Quản lý SQLite database
- `models.py`: Data models (Document, Chunk)

### Services Layer
- `document_service.py`: Xử lý upload, list, discover documents
- `chunking_service.py`: Implement các chunking strategies

### Routes Layer
- `routes.py`: Flask routes và API endpoints

### Templates Layer
- `templates/index.html`: Single-page HTML UI

## ✨ Tính Năng MVP

### Step 1: Document Management

1. **Upload Documents**
   - Upload file `.md` hoặc `.txt`
   - Paste text và lưu thành file

2. **Document Listing**
   - Hiển thị danh sách documents với thông tin:
     - Filename
     - Số dòng (lines)
     - Số ký tự (characters)
     - Kích thước file
   - Search theo filename
   - Refresh/Rescan để discover files mới

3. **Select Documents**
   - Chọn một hoặc nhiều documents bằng checkbox
   - Xem nội dung document (modal)
   - Chuyển sang Step 2 để chunking

### Step 2: Chunking

1. **Chunking Strategies**
   - **Fixed Size**: Chia theo kích thước cố định với overlap
   - **Markdown Header**: Chia theo markdown headers (# ## ###)
   - **Recursive**: Chia đệ quy theo separators
   - **Paragraph-based**: Chia theo paragraphs
   - **Sliding Window**: Chia với sliding window
   - **Semantic**: Chia theo câu và đoạn văn

2. **Parameters**
   - Mỗi strategy có parameters riêng
   - UI tự động cập nhật theo strategy được chọn

3. **Preview & Statistics**
   - Preview 5-10 chunks đầu tiên
   - Statistics: total_chunks, avg_len, min_len, max_len
   - Filter theo document
   - Expand/collapse để xem full text

### Placeholder Steps (Coming Soon)

- Step 3: Embeddings
- Step 4: UMAP Visualization
- Step 5: Retrieval Test
- Step 6: RAGAS Evaluation

## 🔧 Cấu Hình

Các cấu hình có thể thay đổi trong `config.py`:

- `DATA_DIR`: Thư mục lưu trữ documents
- `DATABASE_PATH`: Đường dẫn database
- `ALLOWED_EXTENSIONS`: Các file extension được phép
- `DEFAULT_CHUNK_SIZE`: Kích thước chunk mặc định
- `DEFAULT_CHUNK_OVERLAP`: Overlap mặc định

## 📊 Database Schema

### Documents Table
- `doc_id`: Primary key
- `filename`: Tên file
- `filepath`: Đường dẫn file
- `num_lines`: Số dòng
- `num_chars`: Số ký tự
- `file_size`: Kích thước file (bytes)
- `created_at`: Thời gian tạo
- `updated_at`: Thời gian cập nhật

### Chunks Table
- `chunk_id`: Primary key
- `doc_id`: Foreign key đến documents
- `strategy`: Tên strategy được sử dụng
- `params_json`: Parameters dạng JSON
- `position`: Vị trí chunk trong document
- `text`: Nội dung chunk
- `len_chars`: Độ dài chunk (ký tự)
- `created_at`: Thời gian tạo

## 🎨 UI Features

- **Single Page Application**: Tất cả tính năng trên một trang
- **Pipeline View**: Các bước được hiển thị từ trên xuống
- **Responsive Design**: Giao diện đẹp, dễ sử dụng
- **Real-time Updates**: Cập nhật dữ liệu không cần reload
- **Modal View**: Xem nội dung document trong modal
- **Chunk Preview**: Preview chunks với expand/collapse

## 📝 API Endpoints

### Documents
- `GET /api/documents` - Lấy danh sách documents
- `POST /api/documents/upload` - Upload file
- `POST /api/documents/paste` - Paste text
- `POST /api/documents/discover` - Discover files
- `GET /api/documents/<doc_id>/content` - Lấy nội dung document

### Chunking
- `GET /api/chunking/strategies` - Lấy danh sách strategies
- `POST /api/chunking/run` - Chạy chunking
- `GET /api/chunks` - Lấy chunks với pagination

## 🛠️ Development

### Thêm Chunking Strategy Mới

1. Thêm method trong `services/chunking_service.py`:
```python
@staticmethod
def new_strategy_chunk(text: str, param1: int, param2: str) -> List[str]:
    # Implementation
    pass
```

2. Thêm vào `chunk_document()` method:
```python
elif strategy == 'new_strategy':
    param1 = params.get('param1', default_value)
    chunks_text = ChunkingService.new_strategy_chunk(content, param1, ...)
```

3. Thêm vào API response trong `routes.py`:
```python
'new_strategy': {
    'name': 'New Strategy',
    'description': 'Description',
    'params': {
        'param1': {'type': 'number', 'default': 100, 'label': 'Param 1'}
    }
}
```

### Thêm Step Mới

1. Thêm HTML section trong `templates/index.html`
2. Thêm service trong `services/` nếu cần
3. Thêm routes trong `routes.py`
4. Update UI JavaScript để handle step mới

## 📄 License

MIT License

## 👥 Author

Senior RAG Engineer
