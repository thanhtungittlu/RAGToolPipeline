"""
RAG Tool - Flask Application
MVP: Document Management + Chunking Pipeline

Hướng dẫn chạy:
1. Cài đặt dependencies: pip install -r requirements.txt
2. Chạy app: python app.py
3. Mở browser: http://localhost:5000

Kiến trúc:
- Storage: models.py, database.py
- Services: document_service.py, chunking_service.py
- Routes: routes.py
- Templates: templates/index.html
"""

from flask import Flask
from routes import register_routes
from database import init_db
import logging
from pathlib import Path
from config import LOG_FILE, DATA_DIR

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def create_app():
    """Factory function để tạo Flask app"""
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'dev-secret-key-change-in-production'
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
    
    # Initialize database
    init_db()
    
    # Register routes
    register_routes(app)
    
    # Ensure data directory exists
    DATA_DIR.mkdir(exist_ok=True)
    
    logger.info("RAG Tool application initialized")
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000)
