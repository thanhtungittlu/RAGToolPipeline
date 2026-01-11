"""
RAG Tool - Flask Application
MVP: Document Management + Chunking Pipeline

Setup Instructions:
1. Install dependencies: pip install -r requirements.txt
2. Run app: python app.py
3. Open browser: http://localhost:5000

Architecture:
- Storage: models.py (data models only, no database)
- Services: document_service.py, chunking_service.py
- Routes: routes.py
- Templates: templates/index.html
"""

from flask import Flask
from routes import register_routes
import logging
from pathlib import Path
from config import DATA_DIR

# Setup logging - console only
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def create_app():
    """Factory function to create Flask app"""
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'dev-secret-key-change-in-production'
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
    
    # Register routes
    register_routes(app)
    
    # Ensure data directory exists
    DATA_DIR.mkdir(exist_ok=True)
    
    logger.info("RAG Tool application initialized")
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000)
