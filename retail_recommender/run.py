"""
Entry Point — Jalankan: python run.py
"""

import os
import sys
import logging

# Tambahkan root project ke sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from retail_recommender.app import create_app
from retail_recommender.app.model_manager import initialize_models

logger = logging.getLogger(__name__)


def main():
    app = create_app()

    with app.app_context():
        logger.info("Memulai inisialisasi model (proses ini mungkin memakan 1-3 menit)...")
        result = initialize_models(app.config)
        if result["status"] != "success":
            logger.error(f"Gagal inisialisasi model: {result.get('message')}")
            logger.warning("Server tetap berjalan, tapi endpoint rekomendasi belum aktif.")

    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "True").lower() == "true"

    logger.info(f"Server berjalan di http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=debug, use_reloader=False)


if __name__ == "__main__":
    main()
