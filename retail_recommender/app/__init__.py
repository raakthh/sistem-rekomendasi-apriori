"""
Flask Application Factory
"""

import os
import logging
from flask import Flask
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def create_app(config: dict = None) -> Flask:
    app = Flask(__name__)
    CORS(app)

    # Default config
    app.config.update(
        SECRET_KEY=os.getenv("SECRET_KEY", "dev-secret-key"),
        DATASET_PATH=os.getenv("DATASET_PATH", "data/online_retail_II.xlsx"),
        DATASET_SHEET=os.getenv("DATASET_SHEET", "Year 2010-2011"),
        MIN_SUPPORT=float(os.getenv("MIN_SUPPORT", 0.02)),
        MIN_CONFIDENCE=float(os.getenv("MIN_CONFIDENCE", 0.3)),
        MIN_LIFT=float(os.getenv("MIN_LIFT", 1.0)),
        MAX_LEN=int(os.getenv("MAX_LEN", 4)),
    )

    if config:
        app.config.update(config)

    # Register blueprints
    from app.routes.recommend import recommend_bp
    from app.routes.products import products_bp
    from app.routes.analytics import analytics_bp
    from app.routes.health import health_bp

    app.register_blueprint(health_bp, url_prefix="/api")
    app.register_blueprint(recommend_bp, url_prefix="/api/recommend")
    app.register_blueprint(products_bp, url_prefix="/api/products")
    app.register_blueprint(analytics_bp, url_prefix="/api/analytics")

    return app

app = create_app()
