"""
Flask Application Factory
"""

import os
import logging
from flask import Flask
from flask_cors import CORS
from dotenv import load_dotenv
from flask import send_from_directory

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
    DATASET_PATH=os.getenv(
        "DATASET_PATH",
        "data/online_retail_sample_2000_invoice.xlsx"
    ),
    DATASET_SHEET=os.getenv(
        "DATASET_SHEET",
        "Sheet1"
    ),
    MIN_SUPPORT=float(os.getenv("MIN_SUPPORT", 0.002)),
    MIN_CONFIDENCE=float(os.getenv("MIN_CONFIDENCE", 0.05)),
    MIN_LIFT=float(os.getenv("MIN_LIFT", 0.8)),
    MAX_LEN=int(os.getenv("MAX_LEN", 2)),
)

    if config:
        app.config.update(config)

    # Register blueprints
    from retail_recommender.app.routes.recommend import recommend_bp
    from retail_recommender.app.routes.products import products_bp
    from retail_recommender.app.routes.analytics import analytics_bp
    from retail_recommender.app.routes.health import health_bp

    app.register_blueprint(health_bp, url_prefix="/api")
    app.register_blueprint(recommend_bp, url_prefix="/api/recommend")
    app.register_blueprint(products_bp, url_prefix="/api/products")
    app.register_blueprint(analytics_bp, url_prefix="/api/analytics")

    @app.route("/")
    def home():
        return send_from_directory(
            os.path.join(os.path.dirname(__file__), "../frontend"),
            "index.html"
        )


    return app


app = create_app()