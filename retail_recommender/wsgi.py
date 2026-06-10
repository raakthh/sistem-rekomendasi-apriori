from retail_recommender.app import app
from retail_recommender.app.model_manager import initialize_models

with app.app_context():
    initialize_models(app.config)