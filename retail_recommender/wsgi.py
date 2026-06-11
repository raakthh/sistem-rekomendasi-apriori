from retail_recommender.app import create_app
from retail_recommender.app.model_manager import initialize_models

app = create_app()

with app.app_context():
    initialize_models(app.config)