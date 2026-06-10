from flask import Blueprint, jsonify, current_app
from retail_recommender.app.model_manager import get_model_state, get_engine, get_preprocessor, initialize_models

health_bp = Blueprint("health", __name__)


@health_bp.route("/health", methods=["GET"])
def health():
    state = get_model_state()
    engine = get_engine()
    preprocessor = get_preprocessor()

    payload = {
        "status": "ok" if state["is_ready"] else "initializing",
        "model_ready": state["is_ready"],
        "error": state.get("error"),
    }

    if state["is_ready"] and engine and preprocessor:
        payload["model_summary"] = engine.get_summary()
        payload["dataset_stats"] = preprocessor.get_stats()

    code = 200 if state["is_ready"] else 503
    return jsonify(payload), code


@health_bp.route("/health/reload", methods=["POST"])
def reload_model():
    """Force-reload model (berguna saat dataset diperbarui)."""
    result = initialize_models(current_app.config)
    code = 200 if result["status"] == "success" else 500
    return jsonify(result), code
