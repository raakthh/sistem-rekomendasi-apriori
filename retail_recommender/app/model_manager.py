"""
Model Manager — Singleton yang menyimpan state preprocessor & engine
agar tidak di-reload setiap request.
"""

import os
import logging
from threading import Lock
from retail_recommender.utils.preprocessor import DataPreprocessor
from retail_recommender.models.apriori_engine import AprioriEngine

logger = logging.getLogger(__name__)

_lock = Lock()
_preprocessor: DataPreprocessor = None
_engine: AprioriEngine = None
_is_ready = False
_init_error = None


def get_model_state():
    return {
        "is_ready": _is_ready,
        "error": _init_error,
    }


def get_preprocessor() -> DataPreprocessor:
    return _preprocessor


def get_engine() -> AprioriEngine:
    return _engine


def initialize_models(app_config: dict) -> dict:
    """
    Inisialisasi preprocessing + Apriori pipeline.
    Thread-safe. Dipanggil sekali saat startup atau via /api/health/reload.
    """
    global _preprocessor, _engine, _is_ready, _init_error

    with _lock:
        _is_ready = False
        _init_error = None

        try:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            dataset_path = os.path.join(base_dir, app_config["DATASET_PATH"])

            logger.info("=== Memulai Inisialisasi Model ===")

            # Pipeline Preprocessing
            preprocessor = DataPreprocessor(
                filepath=dataset_path,
                sheet_name=app_config["DATASET_SHEET"],
            )
            preprocessor.run_pipeline()

            # Pipeline Apriori
            engine = AprioriEngine(
                min_support=app_config["MIN_SUPPORT"],
                min_confidence=app_config["MIN_CONFIDENCE"],
                min_lift=app_config["MIN_LIFT"],
                max_len=app_config["MAX_LEN"],
            )
            engine.run_pipeline(preprocessor.transaction_df)

            _preprocessor = preprocessor
            _engine = engine
            _is_ready = True

            summary = engine.get_summary()
            stats = preprocessor.get_stats()
            logger.info("=== Model Siap ===")
            logger.info(
                f"Rules: {summary['rules_count']:,} | "
                f"Itemsets: {summary['frequent_itemsets_count']:,} | "
                f"Transaksi: {stats.get('total_transactions', 0):,}"
            )
            return {"status": "success", "summary": summary, "stats": stats}

        except Exception as e:
            _init_error = str(e)
            logger.exception(f"Gagal inisialisasi model: {e}")
            return {"status": "error", "message": str(e)}
