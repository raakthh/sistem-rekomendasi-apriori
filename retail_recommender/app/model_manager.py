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
            dataset_path = os.path.join(base_dir, app_config.get("DATASET_PATH", "data/online_retail_sample_20k.xlsx"))

            logger.info("=== Memulai Inisialisasi Model ===")

            # Pipeline Preprocessing
            preprocessor = DataPreprocessor(
                filepath=dataset_path,
                sheet_name=app_config.get("DATASET_SHEET", "Sheet1"),
            )
            preprocessor.run_pipeline()

            # ========== PERBAIKAN PARAMETER APRIORI ==========
            # Ambil dari config, tapi jika tidak ada atau nilai tidak tepat, gunakan default yang benar
            min_support = app_config.get("MIN_SUPPORT", 0.0005)   # turunkan dari 0.001
            min_confidence = app_config.get("MIN_CONFIDENCE", 0.05)
            min_lift = app_config.get("MIN_LIFT", 0.8)
            max_len = app_config.get("MAX_LEN", 3)                # PASTIKAN 3, BUKAN 2!

            # Pastikan max_len minimal 3 agar bisa membentuk rules
            if max_len < 2:
                logger.warning(f"MAX_LEN={max_len} terlalu kecil, diubah menjadi 3")
                max_len = 2

            logger.info(f"Parameter Apriori: min_support={min_support}, min_confidence={min_confidence}, min_lift={min_lift}, max_len={max_len}")

            engine = AprioriEngine(
                min_support=min_support,
                min_confidence=min_confidence,
                min_lift=min_lift,
                max_len=max_len,
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