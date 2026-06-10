"""
/api/recommend — Endpoint utama rekomendasi produk
"""

from flask import Blueprint, request, jsonify
from retail_recommender.app.model_manager import get_model_state, get_engine, get_preprocessor

recommend_bp = Blueprint("recommend", __name__)


def _check_ready():
    state = get_model_state()
    if not state["is_ready"]:
        return (
            jsonify(
                {
                    "error": "Model belum siap. Tunggu beberapa saat atau cek /api/health.",
                    "model_ready": False,
                }
            ),
            503,
        )
    return None


# ------------------------------------------------------------------ #
#  POST /api/recommend/products                                        #
# ------------------------------------------------------------------ #
@recommend_bp.route("/products", methods=["POST"])
def recommend_products():
    """
    Rekomendasi produk berdasarkan item di keranjang / yang sedang dilihat.

    Body JSON:
        {
            "products": ["PRODUCT NAME 1", "PRODUCT NAME 2"],
            "top_n": 10
        }

    Response:
        {
            "query_products": [...],
            "recommendations": [
                {
                    "product": "...",
                    "confidence": 0.45,
                    "lift": 3.2,
                    "support": 0.03,
                    "rule_strength": 1.44,
                    "reason": "Sering dibeli bersama"
                }
            ],
            "total": 8
        }
    """
    err = _check_ready()
    if err:
        return err

    body = request.get_json(silent=True) or {}
    products = body.get("products", [])
    top_n = min(int(body.get("top_n", 10)), 50)

    if not products or not isinstance(products, list):
        return jsonify({"error": "Field 'products' harus berupa list string"}), 400

    engine = get_engine()
    recs = engine.get_recommendations(products, top_n=top_n)

    # Tambahkan label alasan untuk frontend
    for r in recs:
        if r["lift"] >= 5:
            r["reason"] = "Sangat sering dibeli bersama"
        elif r["lift"] >= 2:
            r["reason"] = "Sering dibeli bersama"
        else:
            r["reason"] = "Juga dibeli bersama"

    return jsonify(
        {
            "query_products": products,
            "recommendations": recs,
            "total": len(recs),
        }
    )


# ------------------------------------------------------------------ #
#  GET /api/recommend/combos                                          #
# ------------------------------------------------------------------ #
@recommend_bp.route("/combos", methods=["GET"])
def popular_combos():
    """
    Produk yang paling sering dibeli bersama (by support).

    Query params:
        top_n (int, default=20)
    """
    err = _check_ready()
    if err:
        return err

    top_n = min(int(request.args.get("top_n", 20)), 100)
    engine = get_engine()
    combos = engine.get_popular_combos(top_n=top_n)

    return jsonify({"popular_combos": combos, "total": len(combos)})


# ------------------------------------------------------------------ #
#  GET /api/recommend/rules                                           #
# ------------------------------------------------------------------ #
@recommend_bp.route("/rules", methods=["GET"])
def top_rules():
    """
    Top association rules terurut by lift.

    Query params:
        top_n (int, default=50)
        min_lift (float, default=1.0)
        min_confidence (float, default=0.0)
    """
    err = _check_ready()
    if err:
        return err

    top_n = min(int(request.args.get("top_n", 50)), 200)
    min_lift = float(request.args.get("min_lift", 1.0))
    min_conf = float(request.args.get("min_confidence", 0.0))

    engine = get_engine()
    rules = engine.get_top_rules(top_n=top_n * 3)  # Ambil lebih, lalu filter

    filtered = [
        r for r in rules
        if r["lift"] >= min_lift and r["confidence"] >= min_conf
    ][:top_n]

    return jsonify({"rules": filtered, "total": len(filtered)})


# ------------------------------------------------------------------ #
#  GET /api/recommend/similar/<product_name>                          #
# ------------------------------------------------------------------ #
@recommend_bp.route("/similar/<path:product_name>", methods=["GET"])
def similar_products(product_name: str):
    """
    Shortcut GET untuk rekomendasi satu produk.
    Contoh: GET /api/recommend/similar/WHITE HANGING HEART T-LIGHT HOLDER
    """
    err = _check_ready()
    if err:
        return err

    top_n = min(int(request.args.get("top_n", 10)), 50)
    engine = get_engine()
    recs = engine.get_recommendations([product_name.upper()], top_n=top_n)

    for r in recs:
        r["reason"] = (
            "Sangat sering dibeli bersama" if r["lift"] >= 5
            else "Sering dibeli bersama" if r["lift"] >= 2
            else "Juga dibeli bersama"
        )

    return jsonify(
        {
            "query_products": [product_name.upper()],
            "recommendations": recs,
            "total": len(recs),
        }
    )
