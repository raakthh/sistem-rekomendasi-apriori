"""
Apriori Recommendation Engine
Pipeline: Itemset → Apriori → Support/Confidence/Lift → Association Rules → Ranked Recommendations
"""

import pandas as pd
import numpy as np
import logging
import json
import os
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

logger = logging.getLogger(__name__)


class AprioriEngine:
    """
    Mengimplementasikan algoritma Apriori untuk membentuk
    association rules dan menghasilkan rekomendasi produk.
    """

    def __init__(
        self,
        min_support: float = 0.001,      # cocok untuk 8k transaksi
        min_confidence: float = 0.05,    # lebih longgar
        min_lift: float = 0.5,
        max_len: int = 2,                # lebih ringan
    ):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.min_lift = min_lift
        self.max_len = max_len

        self.frequent_itemsets = None
        self.rules = None
        self._rules_cache = {}

    # ------------------------------------------------------------------ #
    #  STEP 4 – Frequent Itemsets (Apriori)                               #
    # ------------------------------------------------------------------ #
    def find_frequent_itemsets(self, transaction_df: pd.DataFrame) -> pd.DataFrame:
        """
        Temukan frequent itemsets menggunakan algoritma Apriori.

        Args:
            transaction_df: One-hot encoded transaction matrix (Invoice × Item)
        Returns:
            DataFrame frequent itemsets dengan kolom support & itemsets
        """
        logger.info(
            f"Menjalankan Apriori — min_support={self.min_support}, "
            f"max_len={self.max_len} ..."
        )

        self.frequent_itemsets = apriori(
            transaction_df,
            min_support=self.min_support,
            use_colnames=True,
            max_len=self.max_len,
            verbose=0,
            low_memory=False,
        )

        if self.frequent_itemsets.empty:
            logger.warning("Tidak ada frequent itemsets yang ditemukan! Coba turunkan min_support.")
            return self.frequent_itemsets

        self.frequent_itemsets["length"] = self.frequent_itemsets["itemsets"].apply(len)

        logger.info(
            f"Distribusi ukuran itemset:\n"
            f"{self.frequent_itemsets['length'].value_counts().sort_index()}"
        )

        # ========== TAMBAHAN: print itemset dengan panjang >= 2 ==========
        print(
            self.frequent_itemsets[
                self.frequent_itemsets["length"] >= 2
            ]
        )
        # ================================================================

        logger.info(
            f"\nTop 20 itemsets:\n{self.frequent_itemsets.head(20)}"
        )

        logger.info(f"Frequent itemsets ditemukan: {len(self.frequent_itemsets):,}")

        return self.frequent_itemsets

    # ------------------------------------------------------------------ #
    #  STEP 5 & 6 – Association Rules                                     #
    # ------------------------------------------------------------------ #
    def generate_rules(self) -> pd.DataFrame:
        """
        Bentuk association rules dari frequent itemsets.
        Hitung support, confidence, lift, leverage, conviction.
        Filter berdasarkan min_confidence dan min_lift.
        """
        if self.frequent_itemsets is None or self.frequent_itemsets.empty:
            raise RuntimeError("Panggil find_frequent_itemsets() terlebih dahulu.")

        logger.info(
            f"Membentuk association rules — "
            f"min_confidence={self.min_confidence}, min_lift={self.min_lift} ..."
        )

        rules = association_rules(
            self.frequent_itemsets,
            metric="confidence",
            min_threshold=self.min_confidence,
            num_itemsets=len(self.frequent_itemsets),
        )

        if rules.empty:
            logger.warning("Tidak ada rules yang terbentuk. Coba turunkan min_confidence atau min_support.")
            self.rules = rules
            return rules

        # Filter lift
        rules = rules[rules["lift"] >= self.min_lift]

        # Kolom tambahan
        rules["antecedents_list"] = rules["antecedents"].apply(sorted)
        rules["consequents_list"] = rules["consequents"].apply(sorted)
        rules["rule_strength"] = rules["confidence"] * rules["lift"]

        # STEP 7 – Ranking aturan
        rules.sort_values(
            by=["lift", "confidence", "support"],
            ascending=False,
            inplace=True,
        )
        rules.reset_index(drop=True, inplace=True)

        self.rules = rules
        logger.info(f"Association rules terbentuk: {len(self.rules):,}")

        # Bangun cache rekomendasi
        self._build_cache()

        return self.rules

    # ------------------------------------------------------------------ #
    #  STEP 7 – Cache Rekomendasi                                         #
    # ------------------------------------------------------------------ #
    def _build_cache(self):
        """Bangun lookup cache: frozenset(items) → daftar rekomendasi."""
        if self.rules is None or self.rules.empty:
            return

        cache = {}
        for _, row in self.rules.iterrows():
            key = frozenset(row["antecedents"])
            rec = {
                "product": sorted(row["consequents"])[0],
                "support": round(float(row["support"]), 4),
                "confidence": round(float(row["confidence"]), 4),
                "lift": round(float(row["lift"]), 4),
                "conviction": round(float(row.get("conviction", 0)), 4),
                "rule_strength": round(float(row["rule_strength"]), 4),
            }
            cache.setdefault(key, []).append(rec)

        # Deduplicate & sort per key
        for key in cache:
            seen = set()
            unique = []
            for r in cache[key]:
                if r["product"] not in seen:
                    seen.add(r["product"])
                    unique.append(r)
            cache[key] = sorted(unique, key=lambda x: -x["lift"])

        self._rules_cache = cache
        logger.info(f"Cache rekomendasi siap: {len(self._rules_cache):,} entri")

    # ------------------------------------------------------------------ #
    #  STEP 8 – Output Rekomendasi Produk                                 #
    # ------------------------------------------------------------------ #
    def get_recommendations(
        self, product_names: list, top_n: int = 10
    ) -> list:
        """
        Ambil rekomendasi produk berdasarkan item yang sedang dilihat/dibeli.

        Args:
            product_names: List nama produk (antecedent)
            top_n: Jumlah rekomendasi yang dikembalikan
        Returns:
            List dict rekomendasi terurut by lift desc
        """
        if not self._rules_cache:
            return []

        query_set = frozenset(p.upper().strip() for p in product_names)
        results = []
        seen_products = set(query_set)

        # Cari exact match dulu, lalu subset match
        for key, recs in self._rules_cache.items():
            if key == query_set or key.issubset(query_set):
                for r in recs:
                    if r["product"] not in seen_products:
                        results.append(r)
                        seen_products.add(r["product"])

        # Sort by rule_strength (confidence × lift)
        results.sort(key=lambda x: -x["rule_strength"])
        return results[:top_n]

    def get_popular_combos(self, top_n: int = 20) -> list:
        """Produk yang paling sering dibeli bersama (berdasarkan support)."""
        if self.frequent_itemsets is None or self.frequent_itemsets.empty:
            return []

        pairs = self.frequent_itemsets[self.frequent_itemsets["length"] >= 2].copy()
        if pairs.empty:
            return []

        pairs.sort_values("support", ascending=False, inplace=True)

        result = []
        for _, row in pairs.head(top_n).iterrows():
            result.append(
                {
                    "items": sorted(row["itemsets"]),
                    "support": round(float(row["support"]), 4),
                    "frequency_pct": f"{row['support']*100:.2f}%",
                }
            )
        return result

    def get_top_rules(self, top_n: int = 50) -> list:
        """Kembalikan top N association rules."""
        if self.rules is None or self.rules.empty:
            return []

        cols = [
            "antecedents_list", "consequents_list",
            "support", "confidence", "lift",
            "conviction", "rule_strength",
        ]
        top = self.rules[cols].head(top_n)
        return top.to_dict(orient="records")

    def get_summary(self) -> dict:
        """Statistik ringkas hasil Apriori."""
        return {
            "frequent_itemsets_count": len(self.frequent_itemsets)
            if self.frequent_itemsets is not None
            else 0,
            "rules_count": len(self.rules) if self.rules is not None else 0,
            "parameters": {
                "min_support": self.min_support,
                "min_confidence": self.min_confidence,
                "min_lift": self.min_lift,
                "max_len": self.max_len,
            },
            "rules_stats": (
                {
                    "avg_confidence": round(float(self.rules["confidence"].mean()), 4),
                    "avg_lift": round(float(self.rules["lift"].mean()), 4),
                    "max_lift": round(float(self.rules["lift"].max()), 4),
                    "max_confidence": round(float(self.rules["confidence"].max()), 4),
                }
                if self.rules is not None and not self.rules.empty
                else {}
            ),
        }

    def run_pipeline(self, transaction_df: pd.DataFrame):
        """Jalankan seluruh pipeline Apriori sekaligus."""
        self.find_frequent_itemsets(transaction_df)
        self.generate_rules()
        return self