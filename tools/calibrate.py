"""Mede a taxa de acerto de cada regra do resolvedor em conjuntos rotulados.

    python -m tools.calibrate [conjunto ...]      (padrão: os conjuntos de iteração)

Use para ajustar CONFIDENCE em src/classify.py. Calibrar no goldenset oficial
seria ajustar ao próprio dev; calibrar em holdout, contaminá-lo.
"""

from __future__ import annotations

import sys
from collections import defaultdict

from src.classify import CanonicalIndex
from src.classify import CONFIDENCE

from .evaluation import DB, iou, load_gold, run_split, split_paths
from .synth import ITERATION


def main() -> None:
    splits = sys.argv[1:] or ITERATION
    index = CanonicalIndex.from_sqlite(DB)
    stats = defaultdict(lambda: [0, 0])  # regra -> [acertos, total]
    for name in splits:
        split = split_paths(name)
        gold = load_gold(split.gold)
        for doc_id, doc in run_split(split, index, debug=True).items():
            rows = gold[doc_id]
            for pred in doc["citacoes"]:
                span = (pred["inicio"], pred["fim"])
                match = max(rows, key=lambda r: iou(span, (int(r["inicio"]), int(r["fim"]))))
                if iou(span, (int(match["inicio"]), int(match["fim"]))) < 0.5:
                    continue
                pred_id = (pred["resolucao"] or {}).get("id_canonico", "")
                ok = pred["classificacao"] == match["classificacao"] and (
                    pred["classificacao"] != "real" or pred_id == match["id_canonico"])
                stats[pred["_regra"]][0] += ok
                stats[pred["_regra"]][1] += 1

    print(f"{'regra':<36}{'acertos':>11}{'atual':>8}{'sugerida':>10}")
    for regra, (hits, total) in sorted(stats.items()):
        # Suavização de Laplace e teto: nunca 1,0 (Brier pune excesso de certeza).
        suggested = min(0.98, max(0.5, (hits + 1) / (total + 2)))
        print(f"{regra:<36}{hits:>5}/{total:<5}{CONFIDENCE[regra]:>8.2f}{suggested:>10.2f}")


if __name__ == "__main__":
    main()
