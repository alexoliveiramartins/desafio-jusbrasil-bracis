"""Mede a taxa de acerto de cada regra do resolvedor em conjuntos rotulados.

    python -m tools.calibrate [conjunto ...]      (padrão: os conjuntos de iteração)
    python -m tools.calibrate v5_iter llm_llama_iter --nlp-model <modelo>   # regras da camada de NLP

Use para ajustar CONFIDENCE em src/classify.py. Calibrar no goldenset oficial
seria ajustar ao próprio dev; calibrar em holdout, contaminá-lo.
"""

from __future__ import annotations

import argparse
from collections import defaultdict

from src.classify import CONFIDENCE, CanonicalIndex
from src.nlp import NLP_CONFIDENCE, NLPLayer, OllamaClient

from .evaluation import DB, iou, load_gold, run_split, split_paths
from .synth import ITERATION


def current(regra: str) -> float:
    base = regra.removesuffix("+novo")
    return NLP_CONFIDENCE.get(base, CONFIDENCE.get(base, 0.0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("splits", nargs="*", default=ITERATION)
    parser.add_argument("--nlp-model", default=None, help="liga a camada de NLP com este modelo (cache do nlp_bench)")
    args = parser.parse_args()
    splits = args.splits
    nlp = None
    if args.nlp_model:
        from .nlp_bench import cache_dir
        nlp = NLPLayer(OllamaClient(model=args.nlp_model, cache_dir=cache_dir(args.nlp_model)))
    index = CanonicalIndex.from_sqlite(DB)
    stats = defaultdict(lambda: [0, 0])  # regra -> [acertos, total]
    for name in splits:
        split = split_paths(name)
        gold = load_gold(split.gold)
        for doc_id, doc in run_split(split, index, debug=True, nlp=nlp).items():
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
        print(f"{regra:<36}{hits:>5}/{total:<5}{current(regra):>8.2f}{suggested:>10.2f}")


if __name__ == "__main__":
    main()
