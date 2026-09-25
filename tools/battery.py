"""Bateria: vários perfis × várias sementes, métrica oficial, só CPU.

    python -m tools.battery                              # todos os perfis, 5 sementes
    python -m tools.battery --profiles ineditos_v2 ineditos_v4 --seeds 3
    python -m tools.battery --regen                      # regera os dados
    python -m tools.battery --fixed dev llm_holdout --profiles ineditos_v2 --base-seed 9000

Gera cada conjunto em data/synthetic/bateria/<perfil>_s<semente>/, roda o
pipeline em processo, aplica kaggle_metric.avaliar e resume média, desvio,
pior caso e τ por perfil. Salva relatorios/bateria.json.

É um instrumento de MEDIDA: não ajuste regras olhando erros dos holdouts.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

from src.classify import CanonicalIndex

from .evaluation import DB, SYNTHETIC, Split, load_gold, run_split, score, split_paths
from .synth import HOLDOUT, ITERATION, PROFILES, Sources, generate

BASE_SEED = 7000
OUT = SYNTHETIC / "bateria"


def run_one(split: Split, index: CanonicalIndex) -> dict:
    result = score(load_gold(split.gold), run_split(split, index))
    niveis = result["niveis"]
    return {
        "score": result["score_final"],
        "tau": max(n["tau"] for n in niveis.values()),
        "recall_spans": result["recall_spans"],
        "niveis": {k: round(v["score"], 4) for k, v in niveis.items()},
        "f1": {k: v["f1_por_classe"] for k, v in niveis.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profiles", nargs="*", default=ITERATION + HOLDOUT, choices=list(PROFILES))
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=BASE_SEED, help="primeira semente (use uma nova para dados nunca vistos)")
    parser.add_argument("--fixed", nargs="*", default=[], help="conjuntos fixos avaliados uma vez (dev, llm_holdout...)")
    parser.add_argument("--regen", action="store_true")
    parser.add_argument("--report", type=Path, default=Path("relatorios/bateria.json"))
    args = parser.parse_args()

    index = CanonicalIndex.from_sqlite(DB)
    src = None
    results = defaultdict(list)
    start = time.monotonic()
    for name in args.fixed:
        results[name].append(run_one(split_paths(name), index))
    for profile in args.profiles:
        for seed in range(args.base_seed, args.base_seed + args.seeds):
            folder = OUT / f"{profile}_s{seed}"
            if args.regen or not (folder / "goldenset.csv").exists():
                src = src or Sources(DB)
                generate(profile, folder, seed, src)
            results[profile].append(run_one(Split(folder.name, folder / "txt", folder / "goldenset.csv"), index))

    print(f"\n{'perfil':<22}{'papel':<10}{'média':>8}{'desvio':>8}{'pior':>8}{'τ máx':>8}{'spans':>8}"
          "   F1 médio (real/inv/inc)")
    report = {}
    for profile, runs in results.items():
        scores = [r["score"] for r in runs]
        f1 = defaultdict(list)
        for r in runs:
            for level in r["f1"].values():
                for cls, v in level.items():
                    f1[cls].append(v)
        f1m = {c: statistics.mean(v) for c, v in f1.items()}
        row = {
            "papel": PROFILES[profile].role if profile in PROFILES else "fixo",
            "media": statistics.mean(scores),
            "desvio": statistics.pstdev(scores),
            "pior": min(scores),
            "tau_max": max(r["tau"] for r in runs),
            "recall_spans": statistics.mean(r["recall_spans"] for r in runs),
            "f1_medio": f1m,
            "execucoes": runs,
        }
        report[profile] = row
        print(f"{profile:<22}{row['papel']:<10}{row['media']:>8.4f}{row['desvio']:>8.4f}{row['pior']:>8.4f}"
              f"{row['tau_max']:>8.3f}{row['recall_spans']:>8.3f}   "
              f"{f1m.get('real', 0):.3f}/{f1m.get('inventada', 0):.3f}/{f1m.get('incompleta', 0):.3f}")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n{sum(len(v) for v in results.values())} conjuntos em {time.monotonic() - start:.0f}s -> {args.report}")


if __name__ == "__main__":
    main()
