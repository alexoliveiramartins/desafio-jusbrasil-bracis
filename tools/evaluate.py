"""Roda o pipeline e pontua com a métrica oficial.

    python -m tools.evaluate                         # dev (goldenset oficial)
    python -m tools.evaluate dev val ineditos_v3     # vários conjuntos
    python -m tools.evaluate ineditos_v3 --diagnose  # lista cada divergência
    python -m tools.evaluate --submission submission.csv [--gold data/goldenset.csv]

Conjuntos: `dev` (data/txt + data/goldenset.csv), nomes de data/synthetic/
(gerados por `python -m tools.synth`) ou uma pasta com txt/ e goldenset.csv.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.classify import CanonicalIndex

from .evaluation import DB, DEV, diagnose, load_gold, run_split, score, score_submission, split_paths, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("splits", nargs="*", default=["dev"])
    parser.add_argument("--diagnose", action="store_true", help="lista faltantes, erros e espúrias")
    parser.add_argument("--submission", type=Path, help="pontua um submission.csv em vez de rodar o pipeline")
    parser.add_argument("--gold", type=Path, default=DEV[1], help="goldenset do --submission")
    parser.add_argument("--db", type=Path, default=DB)
    args = parser.parse_args()

    if args.submission:
        print(json.dumps(score_submission(args.submission, args.gold), indent=2, ensure_ascii=False))
        return

    index = CanonicalIndex.from_sqlite(args.db)
    for name in args.splits:
        split = split_paths(name)
        gold = load_gold(split.gold)
        outputs = run_split(split, index, debug=args.diagnose)
        if args.diagnose:
            confusion = diagnose(gold, outputs)
            print("\nMatriz (gabarito -> predição):")
            for (g, p), n in sorted(confusion.items()):
                print(f"  {g:>10} -> {p:<10} {n}")
        print(f"{name:<20} {summary(score(gold, outputs))}")


if __name__ == "__main__":
    main()
