"""Gera o .zip de saídas (os JSONs completos do contrato, um por documento).

    python3 -m tools.empacotar_saidas --output resultados --zip saidas.zip

Valida as saídas antes (tools.validar_saida) e não empacota se houver problema. Não é o arquivo do
leaderboard do Kaggle: lá vai o submission.csv (python3 json_to_submission.py resultados submission.csv),
e o Kaggle só aceita zip com um único CSV. Só biblioteca padrão.
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

from .validar_saida import check


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True, help="pasta com os <documento_id>.json")
    parser.add_argument("--input", type=Path, default=Path("data/txt"), help="pasta dos .txt (validação)")
    parser.add_argument("--zip", type=Path, default=Path("saidas.zip"))
    args = parser.parse_args()
    if problems := check(args.input, args.output, Path("data/desafio1_bracis.db")):
        print("\n".join(problems[:20]), file=sys.stderr)
        return 1
    with zipfile.ZipFile(args.zip, "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted(args.output.glob("*.json")):
            z.write(path, path.name)
    print(f"{args.zip} ({args.zip.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
