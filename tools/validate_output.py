"""Confere as saídas contra o Contrato de Entrada e Saída antes de submeter.

    python3 -m tools.validate_output --input data/txt --output resultados [--db data/desafio1_bracis.db]

Para cada .txt de --input exige <documento_id>.json em --output com todos os campos, e confere:
trecho == texto[inicio:fim] (codepoints, arquivo lido com newline=""), tipo e classificação válidos,
resolucao.id_canonico só em `real` e existente na base, confianca em [0, 1], spans dentro do texto
e nenhum par de citações com IoU >= 0,5 (a métrica recusa a submissão inteira).
Só biblioteca padrão. Sai com código 1 se houver qualquer problema.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

FIELDS = {"inicio", "fim", "trecho", "tipo", "classificacao", "resolucao", "confianca"}


def _iou(a: tuple[int, int], b: tuple[int, int]) -> float:
    """Interseção sobre união de dois spans.

    Parameters
    ----------
    a, b : tuple of (int, int)
        Spans ``(início, fim)`` não vazios.

    Returns
    -------
    float
        IoU entre 0 e 1.
    """
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    return inter / ((a[1] - a[0]) + (b[1] - b[0]) - inter) if inter else 0.0


def check(txt_dir: Path, out_dir: Path, db: Path | None) -> list[str]:
    """Confere as saídas de uma execução contra o contrato.

    Parameters
    ----------
    txt_dir : pathlib.Path
        Pasta dos ``.txt`` de entrada.
    out_dir : pathlib.Path
        Pasta dos ``<documento_id>.json`` de saída.
    db : pathlib.Path or None
        Base canônica; se existir, cada ``id_canonico`` de ``real`` precisa estar na tabela
        ``documentos``.

    Returns
    -------
    list of str
        Problemas encontrados (vazia se tudo confere). Também imprime o total de documentos,
        citações e problemas.
    """
    ids = None
    if db and db.is_file():
        with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
            ids = {str(i) for (i,) in conn.execute("SELECT id FROM documentos")}
    problems, total = [], 0
    for txt in sorted(txt_dir.glob("*.txt")):
        out = out_dir / f"{txt.stem}.json"
        if not out.is_file():
            problems.append(f"{txt.stem}: sem {out.name}")
            continue
        text = txt.open(encoding="utf-8", newline="").read()
        doc = json.loads(out.read_text(encoding="utf-8"))
        if doc.get("documento_id") != txt.stem or not isinstance(doc.get("citacoes"), list):
            problems.append(f"{txt.stem}: documento_id/citacoes inválidos")
            continue
        for n, c in enumerate(doc["citacoes"]):
            where = f"{txt.stem}#{n}"
            total += 1
            if missing := FIELDS - set(c):
                problems.append(f"{where}: faltam campos {sorted(missing)}")
                continue
            if not (isinstance(c["inicio"], int) and isinstance(c["fim"], int) and 0 <= c["inicio"] < c["fim"] <= len(text)):
                problems.append(f"{where}: span inválido {c['inicio']}-{c['fim']}")
            elif text[c["inicio"]:c["fim"]] != c["trecho"]:
                problems.append(f"{where}: trecho difere de texto[inicio:fim]")
            if c["tipo"] not in ("lei", "jurisprudencia"):
                problems.append(f"{where}: tipo {c['tipo']!r}")
            if c["classificacao"] not in ("real", "inventada", "incompleta"):
                problems.append(f"{where}: classificacao {c['classificacao']!r}")
            res = c["resolucao"]
            if c["classificacao"] == "real":
                cid = (res or {}).get("id_canonico") if isinstance(res, dict) else None
                if not cid or (ids is not None and str(cid) not in ids):
                    problems.append(f"{where}: real sem id_canonico válido ({cid!r})")
            elif res is not None:
                problems.append(f"{where}: resolucao deveria ser null em {c['classificacao']}")
            if not (isinstance(c["confianca"], (int, float)) and 0 <= c["confianca"] <= 1):
                problems.append(f"{where}: confianca {c['confianca']!r}")
        spans = [(c["inicio"], c["fim"]) for c in doc["citacoes"]
                 if isinstance(c.get("inicio"), int) and isinstance(c.get("fim"), int) and c["inicio"] < c["fim"]]
        for a in range(len(spans)):
            for b in range(a + 1, len(spans)):
                if _iou(spans[a], spans[b]) >= 0.5:
                    problems.append(f"{txt.stem}: citações {spans[a]} e {spans[b]} com IoU >= 0,5 (a métrica recusa)")
    extra = {p.stem for p in out_dir.glob("*.json")} - {t.stem for t in txt_dir.glob("*.txt")}
    problems += [f"{e}.json: sem .txt correspondente" for e in sorted(extra)]
    print(f"{len(list(txt_dir.glob('*.txt')))} documentos, {total} citações, {len(problems)} problemas")
    return problems


def main() -> int:
    """Valida uma pasta de saídas e imprime até 50 problemas.

    Returns
    -------
    int
        0 se as saídas cumprem o contrato; 1 se há qualquer problema.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--db", type=Path, default=Path("data/desafio1_bracis.db"))
    args = parser.parse_args()
    problems = check(args.input, args.output, args.db)
    for p in problems[:50]:
        print(" ", p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
