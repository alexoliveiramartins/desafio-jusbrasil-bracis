"""Conjuntos de estresse: os textos do dev com ruído pesado e o gabarito remapeado.

    python -m tools.stress                                # todos os perfis, 3 sementes
    python -m tools.stress --profiles extremo so_ocr --seeds 5
    python -m tools.stress.bench                          # gera o que faltar e pontua as soluções
    python -m tools.stress --source data/synthetic/val --out data/stress/val   # ruído sobre outra fonte

Diferença para tools.synth: lá as peças são escritas do zero a partir de
catálogos; aqui o texto é o próprio `data/txt` (mesmas citações, mesmo
gabarito) degradado por digitalização ruim, erros de digitação e de
codificação. Mede quanto a solução perde só por causa do ruído.

Perfis:
  limpo        controle, texto original (deve reproduzir o dev);
  so_<família> uma família de ruído isolada, na intensidade "pesado" (ablação);
  leve … extremo  todas as famílias juntas, intensidade crescente.

Saída: data/stress/<perfil>_s<semente>/{txt/, goldenset.csv, manifest.json}.
Os rótulos (classe, id_canonico) não mudam: ver invariantes em noise.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

from .noise import FAMILIES, NoisyDoc

DEV_TXT = Path("data/txt")
DEV_GOLD = Path("data/goldenset.csv")
OUT = Path("data/stress")
BASE_SEED = 1000

# Intensidade de cada família no nível "pesado"; o significado de p está em
# cada função de noise.py (em geral, probabilidade por caractere ou palavra).
PESADO = {
    "fim_de_linha": 0.5, "lixo": 0.4, "quebras": 0.2, "espacos": 0.2, "pontuacao": 0.4,
    "typos": 0.08, "caixa": 0.1, "ocr": 0.1, "acentos": 0.3, "mojibake": 0.1, "invisiveis": 0.15,
}
SEVERITY = {"leve": 0.25, "moderado": 0.5, "pesado": 1.0, "extremo": 1.6}

PROFILES: dict[str, dict[str, float]] = {"limpo": {}}
PROFILES |= {f"so_{family}": {family: p} for family, p in PESADO.items()}
PROFILES |= {name: {f: min(1.0, p * k) for f, p in PESADO.items()} for name, k in SEVERITY.items()}
ABLATION = [name for name in PROFILES if name.startswith("so_")]
COMPOSITE = list(SEVERITY)


def read_text(path: Path) -> str:
    with path.open(encoding="utf-8", newline="") as stream:
        return stream.read()


def escape(trecho: str) -> str:
    """Mesma convenção do goldenset oficial: quebras de linha como \\n literal."""
    return trecho.replace("\r", "\\r").replace("\n", "\\n")


def load_gold_rows(path: Path) -> tuple[list[str], dict[str, list[dict]]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
    by_doc: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_doc[row["documento_id"]].append(row)
    return reader.fieldnames, dict(by_doc)


def noisy_document(text: str, spans: list[tuple[int, int]], params: dict[str, float],
                   rng: random.Random) -> tuple[str, list[tuple[int, int]], int]:
    """Aplica as famílias de `params` e devolve (texto, spans remapeados, edições)."""
    doc = NoisyDoc(text, spans)
    for family, fn in FAMILIES.items():
        if family in params:
            fn(doc, rng, params[family])
    new_text, starts, ends = doc.render()
    new_spans = [(starts[s], ends[e - 1]) for s, e in spans]
    for (s, e), (a, b) in zip(spans, new_spans):
        if not 0 <= a < b <= len(new_text):
            raise AssertionError(f"span ({s}, {e}) virou ({a}, {b})")
    return new_text, new_spans, doc.edits()


def generate(profile: str, seed: int, out: Path, txt_dir: Path = DEV_TXT, gold_csv: Path = DEV_GOLD) -> dict:
    params = PROFILES[profile]
    fields, gold = load_gold_rows(gold_csv)
    (out / "txt").mkdir(parents=True, exist_ok=True)
    rows_out, chars, edits, altered = [], 0, 0, 0
    for doc_id in sorted(gold):
        rows = sorted(gold[doc_id], key=lambda r: int(r["inicio"]))
        text = read_text(txt_dir / f"{doc_id}.txt")
        rng = random.Random(f"{seed}:{profile}:{doc_id}")
        spans = [(int(r["inicio"]), int(r["fim"])) for r in rows]
        new_text, new_spans, n_edits = noisy_document(text, spans, params, rng)
        with (out / "txt" / f"{doc_id}.txt").open("w", encoding="utf-8", newline="") as stream:
            stream.write(new_text)
        chars += len(text)
        edits += n_edits
        for row, (s, e), (a, b) in zip(rows, spans, new_spans):
            altered += new_text[a:b] != text[s:e]
            rows_out.append(row | {"inicio": a, "fim": b, "trecho": escape(new_text[a:b])})
    with (out / "goldenset.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows_out)
    manifest = {
        "perfil": profile, "semente": seed, "familias": params, "documentos": len(gold),
        "citacoes": len(rows_out), "taxa_edicao": edits / chars if chars else 0.0,
        "citacoes_alteradas": altered / len(rows_out) if rows_out else 0.0,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def seeds_for(profile: str, base_seed: int, seeds: int) -> list[int]:
    """`limpo` não tem aleatoriedade: uma semente basta."""
    return [base_seed] if profile == "limpo" else list(range(base_seed, base_seed + seeds))


def source_paths(source: Path | None) -> tuple[Path, Path]:
    return (DEV_TXT, DEV_GOLD) if source is None else (source / "txt", source / "goldenset.csv")


def split_dir(profile: str, seed: int, out: Path = OUT) -> Path:
    return out / f"{profile}_s{seed}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profiles", nargs="+", default=list(PROFILES), choices=list(PROFILES))
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--base-seed", type=int, default=BASE_SEED)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--source", type=Path, default=None,
                        help="pasta com txt/ e goldenset.csv (padrão: data/txt + data/goldenset.csv)")
    args = parser.parse_args()
    txt_dir, gold_csv = source_paths(args.source)
    for profile in args.profiles:
        for seed in seeds_for(profile, args.base_seed, args.seeds):
            m = generate(profile, seed, split_dir(profile, seed, args.out), txt_dir, gold_csv)
            print(f"{profile:<16} s{seed}: {m['citacoes']} citações, edição {m['taxa_edicao']:.1%}, "
                  f"citações alteradas {m['citacoes_alteradas']:.0%} -> {split_dir(profile, seed, args.out)}")
