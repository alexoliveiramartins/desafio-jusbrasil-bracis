"""Gerador de conjuntos sintéticos rotulados.

    python -m tools.synth                          # conjuntos de iteração (semente 2026)
    python -m tools.synth --splits ineditos_v3 --seed 7

Por que existe: o goldenset tem 26 documentos. Ajustar padrões até acertar
todos eles não diz nada sobre o conjunto cego. Cada perfil gera peças com
citações de rótulo conhecido (ver sources.py) e ataca uma fraqueza.

Papéis:
  iteração  diagnosticar e corrigir aqui (só fenômenos gerais);
  holdout   só medir, pela bateria (tools.battery). Se um holdout for usado
            para decidir algo, ele vira iteração e é preciso gerar outro.
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from . import catalog_v1, catalog_v2, catalog_v3, catalog_v4, known
from .noise import ocr_noise, page_junk, page_junk_v4, pollute_citation, pollute_v4
from .sources import Sources

CATALOGS = {"v1": catalog_v1.CATALOG, "v2": catalog_v2.CATALOG,
            "v3": catalog_v3.CATALOG, "v4": catalog_v4.CATALOG}
NOISES = {"v1": (pollute_citation, page_junk), "v4": (pollute_v4, page_junk_v4)}


@dataclass
class Profile:
    """Como um conjunto é gerado."""
    role: str                         # "iteracao" ou "holdout"
    extra_formats: bool = False       # formatos exclusivos (nunca vistos no val)
    holdout_formats: bool = False     # formatos de `formatos_ineditos`
    ocr_rate: float = 0.06            # ruído de caracteres no nível 2
    level2_share: float = 0.5
    cites: tuple[int, int] = (6, 10)
    dense: bool = False               # duas citações no mesmo parágrafo
    weights: dict | None = None       # gerador -> peso (padrão: known.GENERATORS)
    docs: int = 80
    novel_share: float = 0.0          # fração de citações do catálogo inédito
    catalog: str = "v1"               # catálogo inédito (e frases de contexto)
    pollution: float = 0.0            # poluição de digitalização na citação (nível 2)
    junk_rate: float = 0.0            # lixo de página entre parágrafos (nível 2)
    noise: str = "v1"                 # família de poluição (noise.py)


_POLLUTED = dict(ocr_rate=0.10, pollution=0.4, junk_rate=0.4, docs=60)
PROFILES = {
    # --- iteração
    "val": Profile("iteracao"),
    "armadilhas": Profile("iteracao", weights={   # número a 1 dígito de um real, fases recursais
        "gen_invented_process": 0.35, "gen_stage": 0.25, "gen_real_process": 0.2, "gen_descriptive": 0.2,
    }),
    "denso": Profile("iteracao", cites=(18, 26), dense=True, docs=40),
    "poluido": Profile("iteracao", extra_formats=True, ocr_rate=0.10, level2_share=1.0,
                       pollution=0.35, junk_rate=0.35, docs=60),
    "ineditos": Profile("iteracao", novel_share=0.9, docs=60),
    "ineditos_poluido": Profile("iteracao", novel_share=0.9, **_POLLUTED),
    "ineditos_v3": Profile("iteracao", novel_share=0.9, catalog="v3", dense=True, cites=(8, 14), docs=60),
    "ineditos_v3_poluido": Profile("iteracao", novel_share=0.9, catalog="v3", dense=True, cites=(8, 14),
                                   **_POLLUTED),
    # Ruído extremo com catálogo de iteração: estuda τ sem tocar em `poluido_extremo`.
    "ineditos_v3_extremo": Profile("iteracao", novel_share=0.9, catalog="v3", dense=True, cites=(8, 14),
                                   ocr_rate=0.18, level2_share=1.0, pollution=0.7, junk_rate=0.6, docs=60),
    # --- holdout
    "test": Profile("holdout", extra_formats=True),
    "ocr_forte": Profile("holdout", extra_formats=True, ocr_rate=0.14, level2_share=1.0),
    "formatos_ineditos": Profile("holdout", extra_formats=True, holdout_formats=True),
    "poluido_extremo": Profile("holdout", extra_formats=True, ocr_rate=0.18, level2_share=1.0,
                               pollution=0.7, junk_rate=0.6, docs=60),
    "ineditos_v2": Profile("holdout", novel_share=0.9, catalog="v2", docs=60),
    "ineditos_v2_poluido": Profile("holdout", novel_share=0.9, catalog="v2", **_POLLUTED),
    "ineditos_v4": Profile("holdout", novel_share=0.9, catalog="v4", dense=True, cites=(8, 14), docs=60),
    "ineditos_v4_poluido": Profile("holdout", novel_share=0.9, catalog="v4", dense=True, cites=(8, 14),
                                   noise="v4", **_POLLUTED),
}
ITERATION = [name for name, p in PROFILES.items() if p.role == "iteracao"]
HOLDOUT = [name for name, p in PROFILES.items() if p.role == "holdout"]


def build_document(rng: random.Random, src: Sources, level: int, profile: Profile):
    def filler(template: str) -> str:
        return template.format(
            autos=f"{rng.randint(1000000, 9999999)}-{rng.randint(10, 99)}.20{rng.randint(10, 25)}."
                  f"{rng.randint(1, 8)}.{rng.randint(1, 27):02d}.{rng.randint(1, 9999):04d}",
            parte=rng.choice(known.PARTES),
            data=f"{rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}/20{rng.randint(15, 25)}",
            data2=f"{rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}/20{rng.randint(15, 25)}",
            valor=f"{rng.randint(1, 900)}.{rng.randint(100, 999)},00", fls=rng.randint(10, 900),
            c="{c}", c2="{c2}",
        )

    if profile.weights:
        by_name = {f.__name__: f for f, _ in known.GENERATORS} | {"gen_stage": known.gen_stage}
        funcs, weights = zip(*[(by_name[n], w) for n, w in profile.weights.items()])
    else:
        funcs, weights = zip(*known.GENERATORS)
    catalog = CATALOGS[profile.catalog]
    pollute, junk = NOISES[profile.noise]

    def noisy(text: str) -> str:
        if level == 2:
            text = ocr_noise(rng, text, profile.ocr_rate)
            if profile.pollution:
                text = pollute(rng, text, profile.pollution)
        return text

    def next_cite():
        """(texto da citação, texto depois dela, Cite)."""
        while True:
            if profile.novel_share and rng.random() < profile.novel_share:
                novel_funcs, novel_weights = zip(*catalog.generators)
                cite = rng.choices(novel_funcs, novel_weights)[0](rng, src, True)
            else:
                cite = rng.choices(funcs, weights)[0](rng, src, profile.extra_formats)
            if cite is not None and profile.holdout_formats and rng.random() < 0.6:
                cite = known.holdout_variant(rng, cite)
            if cite is not None:
                return noisy(cite.text), noisy(cite.after) if cite.after else "", cite

    parts = [filler(rng.choice(known.OPENINGS))]
    labels = []
    cursor = len(parts[0])
    target = rng.randint(*profile.cites)
    while len(labels) < target:
        if profile.dense and rng.random() < 0.6:
            template = filler(rng.choice(catalog.dense_paragraphs or known.DENSE_PARAGRAPHS))
            (t1, a1, c1), (t2, a2, c2) = next_cite(), next_cite()
            before, rest = template.split("{c}")
            middle, after = rest.split("{c2}")
            para = "\n" + before + t1 + a1 + middle + t2 + a2 + after + "\n"
            s1 = cursor + 1 + len(before)
            s2 = s1 + len(t1) + len(a1) + len(middle)
            labels += [(s1, s1 + len(t1), t1, c1), (s2, s2 + len(t2), t2, c2)]
        else:
            text, tail, cite = next_cite()
            template = filler(rng.choice(catalog.paragraphs))
            before, after = template.split("{c}")
            para = "\n" + before + text + tail + after + "\n"
            start = cursor + 1 + len(before)
            labels.append((start, start + len(text), text, cite))
        parts.append(para)
        cursor += len(para)
        if level == 2 and rng.random() < profile.junk_rate:
            page = junk(rng)
            parts.append(page)
            cursor += len(page)
        if rng.random() < 0.3:
            distractor = "\n" + catalog.distractor(rng, filler) + "\n"
            parts.append(distractor)
            cursor += len(distractor)
    parts.append(rng.choice(known.CLOSINGS))
    return "".join(parts), labels


def generate(split: str, dest: Path, seed: int, src: Sources, docs: int | None = None) -> None:
    """Gera <dest>/txt/*.txt e <dest>/goldenset.csv (mesmo formato do goldenset oficial)."""
    profile = PROFILES[split]
    docs = docs or profile.docs
    rng = random.Random(f"{split}-{seed}")
    txt_dir = dest / "txt"
    if txt_dir.exists():
        for old in txt_dir.glob("*.txt"):
            old.unlink()
    txt_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    n_level1 = round(docs * (1 - profile.level2_share))
    for i in range(docs):
        level = 1 if i < n_level1 else 2
        doc_id = f"syn_{split}_n{level}_{i + 1:03d}"
        text, labels = build_document(rng, src, level, profile)
        (txt_dir / f"{doc_id}.txt").write_text(text, encoding="utf-8", newline="")
        for j, (start, end, trecho, cite) in enumerate(labels, 1):
            assert text[start:end] == trecho
            rows.append({
                "nivel": level, "documento_id": doc_id, "citacao_id": f"g{j}",
                "inicio": start, "fim": end, "trecho": trecho.replace("\n", "\\n"),
                "tipo": cite.tipo, "classificacao": cite.classe, "id_canonico": cite.doc_id,
            })
    with (dest / "goldenset.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    counts = Counter(r["classificacao"] for r in rows)
    print(f"{split}: {docs} documentos, {len(rows)} citações {dict(counts)} -> {dest}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", type=Path, default=Path("data/desafio1_bracis.db"))
    parser.add_argument("--out", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--docs", type=int, default=None, help="sobrescreve o tamanho do perfil")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--splits", nargs="+", default=ITERATION, choices=list(PROFILES))
    args = parser.parse_args()
    src = Sources(args.db)
    for split in args.splits:
        generate(split, args.out / split, args.seed, src, args.docs)
