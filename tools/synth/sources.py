"""Fontes de rótulo dos conjuntos sintéticos.

Cuidados contra circularidade (o gerador não pode "concordar" com o
resolvedor por construção):
  * `real` numerado: só registros cujo número é único na base (sem
    duplicatas nem fases recursais concorrentes); o id é o do registro.
  * `inventada` numerado: números que não aparecem em nenhum cabeçalho de
    toda a base (busca por dígitos no texto bruto, não pelo índice).
  * descritivas: o rótulo vem da contagem de registros com mesmo tribunal,
    ano e relator (tokens do nome), calculada aqui e não pelo resolvedor.
"""

from __future__ import annotations

import random
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from src.classify import CanonicalIndex
from src.normalize import name_tokens


@dataclass
class Cite:
    text: str
    tipo: str
    classe: str
    doc_id: str = ""
    # Texto logo após a citação, fora do span anotado (", Rel. Min. X, DJe ...").
    after: str = ""


@dataclass(frozen=True)
class Catalog:
    """Superfícies de um conjunto: geradores de citação, frases e distratores."""
    generators: list[tuple[Callable, float]]
    paragraphs: list[str]
    # (rng, filler) -> parágrafo sem citação
    distractor: Callable[[random.Random, Callable[[str], str]], str]
    # Frases com duas citações ({c} e {c2}), usadas nos perfis densos.
    dense_paragraphs: list[str] | None = None


# Súmulas presentes na base (a cobertura é congelada).
SUMULA_REAL = [("STJ", 83, False), ("STJ", 211, False), ("STJ", 443, False),
               ("STF", 10, True), ("TST", 331, False)]
COURT_LONG = {"STJ": "Superior Tribunal de Justiça", "STF": "Supremo Tribunal Federal",
              "TST": "Tribunal Superior do Trabalho", "TSE": "Tribunal Superior Eleitoral",
              "STM": "Superior Tribunal Militar"}


def fmt_thousands(n: int, sep: str = ".") -> str:
    return f"{n:,}".replace(",", sep)


def clean_relator(relator: str) -> str:
    """Nome do relator como aparece numa citação: sem título nem cargo."""
    relator = re.sub(r"^(?:Min\.|Ministr[oa])\s*", "", relator)
    return re.split(r"\s*\(|\s+DESEMBARGADOR|\s+JUIZ", relator)[0].strip()


class Sources:
    def __init__(self, db: Path):
        self.index = CanonicalIndex.from_sqlite(db)
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        heads = [re.sub(r"\D", "", t) for (t,) in conn.execute(
            "SELECT substr(texto, 1, 1500) FROM documentos WHERE natureza = 'acordao'")]
        conn.close()
        self.header_digits = "\n".join(heads)
        ix = self.index

        # Registros cujo número é exclusivo: rótulo `real` inequívoco.
        self.unique_short = [ids[0] for ids in ix.by_short.values() if len(ids) == 1]
        self.unique_cnj = [ids[0] for ids in ix.by_cnj.values() if len(ids) == 1]
        self.short_of = {ids[0]: k for k, ids in ix.by_short.items() if len(ids) == 1}
        self.cnj_of = {ids[0]: k for k, ids in ix.by_cnj.items() if len(ids) == 1}

        # Descritivas: todos os acórdãos do mesmo tribunal e ano cujo relator
        # contém todos os tokens do nome citado ("Sérgio Banhos" também é
        # "Sergio Silveira Banhos"). Comparação exata de tokens, sem fuzzy.
        self.by_court_year = defaultdict(list)
        for r in ix.records.values():
            if r.natureza == "acordao" and r.relator and r.ano and r.tribunal:
                self.by_court_year[(r.tribunal, r.ano)].append(r)
        self.desc_groups = []
        seen = set()
        for (tribunal, ano), records in self.by_court_year.items():
            for r in records:
                tokens = frozenset(name_tokens(r.relator))
                if not tokens or (tribunal, ano, tokens) in seen:
                    continue
                seen.add((tribunal, ano, tokens))
                same = [x for x in records if tokens <= frozenset(name_tokens(x.relator))]
                self.desc_groups.append(((tribunal, ano, r.relator), same))

    def number_exists(self, digits: str) -> bool:
        return digits in self.header_digits

    def key_of(self, doc_id: str):
        """Número próprio (CNJ ou curto) de um registro de número único."""
        return self.cnj_of.get(doc_id) or self.short_of.get(doc_id)

    def same_relator(self, tribunal: str, ano: int, name: str, classe: str | None = None) -> list:
        """Registros compatíveis com (tribunal, ano, tokens do nome[, classe])."""
        tokens = frozenset(name_tokens(name))
        return [r for r in self.by_court_year[(tribunal, ano)]
                if tokens and tokens <= frozenset(name_tokens(r.relator))
                and (classe is None or r.classe == classe)]


def perturb_number(rng: random.Random, src: Sources, key):
    """Troca um dígito do sequencial até sair um número ausente de toda a base."""
    for _ in range(30):
        seq = key[0] if isinstance(key, tuple) else key
        s = str(seq)
        pos = rng.randrange(len(s))
        new = s[:pos] + str((int(s[pos]) + rng.randint(1, 8)) % 10) + s[pos + 1:]
        if new[0] != "0" and not src.number_exists(new):
            return (int(new), key[1]) if isinstance(key, tuple) else int(new)
    return None
