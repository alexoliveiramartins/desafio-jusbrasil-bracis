"""Catálogo inédito v2 — HOLDOUT PROCEDURAL.

Escrito antes das mudanças do pipeline para formatos inéditos, e diferente do
v1 (catalog_v1.py), que passou a ser conjunto de iteração. Para não dar
para "decorar" o teste, as superfícies são geradas por procedimento:

  * abreviação por TRUNCAMENTO ALEATÓRIO de cada palavra ("Recurso" pode sair
    "Rec.", "Recu.", "Recurs."), com trocas de caixa;
  * descritivas com tribunal, ano e relator em ORDEM ALEATÓRIA;
  * frases de contexto e distratores próprios, incluindo armadilhas (número de
    registro do próprio recurso, CPF perto de "habeas corpus", protocolo).

Nunca ajuste o pipeline olhando erros dos perfis `ineditos_v2*`.
"""

from __future__ import annotations

import random

from .known import PARAGRAPHS, PARTES
from .sources import COURT_LONG, SUMULA_REAL, Catalog, Cite, Sources, clean_relator, fmt_thousands, perturb_number

# ------------------------------------------------------------ truncamentos

STOP = {"de", "em", "com", "do", "da", "no", "na", "e"}


def abbreviate_word(rng: random.Random, word: str) -> str:
    """Truncamento com ponto, como se abrevia em peças: 'Especial' -> 'Espec.'."""
    if word.lower() in STOP or len(word) <= 4:
        return word
    low = 2 if rng.random() < 0.15 else 3
    k = rng.randint(low, max(low, min(len(word) - 2, 6)))
    return word[:k] + "."


def render_phrase(rng: random.Random, phrase: str, p_abbrev: float = 0.55) -> str:
    return " ".join(abbreviate_word(rng, w) if rng.random() < p_abbrev else w for w in phrase.split())


def recase(rng: random.Random, text: str) -> str:
    r = rng.random()
    if r < 0.12:
        return text.upper()
    if r < 0.24:
        return text.lower()
    return text


# ------------------------------------------------------------ processos

FULL_CLASS = {
    "REsp": ["Recurso Especial"],
    "AREsp": ["Agravo em Recurso Especial"],
    "RHC": ["Recurso Ordinário em Habeas Corpus", "Recurso em Habeas Corpus"],
    "RMS": ["Recurso Ordinário em Mandado de Segurança", "Recurso em Mandado de Segurança"],
    "HC": ["Habeas Corpus"],
    "Rcl": ["Reclamação"],
    "REspe": ["Recurso Especial Eleitoral"],
    "APL": ["Apelação Criminal", "Apelação"],
    "RSE": ["Recurso em Sentido Estrito"],
    "RO": ["Recurso Ordinário"],
    "AI": ["Agravo de Instrumento"],
    "AR": ["Ação Rescisória"],
    "RE": ["Recurso Extraordinário"],
    "ARE": ["Recurso Extraordinário com Agravo"],
    "AgInt": ["Agravo Interno"],
}
ACRONYM_SWAPS = {"Habeas Corpus": "HC", "Mandado de Segurança": "MS"}
CHAIN_FULL = {
    "AGINT": "Agravo Interno", "AGRG": "Agravo Regimental", "ED": "Embargos de Declaração",
    "EDV": "Embargos de Divergência", "PEXT": "Pedido de Extensão",
}
V2_MARKERS = ["nº", "n.", "nr.", "Nº.", "n°.", "No.", "sob o nº", "de nº", "Num.", "número"]
V2_COURTS = {
    "STJ": ["STJ", "Superior Tribunal de Justiça", "Egrégio STJ", "Colendo Superior Tribunal de Justiça"],
    "STF": ["STF", "Supremo Tribunal Federal", "Excelso Supremo Tribunal Federal", "Suprema Corte"],
    "TSE": ["TSE", "Tribunal Superior Eleitoral", "Colendo TSE"],
    "STM": ["STM", "Superior Tribunal Militar", "Egrégio STM"],
    "TST": ["TST", "Tribunal Superior do Trabalho", "Colendo TST"],
}
UF_FORMS = ["/{uf}", " – {uf}", "({uf})", "-{uf}", " ({uf})", ""]
COURT_FORMS = ["{c}: {b}", "{b} – {c}", "{b}, do {c}", "({c}, {b})", "{b} ({c})", "{b}", "{b}", "{b}"]


def _short(rng, n: int) -> str:
    return rng.choice([
        fmt_thousands(n), fmt_thousands(n), str(n), fmt_thousands(n, " "),
        fmt_thousands(n).replace(".", ". "),
    ])


def _cnj(rng, key) -> str:
    seq, rest = key
    dv, ano, j, tr, orig = rest[:2], rest[2:6], rest[6], rest[7:9], rest[9:]
    s7 = f"{seq:07d}"
    return rng.choice([
        f"{s7}-{dv}.{ano}.{j}.{tr}.{orig}",
        f"{s7}-{dv}.{ano}.{j}{tr}.{orig}",
        f"{s7}{dv}.{ano}.{j}.{tr}.{orig}",
        f"{s7}-{dv}/{ano}.{j}.{tr}.{orig}",
        f"{s7}.{dv}.{ano}.{j}.{tr}.{orig}",
    ])


def _class_phrase(rng, record) -> str | None:
    names = FULL_CLASS.get(record.classe or "")
    if not names:
        return None
    if any(w not in CHAIN_FULL for w in record.chain) or len(record.chain) > 3:
        return None
    phrase = rng.choice(names)
    for full, acro in ACRONYM_SWAPS.items():
        if full in phrase and rng.random() < 0.3:
            phrase = phrase.replace(full, acro)
    parts = [render_phrase(rng, CHAIN_FULL[w]) for w in sorted(set(record.chain), key=record.chain.index)]
    parts.append(render_phrase(rng, phrase))
    connector = rng.choice(["no", "no", "nos", "na", "em"])
    return f" {connector} ".join(parts)


def _render_tst(rng, record, key) -> str | None:
    marks = {"ED": "ED-", "E": "E-", "AG": "Ag-"}
    if any(w not in marks for w in record.chain):
        return None
    chain = "".join(marks[w] for w in record.chain)
    prefix = {"RR": "RR", "AIRR": "AIRR", "ARR": "ARR"}.get(record.classe or "", "RR")
    number = _cnj(rng, key)
    forms = [f"TST/{chain}{prefix}-{number}", f"{chain}{prefix} {number}", f"TST {chain}{prefix} - {number}"]
    if not chain:
        forms += [f"{prefix} nº {number}"]
        if prefix == "RR":
            forms += [f"{render_phrase(rng, 'Recurso de Revista')} nº {number}"]
        if prefix == "AIRR":
            forms += [f"{render_phrase(rng, 'Agravo de Instrumento em Recurso de Revista')} {number}"]
    return rng.choice(forms)


def gen_v2_process(rng, src: Sources, invent: bool) -> Cite | None:
    ix = src.index
    doc_id = rng.choice(src.unique_short + src.unique_cnj)
    record = ix.records[doc_id]
    key = src.key_of(doc_id)
    if key is None:
        return None
    if invent and (key := perturb_number(rng, src, key)) is None:
        return None
    if record.tribunal == "TST":
        text = _render_tst(rng, record, key)
    else:
        phrase = _class_phrase(rng, record)
        if phrase is None:
            return None
        number = _cnj(rng, key) if isinstance(key, tuple) else _short(rng, key)
        uf = rng.choice(UF_FORMS).format(uf=record.uf) if record.uf else ""
        body = recase(rng, f"{phrase} {rng.choice(V2_MARKERS)} ") + f"{number}{uf}"
        court = rng.choice(V2_COURTS.get(record.tribunal, [record.tribunal]))
        text = rng.choice(COURT_FORMS).format(c=court, b=body)
    if not text:
        return None
    return Cite(text, "jurisprudencia", "inventada" if invent else "real", "" if invent else doc_id)


# ------------------------------------------------------------ súmulas

V2_SUMULA = [
    "Súmula de nº {n} do {t}", "Súmula {n}/{T}", "Enunciado de Súmula nº {n} do {t}",
    "Súmula n. {n} do Egrégio {T}", "Súmula-{t} nº {n}",
    "enunciado nº {n} da Súmula de jurisprudência do {t}", "Súmula nº {n} ({t})",
]
V2_VINC = ["Súmula Vinc. {n}", "S.V. nº {n}", "Enunciado Vinculante nº {n} do STF",
           "Súmula Vinculante de nº {n}", "SV nº {n}/STF"]


def gen_v2_sumula(rng, src: Sources, invent: bool) -> Cite:
    if not invent:
        trib, num, vinc = rng.choice(SUMULA_REAL)
        doc_id = src.index.sumulas[(trib, num, vinc)]
    else:
        trib, vinc = rng.choice([("STJ", False), ("STF", False), ("TST", False), ("STF", True)])
        existing = {n for (t, n, v) in src.index.sumulas if t == trib and v == vinc}
        num = rng.choice([n for n in range(100, 999) if n not in existing])
        doc_id = ""
    text = rng.choice(V2_VINC if vinc else V2_SUMULA).format(n=num, t=trib, T=COURT_LONG[trib])
    return Cite(recase(rng, text), "jurisprudencia", "inventada" if invent else "real", doc_id)


# ------------------------------------------------------------ artigos

V2_DIPLOMAS = {
    "CF": [("Texto Constitucional", "do"), ("Constituição Cidadã", "da"), ("Carta da República", "da"),
           ("Constituição da República Federativa do Brasil", "da")],
    "CLT": [("Consolidação Trabalhista", "da"), ("CLT/1943", "da"),
            ("Consolidação das Leis do Trabalho (CLT)", "da")],
    "CPC": [("Código de Processo Civil/2015", "do"), ("codex processual civil", "do"),
            ("estatuto processual civil", "do"), ("CPC de 2015", "do")],
    "CPP": [("Código de Proc. Penal", "do"), ("CPP/1941", "do"), ("estatuto processual penal", "do")],
    "CPM": [("Código Penal Castrense", "do"), ("CPM de 1969", "do")],
    "CC": [("Código Civil/2002", "do"), ("Cód. Civil", "do"), ("codificação civil", "da")],
    "CDC": [("Código de Proteção e Defesa do Consumidor", "do"), ("estatuto consumerista", "do"),
            ("CDC (Lei 8.078/90)", "do")],
    "CE": [("Código Eleitoral (Lei nº 4.737/1965)", "do"), ("Cód. Eleitoral", "do")],
    "LC64/1990": [("LC nº 64, de 1990", "da"), ("Lei de Inelegibilidade", "da"),
                  ("Lei Complementar n. 64, de 18 de maio de 1990", "da")],
}
V2_ARTICLE = [
    "art. {a}, alínea 'b', {p} {d}", "art. {a}, incs. I e II, {p} {d}", "art. {a}, § único, {p} {d}",
    "{d}, em seu art. {a}", "{d} (art. {a})", "art. {a}, inciso III, alínea 'a', {p} {d}",
    "artigo n. {a} {p} {d}",
]


def gen_v2_article(rng, src: Sources, invent: bool) -> Cite:
    if not invent:
        (diploma, num), doc_id = rng.choice(sorted(src.index.dispositivos.items()))
    else:
        diploma = rng.choice(sorted(V2_DIPLOMAS))
        existing = {n for (d, n) in src.index.dispositivos if d == diploma}
        num = rng.choice([n for n in range(2, 400) if n not in existing])
        doc_id = ""
    name, prep = rng.choice(V2_DIPLOMAS[diploma])
    art = f"{num}º" if num < 10 else fmt_thousands(num)
    text = rng.choice(V2_ARTICLE).format(a=art, p=prep, d=name)
    return Cite(recase(rng, text), "lei", "inventada" if invent else "real", doc_id)


# ------------------------------------------------------------ descritivas

DESC_NOUNS = ["julgado", "precedente", "aresto", "acórdão", "decisão", "julgamento"]
DESC_COURT = ["do {t}", "no {t}", "{t}", "({t})"]
DESC_YEAR = ["em {a}", "de {a}", "{a}", "no ano de {a}"]
DESC_REL = ["Rel. Min. {r}", "relator o Ministro {r}", "sob relatoria de {r}",
            "de relatoria do Ministro {r}", "Min. {r}", "Relator: Min. {r}"]


def gen_v2_descriptive(rng, src: Sources, invent: bool) -> Cite:
    (tribunal, ano, relator), records = rng.choice(src.desc_groups)
    relator = clean_relator(relator)
    court = tribunal if rng.random() < 0.6 else rng.choice(V2_COURTS[tribunal][1:])
    if rng.random() < 0.1:
        # Forma difícil: sem palavra-chave de relatoria ("condução").
        text = f"entendimento firmado pelo {court} em {ano}, sob a condução do Ministro {relator}"
    else:
        parts = [rng.choice(DESC_COURT).format(t=court), rng.choice(DESC_YEAR).format(a=ano),
                 rng.choice(DESC_REL).format(r=relator)]
        rng.shuffle(parts)
        sep = rng.choice([", ", " "])
        text = f"{rng.choice(DESC_NOUNS)} " + sep.join(parts)
    text = recase(rng, text) if rng.random() < 0.5 else text
    if len(records) == 1:
        return Cite(text, "jurisprudencia", "real", records[0].doc_id)
    return Cite(text, "jurisprudencia", "incompleta")


V2_GENERATORS = [
    (lambda rng, src, t: gen_v2_process(rng, src, False), 0.30),
    (lambda rng, src, t: gen_v2_process(rng, src, True), 0.22),
    (lambda rng, src, t: gen_v2_descriptive(rng, src, False), 0.14),
    (lambda rng, src, t: gen_v2_sumula(rng, src, rng.random() < 0.5), 0.12),
    (lambda rng, src, t: gen_v2_article(rng, src, rng.random() < 0.45), 0.22),
]

# ------------------------------------------------------------ contexto

V2_PARAGRAPHS = [
    "Confira-se: {c}.",
    "É o que se colhe de {c}, cuja ementa se transcreve.",
    "No mesmo diapasão, {c}.",
    "Ressalte-se o decidido em {c}, que bem delimita a controvérsia.",
    "{c} consolidou a orientação aqui defendida.",
    "Veja-se, a propósito, {c}.",
    "A tese foi acolhida em {c} e deve prevalecer.",
]
# Distratores com armadilhas para a âncora numérica (nenhum é citação).
V2_DISTRACTORS = [
    "O presente recurso especial, registrado sob o nº {reg}, foi distribuído em {data}.",
    "Recurso de apelação interposto às fls. {fls}, tempestivamente.",
    "Habeas corpus impetrado em favor de {parte}, CPF {cpf}.",
    "O agravo de instrumento foi distribuído à 3ª Turma em {data}.",
    "Processo administrativo nº {pa} arquivado.",
    "Protocolo nº {prot} registrado no sistema em {data}.",
]


def v2_distractor(rng: random.Random, parte: str, data: str) -> str:
    return rng.choice(V2_DISTRACTORS).format(
        reg=f"20{rng.randint(10, 25)}/{rng.randint(1, 9999999):07d}-{rng.randint(0, 9)}",
        data=data, fls=rng.randint(10, 900), parte=parte,
        cpf=f"{rng.randint(100, 999)}.{rng.randint(100, 999)}.{rng.randint(100, 999)}-{rng.randint(10, 99)}",
        pa=f"{rng.randint(1000, 99999)}/20{rng.randint(10, 25)}",
        prot=f"20{rng.randint(10, 25)}.{rng.randint(100, 999)}.{rng.randint(100, 999)}.{rng.randint(100, 999)}",
    )


def _distractor(rng: random.Random, filler) -> str:
    if rng.random() < 0.6:
        return v2_distractor(rng, rng.choice(PARTES), filler("{data}"))
    return filler(rng.choice(PARAGRAPHS[7:]))


CATALOG = Catalog(generators=V2_GENERATORS, paragraphs=V2_PARAGRAPHS, distractor=_distractor)
