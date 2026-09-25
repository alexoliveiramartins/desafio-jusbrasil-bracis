"""Catálogo v4: HOLDOUT, escrito antes das melhorias que ele mede.

Nunca ajuste o pipeline olhando erros dos perfis `ineditos_v4*`. O que ele
tem de diferente de v1, v2 e v3:
  * rabo da citação FORA do span ("…, Rel. Min. X, Terceira Turma, j. …");
  * abreviação por sílaba ("Declar.", "Regim.") e cadeias com hífen ("AgInt-REsp");
  * súmula do TST com item ("Súmula 331, IV, do TST");
  * artigo sem preposição ("art. 5º, XXXV, CF") e numeração "5.º";
  * descritivas com data completa, mês por extenso e nome abreviado;
  * duas citações na mesma frase e distratores-armadilha (Informativo,
    composição da turma, autos da origem).

Rótulos: mesmas garantias de sources.py. Descritivas com nome abreviado ou
classe citada são recontadas com esse nome/classe (`Sources.same_relator`).
"""

from __future__ import annotations

import random

from .sources import COURT_LONG, SUMULA_REAL, Catalog, Cite, Sources, clean_relator, fmt_thousands, perturb_number

VOWELS = set("aeiouáéíóúâêôãõAEIOUÁÉÍÓÚÂÊÔÃÕ")
STOP = {"de", "em", "com", "do", "da", "no", "na", "e"}


def syllable_abbrev(rng: random.Random, word: str) -> str:
    """Corta antes de uma vogal que segue consoante: 'Declaração' -> 'Decl.'/'Declar.'."""
    if word.lower() in STOP or len(word) <= 4:
        return word
    cuts = [i for i in range(3, len(word) - 1) if word[i] in VOWELS and word[i - 1] not in VOWELS]
    return word[:rng.choice(cuts)] + "." if cuts else word


def abbreviate(rng: random.Random, phrase: str) -> str:
    return " ".join(syllable_abbrev(rng, w) if rng.random() < 0.7 else w for w in phrase.split())


# ------------------------------------------------------------------ processos

V4_CLASS = {
    "REsp": ("Recurso Especial", ["REsp", "Resp", "RESP", "R.Esp."]),
    "AREsp": ("Agravo em Recurso Especial", ["AREsp", "ARESP", "AResp"]),
    "RHC": ("Recurso Ordinário em Habeas Corpus", ["RHC", "R.H.C."]),
    "RMS": ("Recurso Ordinário em Mandado de Segurança", ["RMS", "R.M.S."]),
    "HC": ("Habeas Corpus", ["HC", "H.C."]),
    "Rcl": ("Reclamação", ["Rcl", "RCL", "Recl."]),
    "RE": ("Recurso Extraordinário", ["RE", "R.E.", "REx"]),
    "ARE": ("Recurso Extraordinário com Agravo", ["ARE"]),
    "REspe": ("Recurso Especial Eleitoral", ["REspe", "RESPE", "REspEl"]),
    "RO": ("Recurso Ordinário", ["RO", "R.O."]),
    "AI": ("Agravo de Instrumento", ["AI", "A.I."]),
    "APL": ("Apelação", ["APL", "Apel."]),
    "RSE": ("Recurso em Sentido Estrito", ["RSE"]),
    "AR": ("Ação Rescisória", ["AR"]),
    "AgInt": ("Agravo Interno", ["AgInt"]),
}
FEMININE = {"Rcl", "APL", "AR"}
V4_CHAIN = {
    "AGINT": ("Agravo Interno", ["AgInt", "AGINT", "Ag. Int."]),
    "AGRG": ("Agravo Regimental", ["AgRg", "AGRG", "AgR", "Ag. Reg."]),
    "ED": ("Embargos de Declaração", ["EDcl", "EDCL", "ED", "Emb. Decl."]),
    "EDV": ("Embargos de Divergência", ["EDv", "EDiv"]),
    "PEXT": ("Pedido de Extensão", ["PExt"]),
}
V4_MARKERS = ["nº ", "n.º ", "n° ", "Nº ", "n. ", "", "", "", "nro. ", "número "]
V4_COURT = {
    "STJ": [("STJ", "o"), ("STJ", "o"), (COURT_LONG["STJ"], "o"), ("Tribunal da Cidadania", "o")],
    "STF": [("STF", "o"), ("STF", "o"), (COURT_LONG["STF"], "o"), ("Excelsa Corte", "a")],
    "TST": [("TST", "o"), ("TST", "o"), (COURT_LONG["TST"], "o"), ("Corte Superior Trabalhista", "a")],
    "TSE": [("TSE", "o"), ("TSE", "o"), (COURT_LONG["TSE"], "o"), ("Corte Superior Eleitoral", "a")],
    "STM": [("STM", "o"), ("STM", "o"), (COURT_LONG["STM"], "o"), ("Corte Castrense", "a")],
}
TURMAS = ["Primeira Turma", "Segunda Turma", "Terceira Turma", "Quarta Turma", "Quinta Turma",
          "Sexta Turma", "Corte Especial", "Primeira Seção", "Tribunal Pleno", "1ª Turma", "3ª Turma"]
MONTHS = ["janeiro", "fevereiro", "março", "abril", "maio", "junho", "julho", "agosto",
          "setembro", "outubro", "novembro", "dezembro"]


def _date(rng: random.Random, year: int | None = None) -> str:
    year = year or rng.randint(2012, 2025)
    return f"{rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}/{year}"


def _element(rng: random.Random, full: str, acronyms: list[str]) -> str:
    r = rng.random()
    if r < 0.55:
        return rng.choice(acronyms)
    if r < 0.8:
        return full
    return abbreviate(rng, full)


def _number(rng: random.Random, key) -> str:
    if isinstance(key, tuple):
        seq, rest = key
        dv, ano, j, tr, orig = rest[:2], rest[2:6], rest[6], rest[7:9], rest[9:]
        return rng.choice([
            f"{seq:07d}-{dv}.{ano}.{j}.{tr}.{orig}", f"{seq:07d}-{dv}.{ano}.{j}.{tr}.{orig}",
            f"{seq}-{dv}.{ano}.{j}.{tr}.{orig}", f"{seq:07d} - {dv}.{ano}.{j}.{tr}.{orig}",
            f"{seq:07d}{dv}{ano}{j}{tr}{orig}", f"{seq:07d}.{dv}.{ano}.{j}.{tr}.{orig}",
        ])
    return rng.choice([fmt_thousands(key), fmt_thousands(key), str(key),
                       fmt_thousands(key).replace(".", ". "), fmt_thousands(key, " ")])


def _tst(rng: random.Random, record, key) -> str | None:
    marks = {"ED": "ED", "E": "E", "AG": "Ag"}
    if any(w not in marks for w in record.chain):
        return None
    prefix = {"RR": "RR", "AIRR": "AIRR", "ARR": "ARR"}.get(record.classe or "", "RR")
    chain = "-".join([marks[w] for w in record.chain] + [prefix])
    number = _number(rng, key)
    return rng.choice([f"{chain} - {number}", f"TST-{chain} {number}", f"TST-{chain}-{number}",
                       f"processo TST-{chain}-{number}", f"{chain} nº {number}"])


def _tail(rng: random.Random, relator: str | None) -> str:
    """Informações depois da citação, fora do span anotado."""
    rel = relator or "Fulano de Tal"
    return rng.choice([
        f", Rel. Min. {rel}, {rng.choice(TURMAS)}, julgado em {_date(rng)}, DJe {_date(rng)}",
        f", Rel. Min. {rel}, {rng.choice(TURMAS)}, j. {_date(rng)}",
        f", relator Ministro {rel}, DJe de {_date(rng)}",
        f" (Rel. Min. {rel}, {rng.choice(TURMAS)}, DJe {_date(rng)})",
        f", {rng.choice(TURMAS)}, julgado em {_date(rng)}",
        f", DJe {_date(rng)}",
    ])


def gen_v4_process(rng: random.Random, src: Sources, invent: bool) -> Cite | None:
    ix = src.index
    doc_id = rng.choice(src.unique_short + src.unique_cnj)
    record = ix.records[doc_id]
    key = src.key_of(doc_id)
    if key is None or (invent and (key := perturb_number(rng, src, key)) is None):
        return None
    if record.tribunal == "TST":
        text = _tst(rng, record, key)
    else:
        if record.classe not in V4_CLASS or len(record.chain) > 3 or any(w not in V4_CHAIN for w in record.chain):
            return None
        chain = sorted(set(record.chain), key=record.chain.index)
        parts = [_element(rng, *V4_CHAIN[w]) for w in chain] + [_element(rng, *V4_CLASS[record.classe])]
        hyphen = rng.random() < 0.15 and all(len(p) <= 6 and " " not in p for p in parts)
        text = parts[0]
        for i, part in enumerate(parts[1:], 1):
            if hyphen:
                text += "-" + part
                continue
            following = chain[i] if i < len(chain) else record.classe
            natural = " na " if following in FEMININE else " nos " if following == "ED" else " no "
            text += (natural if rng.random() < 0.85 else " em ") + part
        uf = rng.choice(["/{u}", "/{u}", "-{u}", " - {u}", " ({u})", "/ {u}", ""]).format(u=record.uf) if record.uf else ""
        text = f"{text} {rng.choice(V4_MARKERS)}{_number(rng, key)}{uf}"
        if rng.random() < 0.4:
            court, _ = rng.choice(V4_COURT[record.tribunal])
            text = rng.choice([f"{text} ({court})", f"{text}, {court}", f"{court}, {text}", f"{text} – {court}"])
    if not text:
        return None
    after = _tail(rng, clean_relator(record.relator or "") or None) if rng.random() < 0.45 else ""
    return Cite(text, "jurisprudencia", "inventada" if invent else "real", "" if invent else doc_id, after)


# ------------------------------------------------------------------ súmulas

V4_SUMULA = ["Súmula nº {n} do {L}", "Súmula {n} do c. {t}", "súmula n.º {n}, do {t}", "Súm. nº {n} do {t}",
             "verbete sumular nº {n} do {t}", "{t}, Súmula {n}", "Súmula {n}/{t}", "Súmula do {t} nº {n}",
             "enunciado nº {n} da Súmula do {L}"]
V4_SUMULA_TST_ITEM = ["Súmula {n}, item {i}, do TST", "Súmula nº {n}, {i}, do TST", "Súmula {n}/TST, item {i}"]
V4_VINC = ["Súmula Vinculante n.º {n} do STF", "enunciado {n} da Súmula Vinculante", "SV n. {n}",
           "Súmula Vinculante nº {n} do Supremo Tribunal Federal", "Súmula Vinculante {n}/STF"]


def gen_v4_sumula(rng: random.Random, src: Sources, invent: bool) -> Cite:
    if not invent:
        trib, num, vinc = rng.choice(SUMULA_REAL)
        doc_id = src.index.sumulas[(trib, num, vinc)]
    else:
        trib, vinc = rng.choice([("STJ", False), ("STF", False), ("TST", False), ("TSE", False), ("STF", True)])
        existing = {n for (t, n, v) in src.index.sumulas if t == trib and v == vinc}
        num = rng.choice([n for n in range(1, 700) if n not in existing])
        doc_id = ""
    if vinc:
        form = rng.choice(V4_VINC)
    elif trib == "TST" and rng.random() < 0.5:
        form = rng.choice(V4_SUMULA_TST_ITEM)
    else:
        form = rng.choice(V4_SUMULA)
    text = form.format(n=num, t=trib, L=COURT_LONG[trib], i=rng.choice(["I", "II", "III", "IV", "V", "VI"]))
    if rng.random() < 0.1:
        text = text.upper()
    return Cite(text, "jurisprudencia", "inventada" if invent else "real", doc_id)


# ------------------------------------------------------------------ artigos

V4_DIPLOMAS = {
    "CF": [("CF", "da"), ("CF/88", "da"), ("CRFB/1988", "da"), ("Constituição Federal de 1988", "da"),
           ("Constituição da República de 1988", "da"), ("Carta de 1988", "da"), ("Lei Fundamental", "da")],
    "CPC": [("CPC", "do"), ("CPC/15", "do"), ("Novo CPC", "do"), ("Código de Processo Civil vigente", "do"),
            ("diploma processual civil", "do"), ("Lei 13.105/15", "da")],
    "CLT": [("CLT", "da"), ("Consolidação das Leis Trabalhistas", "da"), ("Decreto-Lei 5.452/43", "do"),
            ("texto consolidado", "do")],
    "CPP": [("CPP", "do"), ("Código de Processo Penal", "do"), ("Decreto-Lei 3.689/41", "do"),
            ("diploma processual penal", "do")],
    "CPM": [("CPM", "do"), ("Código Penal Militar", "do"), ("Decreto-Lei nº 1.001/69", "do")],
    "CC": [("CC", "do"), ("CC/02", "do"), ("Código Civil de 2002", "do"), ("Lei 10.406/02", "da"),
           ("Código Civil vigente", "do")],
    "CDC": [("CDC", "do"), ("Lei 8.078/90", "da"), ("Código Consumerista", "do"),
            ("Código de Defesa do Consumidor (CDC)", "do")],
    "CE": [("Código Eleitoral", "do"), ("Lei 4.737/65", "da")],
    "LC64/1990": [("LC 64/90", "da"), ("Lei Complementar nº 64/1990", "da"), ("LC n. 64, de 1990", "da")],
}
V4_ENUM = ["", "", ", caput", ", I", ", II", ", inciso LV", ", LV", ", § 1º", ", § 3º, II", ", parágrafo único",
           ", II, 'a'", ", alínea 'a'", ", incisos I e II", ", inc. IV"]
V4_ARTICLE = ["art. {a}{e}, {p} {d}", "art. {a}{e} {p} {d}", "artigo {a}{e}, {p} {d}", "{d}, art. {a}{e}",
              "art. {a}{e}, {d}", "Art. {a}{e} {p} {d}"]


def gen_v4_article(rng: random.Random, src: Sources, invent: bool) -> Cite:
    if not invent:
        (diploma, num), doc_id = rng.choice(sorted(src.index.dispositivos.items()))
    else:
        diploma = rng.choice(sorted(V4_DIPLOMAS))
        existing = {n for (d, n) in src.index.dispositivos if d == diploma}
        num = rng.choice([n for n in range(2, 400) if n not in existing])
        doc_id = ""
    name, prep = rng.choice(V4_DIPLOMAS[diploma])
    art = rng.choice([f"{num}º", f"{num}.º", f"{num}°", str(num)]) if num < 10 else rng.choice([fmt_thousands(num), str(num)])
    enum = rng.choice(V4_ENUM)
    text = rng.choice(V4_ARTICLE).format(a=art, e=enum, p=prep, d=name).replace(", , ", ", ")
    if rng.random() < 0.1:
        text = text.upper()
    return Cite(text, "lei", "inventada" if invent else "real", doc_id)


# ------------------------------------------------------------------ descritivas

V4_REL = ["Rel. Min. {r}", "Relator Min. {r}", "Rel.ª Min.ª {r}", "Ministro Relator {r}", "da lavra do Ministro {r}",
          "sob a relatoria do Min. {r}", "relatado pelo Ministro {r}", "voto condutor do Ministro {r}"]
V4_NOUN = ["julgado", "precedente", "acórdão", "aresto", "decisão", "paradigma"]


def _year_form(rng: random.Random, ano: int) -> str:
    return rng.choice([f"em {ano}", f"de {ano}", f"({ano})", f"julgado em {_date(rng, ano)}",
                       f"em {rng.choice(MONTHS)} de {ano}", f"no ano de {ano}", f"j. {_date(rng, ano)}"])


def _name_form(rng: random.Random, relator: str) -> str:
    words = relator.split()
    r = rng.random()
    if len(words) >= 3 and r < 0.25:
        name = f"{words[0]} {words[-1]}"
    elif len(words) >= 3 and r < 0.4:
        name = " ".join(words[-2:])
    else:
        name = relator
    r = rng.random()
    return name.upper() if r < 0.2 else name.title() if r < 0.4 else name


def gen_v4_descriptive(rng: random.Random, src: Sources, invent: bool) -> Cite | None:
    (tribunal, ano, _), records = rng.choice(src.desc_groups)
    record = rng.choice(records)
    name = _name_form(rng, clean_relator(record.relator))
    classe = record.classe if record.classe in V4_CLASS and rng.random() < 0.25 else None
    matches = src.same_relator(tribunal, ano, name, classe)
    if not matches:
        return None
    court, article = rng.choice(V4_COURT[tribunal])
    court_part = rng.choice([f"d{article} {court}", f"n{article} {court}", court, f"({court})", f"pel{article} {court}"])
    year, rel = _year_form(rng, ano), rng.choice(V4_REL).format(r=name)
    head = rng.choice(V4_CLASS[classe][1]) if classe else rng.choice(V4_NOUN)
    if rng.random() < 0.15:
        text = f"{head} d{article} {court} ({rel}, {year})"
    else:
        parts = [court_part, year, rel]
        rng.shuffle(parts)
        text = f"{head} " + rng.choice([", ", " "]).join(parts)
    if len(matches) == 1:
        return Cite(text, "jurisprudencia", "real", matches[0].doc_id)
    return Cite(text, "jurisprudencia", "incompleta")


V4_GENERATORS = [
    (lambda rng, src, t: gen_v4_process(rng, src, False), 0.28),
    (lambda rng, src, t: gen_v4_process(rng, src, True), 0.22),
    (lambda rng, src, t: gen_v4_descriptive(rng, src, False), 0.16),
    (lambda rng, src, t: gen_v4_sumula(rng, src, rng.random() < 0.5), 0.12),
    (lambda rng, src, t: gen_v4_article(rng, src, rng.random() < 0.45), 0.22),
]

# ------------------------------------------------------------------ contexto

V4_PARAGRAPHS = [
    "Cf. {c}.",
    "Nessa linha: {c}.",
    "Invoca-se, ainda, {c}, aplicável à hipótese.",
    "À luz de {c}, impõe-se o desprovimento.",
    "Precedente: {c}.",
    "O tema foi enfrentado em {c}, cujo entendimento se adota.",
    "¹ {c}.",
    "— {c};",
    "Segundo {c}, não há falar em nulidade.",
]
V4_DENSE = [
    "Precedentes: {c}; {c2}.",
    "Cf. {c} e {c2}.",
    "No mesmo sentido: {c}, e também {c2}.",
    "(v.g., {c}; {c2})",
]
V4_DISTRACTORS = [
    "Os autos nº {cnj8} tramitam na origem desde {data}.",
    "O Informativo nº {n3} do STJ, de {ano}, noticiou o julgamento.",
    "Votaram com o Relator os Ministros {r1} e {r2}, na sessão de {data}.",
    "Participaram do julgamento, no STJ, em {ano}, os Ministros {r1} e {r2}.",
    "A Lei nº 13.105/2015 entrou em vigor em março de 2016.",
    "O Recurso Especial foi interposto em {data} e admitido em {data}.",
    "Ação ajuizada em {data}, distribuída por dependência ao processo nº {n5}/{ano}.",
    "Certidão de fls. {fls}: prazo de 15 dias úteis.",
]


def _distractor(rng: random.Random, src_relators: list[str]) -> str:
    return rng.choice(V4_DISTRACTORS).format(
        cnj8=f"{rng.randint(1000000, 9999999)}-{rng.randint(10, 99)}.20{rng.randint(10, 25)}.8."
             f"{rng.randint(1, 27):02d}.{rng.randint(1, 9999):04d}",
        data=_date(rng), ano=rng.randint(2012, 2025), n3=rng.randint(500, 820), n5=rng.randint(10000, 99999),
        fls=rng.randint(10, 900), r1=rng.choice(src_relators), r2=rng.choice(src_relators),
    )


RELATORS = ["Herman Benjamin", "Nancy Andrighi", "Og Fernandes", "Luis Felipe Salomão", "Regina Helena Costa"]

CATALOG = Catalog(
    generators=V4_GENERATORS,
    paragraphs=V4_PARAGRAPHS,
    distractor=lambda rng, filler: _distractor(rng, RELATORS),
    dense_paragraphs=V4_DENSE,
)
