"""Catálogo inédito v1: formas escritas antes de o pipeline suportá-las.

Foi holdout até as âncoras gerais existirem; hoje é conjunto de ITERAÇÃO
(perfis `ineditos` e `ineditos_poluido`).
"""

from __future__ import annotations

import re

from .known import DIPLOMA_FORMS, PARAGRAPHS
from .sources import COURT_LONG, SUMULA_REAL, Catalog, Cite, Sources, clean_relator, fmt_thousands, perturb_number

# ------------------------------------------------------------------ processos

# Nomes inéditos por classe (nenhum aparece nos geradores de val/test).
NOVEL_CLASS = {
    "REsp": ["Resp.", "Rec. Especial", "recurso especial", "RECURSO ESPECIAL", "R. Especial"],
    "AREsp": ["Ag. em REsp", "agravo em recurso especial", "AGRAVO EM RECURSO ESPECIAL"],
    "RHC": ["Rec. em HC", "recurso em habeas corpus", "RECURSO EM HABEAS CORPUS"],
    "RMS": ["Rec. em MS", "recurso em mandado de segurança"],
    "HC": ["habeas corpus", "HABEAS CORPUS", "Hab. Corpus"],
    "Rcl": ["Reclam.", "reclamação", "RECLAMAÇÃO"],
    "REspe": ["Rec. Esp. Eleitoral", "recurso especial eleitoral", "RESPE"],
    "APL": ["Ap. Crim.", "apelação criminal", "APELAÇÃO"],
    "RSE": ["Rec. em Sentido Estrito", "recurso em sentido estrito"],
    "RO": ["Rec. Ordinário", "recurso ordinário"],
    "AI": ["Ag. de Instrumento", "agravo de instrumento"],
}
NOVEL_CHAIN = {
    "AGINT": ["Ag. Int. no", "agravo interno no", "AGINT no"],
    "AGRG": ["Ag. Rg. no", "agravo regimental no", "AGRG no"],
    "ED": ["Emb. Decl. no", "embargos de declaração no", "EDCL no"],
    "EDV": ["Emb. Div. no"],
    "PEXT": ["PExt no"],
}
NOVEL_MARKERS = ["n.", "nº", "n.º", "Nº", "núm.", "número", "n.o"]
COURT_NAMES = {
    "STJ": ["STJ", "Superior Tribunal de Justiça", "E. STJ", "C. STJ"],
    "STF": ["STF", "Supremo Tribunal Federal", "Pretório Excelso", "E. STF"],
    "TSE": ["TSE", "Tribunal Superior Eleitoral", "C. TSE"],
    "STM": ["STM", "Superior Tribunal Militar"],
    "TST": ["TST", "Tribunal Superior do Trabalho", "C. TST"],
}


def _novel_number(rng, key) -> str:
    if isinstance(key, tuple):
        seq, rest = key
        dv, ano, j, tr, orig = rest[:2], rest[2:6], rest[6], rest[7:9], rest[9:]
        return rng.choice([
            f"{seq:07d}-{dv}.{ano}.{j}.{tr}.{orig}",
            f"{seq:07d} - {dv} . {ano} . {j} . {tr} . {orig}",
            f"{seq:07d}.{dv}.{ano}.{j}.{tr}.{orig}",
            f"{seq}-{dv}/{ano}.{j}.{tr}.{orig}",
        ])
    return rng.choice([fmt_thousands(key), str(key), fmt_thousands(key).replace(".", ","),
                       fmt_thousands(key).replace(".", " . ")])


def _render_process(rng, record, key) -> str | None:
    if record.tribunal == "TST":
        chain = "".join({"ED": "ED-", "E": "E-", "AG": "Ag-"}.get(w, "") for w in record.chain)
        prefix = {"RR": "RR", "AIRR": "AIRR", "ARR": "ARR"}.get(record.classe or "", "RR")
        head = rng.choice(["TST - ", "Proc. TST-", "PROCESSO Nº TST-", "autos TST-"])
        return f"{head}{chain}{prefix} - {_novel_number(rng, key)}"
    names = NOVEL_CLASS.get(record.classe)
    if not names:
        return None
    if any(w not in NOVEL_CHAIN for w in record.chain) or len(record.chain) > 3:
        return None
    chain = sorted(set(record.chain), key=record.chain.index)
    prefix = " ".join(rng.choice(NOVEL_CHAIN[w]) for w in chain)
    body = f"{rng.choice(names)} {rng.choice(NOVEL_MARKERS)} {_novel_number(rng, key)}"
    body = f"{prefix} {body}".strip()
    court = rng.choice(COURT_NAMES.get(record.tribunal, [record.tribunal]))
    uf = record.uf
    forms = [
        f"{body}{f'/{uf}' if uf else ''}",
        f"{court}, {body}",
        f"{body} ({court})",
        f"{body}{f', {uf}' if uf else ''}, {court}",
        f"{body}{f' — {uf}' if uf else ''}",
    ]
    return rng.choice(forms)


def gen_novel_process(rng, src: Sources, invent: bool) -> Cite | None:
    ix = src.index
    doc_id = rng.choice(src.unique_short + src.unique_cnj)
    record = ix.records[doc_id]
    key = src.key_of(doc_id)
    if key is None:
        return None
    if invent:
        key = perturb_number(rng, src, key)
        if key is None:
            return None
    text = _render_process(rng, record, key)
    if not text:
        return None
    return Cite(text, "jurisprudencia", "inventada" if invent else "real", "" if invent else doc_id)


# ------------------------------------------------------------------ súmulas

SUMULA_FORMS = [
    "Súm. {n}/{t}", "Súmula {n}, {t}", "verbete nº {n} da Súmula do {t}",
    "enunciado sumular nº {n} do {t}", "Súmula n.º {n} do {T}", "Súmula {n} do Col. {t}",
    "SÚMULA {n} DO {t}", "súmula {n} do {t}",
]
VINC_FORMS = ["SV {n}", "Súmula Vinculante nº {n}/STF", "enunciado vinculante {n}",
              "Súmula Vinculante {n} do Supremo Tribunal Federal", "SÚMULA VINCULANTE {n}"]


def gen_novel_sumula(rng, src: Sources, invent: bool) -> Cite:
    if not invent:
        trib, num, vinc = rng.choice(SUMULA_REAL)
        doc_id = src.index.sumulas[(trib, num, vinc)]
    else:
        trib, vinc = rng.choice([("STJ", False), ("STF", False), ("TST", False), ("STF", True)])
        existing = {n for (t, n, v) in src.index.sumulas if t == trib and v == vinc}
        num = rng.choice([n for n in range(100, 999) if n not in existing])
        doc_id = ""
    form = rng.choice(VINC_FORMS if vinc else SUMULA_FORMS)
    text = form.format(n=num, t=trib, T=COURT_LONG[trib])
    return Cite(text, "jurisprudencia", "inventada" if invent else "real", doc_id)


# ------------------------------------------------------------------ artigos

NOVEL_DIPLOMA = {
    "CF": ["CF/1988", "Carta Magna", "Constituição de 1988", "Lei Maior", "CRFB/88"],
    "CLT": ["Consolidação das Leis Trabalhistas", "Decreto-Lei nº 5.452/1943", "diploma celetista"],
    "CPC": ["CPC/2015", "Código de Processo Civil de 2015", "NCPC"],
    "CPP": ["Código de Processo Penal brasileiro", "Decreto-Lei nº 3.689/1941"],
    "CPM": ["Código Penal Militar (Decreto-Lei nº 1.001/1969)", "CPM/1969"],
    "CC": ["CC/2002", "Código Civil de 2002", "Lei nº 10.406/2002"],
    "CDC": ["Lei nº 8.078/1990", "Código do Consumidor", "CDC/1990"],
    "CE": ["Lei nº 4.737/1965", "Código Eleitoral brasileiro"],
    "LC64/1990": ["LC 64/90", "Lei das Inelegibilidades", "Lei Complementar 64/90"],
}
NOVEL_ARTICLE = [
    "art. {a}, caput, {p} {d}", "art. {a}, inciso I, {p} {d}", "art. {a}, inc. II, {p} {d}",
    "artigo {a} {p} {d}", "ART. {a} {P} {D}", "art. {a}, § 2º, {p} {d}",
    "{d}, art. {a}, caput", "dispositivo do art. {a} {p} {d}", "Art. {a} {p} {d}",
]


def gen_novel_article(rng, src: Sources, invent: bool) -> Cite:
    if not invent:
        (diploma, num), doc_id = rng.choice(sorted(src.index.dispositivos.items()))
    else:
        diploma = rng.choice(sorted(NOVEL_DIPLOMA))
        existing = {n for (d, n) in src.index.dispositivos if d == diploma}
        num = rng.choice([n for n in range(2, 400) if n not in existing])
        doc_id = ""
    name = rng.choice(NOVEL_DIPLOMA[diploma] + DIPLOMA_FORMS[diploma][0])
    art = f"{num}º" if num < 10 else fmt_thousands(num)
    prep = "da" if re.match(r"(Constituição|Consolidação|Lei|Carta|CF|CLT|LC|CRFB)", name) else "do"
    text = rng.choice(NOVEL_ARTICLE).format(a=art, p=prep, d=name, P=prep.upper(), D=name.upper())
    return Cite(text, "lei", "inventada" if invent else "real", doc_id)


# ------------------------------------------------------------------ descritivas

NOVEL_DESCRIPTIVE = [
    "julgamento do {t} em {a}, sob a relatoria do Ministro {r}",
    "precedente de {a} do {t} (Rel. Min. {r})",
    "acórdão relatado pelo Min. {r} no {t} em {a}",
    "julgado de relatoria do Ministro {r} ({t}, {a})",
    "decisão colegiada do {t}, {a}, Relator Ministro {r}",
    "no {t}, em {a}, sob relatoria de {r}",
]


def gen_novel_descriptive(rng, src: Sources, invent: bool) -> Cite:
    (tribunal, ano, relator), records = rng.choice(src.desc_groups)
    relator = clean_relator(relator)
    text = rng.choice(NOVEL_DESCRIPTIVE).format(t=tribunal, a=ano, r=relator)
    if len(records) == 1:
        return Cite(text, "jurisprudencia", "real", records[0].doc_id)
    return Cite(text, "jurisprudencia", "incompleta")


NOVEL_GENERATORS = [
    (lambda rng, src, t: gen_novel_process(rng, src, False), 0.30),
    (lambda rng, src, t: gen_novel_process(rng, src, True), 0.22),
    (lambda rng, src, t: gen_novel_descriptive(rng, src, False), 0.14),
    (lambda rng, src, t: gen_novel_sumula(rng, src, rng.random() < 0.5), 0.12),
    (lambda rng, src, t: gen_novel_article(rng, src, rng.random() < 0.45), 0.22),
]


CATALOG = Catalog(
    generators=NOVEL_GENERATORS,
    paragraphs=PARAGRAPHS[:7],
    distractor=lambda rng, filler: filler(rng.choice(PARAGRAPHS[7:])),
)
