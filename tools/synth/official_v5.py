"""Catálogo v5 ("oficial"): peças no molde do gerador oficial, com superfícies novas.

    python -m tools.synth.official_v5 --split v5_iter --seed 1
    python -m tools.synth.official_v5 --split v5_holdout --seed 101 --holdout
    python -m tools.synth.official_v5 --split v5_holdout_ruido --seed 201 --holdout --noise 1.8

Escrito em 28/09/2026, ANTES da camada de NLP (src/nlp.py) e sem olhar erros
do pipeline. O que ele tem de diferente dos catálogos v1–v4:

  * a estrutura da peça segue a amostra oficial: cabeçalho por matéria com
    distratores (autos CNJ, protocolo, OAB, CPF), seções numeradas, prosa com
    frases de enchimento, referências vagas que não são citação e quebra de
    linha dura a cada ~100 caracteres (que cai também dentro das citações);
  * nível 1 só com formas canônicas; nível 2 com as variações descritas no
    PDF do desafio (abreviação, número sem ponto ou com espaço, separador de
    UF, OCR 0↔O 1↔l 5↔S m↔rn, quebra de linha no identificador) e OCR no corpo;
  * formas `HOLDOUT` (apelidos de diplomas, lei pelo número, súmula por
    extenso, descritivas em ordem nova, tribunal junto do número) só entram
    com --holdout. Nunca ajuste nada olhando os erros delas.

Rótulos: mesmas garantias de sources.py (número único na base para `real`,
número ausente de todos os cabeçalhos para `inventada`, descritivas contadas
por tribunal + ano + tokens do nome escrito [+ classe citada]).
"""

from __future__ import annotations

import argparse
import csv
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .sources import COURT_LONG, SUMULA_REAL, Cite, Sources, clean_relator, fmt_thousands, perturb_number

# ------------------------------------------------------------------ matérias

MATTERS = {
    "cível": {"courts": {"STJ": ["REsp", "AREsp", "RMS", "AR"], "STF": ["RE", "ARE", "Rcl"]},
              "diplomas": ["CC", "CPC", "CDC", "CF"], "sumulas": ["STJ", "STF"],
              "leis_fora": [("Lei nº 8.245/1991", "Lei do Inquilinato"), ("Lei nº 9.099/1995", None)]},
    "penal": {"courts": {"STJ": ["HC", "RHC", "REsp", "AREsp"], "STF": ["HC", "Rcl", "RE"]},
              "diplomas": ["CPP", "CF"], "sumulas": ["STJ", "STF"],
              "leis_fora": [("Lei nº 11.343/2006", "Lei de Drogas"), ("Lei nº 7.210/1984", "Lei de Execução Penal")]},
    "trabalhista": {"courts": {"TST": ["RR", "AIRR", "ARR"]}, "diplomas": ["CLT", "CF"], "sumulas": ["TST"],
                    "leis_fora": [("Lei nº 13.467/2017", "Reforma Trabalhista"), ("Lei nº 5.584/1970", None)]},
    "eleitoral": {"courts": {"TSE": ["REspe", "RO", "AI", "AR", "HC"]}, "diplomas": ["CE", "LC64/1990", "CF"],
                  "sumulas": ["TSE"],
                  "leis_fora": [("Lei nº 9.504/1997", "Lei das Eleições"), ("Lei nº 9.096/1995", None)]},
    "militar": {"courts": {"STM": ["APL", "RSE", "HC", "AgInt", "Rcl"]}, "diplomas": ["CPM", "CF"],
                "sumulas": [], "leis_fora": [("Lei nº 8.457/1992", None), ("Lei nº 6.880/1980", "Estatuto dos Militares")]},
}

# ------------------------------------------------------------------ superfícies
# Cada entrada: (formas de iteração, formas exclusivas de holdout).

CLASS_FORMS = {
    "REsp": (["REsp", "Recurso Especial"], ["Resp.", "Rec. Especial", "Recurso Especial Cível"]),
    "AREsp": (["AREsp", "Agravo em Recurso Especial"], ["Ag. em REsp", "Agravo em REsp"]),
    "RMS": (["RMS", "Recurso em Mandado de Segurança"], ["Rec. em MS", "Recurso Ordinário em MS"]),
    "AR": (["AR", "Ação Rescisória"], ["Rescisória"]),
    "RE": (["RE", "Recurso Extraordinário"], ["Rec. Extraordinário", "R. Extraordinário"]),
    "ARE": (["ARE", "Recurso Extraordinário com Agravo"], ["Ag. em RE"]),
    "Rcl": (["Rcl", "Reclamação"], ["Reclamação Constitucional", "Recl"]),
    "HC": (["HC", "Habeas Corpus"], ["Habeas Corpus", "H. C."]),
    "RHC": (["RHC", "Recurso em Habeas Corpus"], ["Recurso Ordinário em Habeas Corpus", "Rec. em HC"]),
    "REspe": (["REspe", "Recurso Especial Eleitoral"], ["REspEl", "Rec. Esp. Eleitoral"]),
    "RO": (["RO", "Recurso Ordinário"], ["Rec. Ordinário", "RO-El"]),
    "AI": (["AI", "Agravo de Instrumento"], ["Ag. de Instrumento"]),
    "APL": (["APL", "Apelação"], ["Apelação Criminal", "Ap."]),
    "RSE": (["RSE", "Recurso em Sentido Estrito"], ["Rec. em Sentido Estrito"]),
    "AgInt": (["AgInt", "Agravo Interno"], ["Ag. Interno"]),
}
# Abreviações do nível 2 (PDF: "REsp / R.Esp. / Recurso Especial").
CLASS_FORMS_N2 = {
    "REsp": ["RESP", "Rec. Esp.", "R.Esp.", "REsp."], "AREsp": ["ARESP", "A.REsp", "AResp"],
    "RMS": ["R.M.S.", "RMS."], "RE": ["RE.", "R.E."], "Rcl": ["RCL", "Recl.", "Rcl."],
    "HC": ["H.C.", "HC."], "RHC": ["R.H.C.", "RHC."], "REspe": ["RESPE", "REspe."],
    "APL": ["Apel.", "APL."], "RSE": ["R.S.E."], "AgInt": ["AGINT", "Ag.Int."],
}
FEMININE = {"Rcl", "APL", "AR", "RMS"}
CHAIN_FORMS = {
    "AGINT": (["AgInt", "Agravo Interno"], ["Ag. Int.", "AGINT"]),
    "AGRG": (["AgRg", "Agravo Regimental", "AgR"], ["Ag. Reg.", "AG.REG."]),
    "ED": (["EDcl", "Embargos de Declaração"], ["Emb. Decl.", "EDs"]),
    "EDV": (["EDv", "Embargos de Divergência"], ["EDiv"]),
    "PEXT": (["PExt"], ["Pedido de Extensão"]),
}
TST_CHAIN = {"ED": "ED", "E": "E", "AG": "Ag"}
MARKERS_N1 = ["nº ", "nº ", "", "n. "]
MARKERS_N2 = ["n° ", "No ", "Nº ", "n. ", "nº  ", "N. ", "", "n.º "]
UF_N1 = ["/{uf}"]
UF_N2 = ["/{uf}", " - {uf}", " ({uf})", "/ {uf}", " – {uf}", "-{uf}", "- {uf}"]
UF_HOLDOUT = [", {uf}", " — {uf}", "/{uf}."]
COURT_SUFFIX_HOLDOUT = [" do {t}", " ({t})", ", {t}", " – {t}"]

SUMULA_ITER = ["Súmula {n} do {t}", "Súmula nº {n} do {t}", "Súm. {n} do {t}", "SÚMULA {n} do {t}"]
SUMULA_HOLDOUT = ["Súmula {n}/{t}", "Enunciado nº {n} da Súmula do {t}", "verbete sumular nº {n} do {t}",
                  "Súmula {n} do {longo}", "Súmula n. {n}, {t}"]
SV_ITER = ["Súmula Vinculante {n}", "Súmula Vinculante nº {n}"]
SV_HOLDOUT = ["SV {n}", "Súmula Vinculante n. {n} do STF", "Enunciado Vinculante {n}"]

DIPLOMA_ITER = {
    "CF": ["Constituição Federal", "Constituição da República", "CF"],
    "CLT": ["CLT", "Consolidação das Leis do Trabalho"], "CPC": ["CPC", "Código de Processo Civil"],
    "CPP": ["CPP", "Código de Processo Penal"], "CPM": ["Código Penal Militar"], "CC": ["Código Civil"],
    "CDC": ["Código de Defesa do Consumidor", "CDC"], "CE": ["Código Eleitoral"],
    "LC64/1990": ["Lei Complementar nº 64/1990"],
}
DIPLOMA_HOLDOUT = {
    "CF": ["Carta Magna", "Lei Maior", "CF/88", "CRFB/88"], "CLT": ["Decreto-Lei nº 5.452/1943", "diploma consolidado"],
    "CPC": ["CPC/2015", "Lei nº 13.105/2015", "Código de Processo Civil de 2015"], "CPP": ["Decreto-Lei nº 3.689/1941"],
    "CPM": ["CPM", "Decreto-Lei nº 1.001/1969"], "CC": ["CC/2002", "Lei nº 10.406/2002", "Código Civil de 2002"],
    "CDC": ["Lei nº 8.078/1990"], "CE": ["Lei nº 4.737/1965"], "LC64/1990": ["LC nº 64/90", "Lei das Inelegibilidades"],
}
FEM_DIPLOMA = ("Constituição", "Consolidação", "Lei", "CF", "CLT", "Carta", "CRFB", "LC")
INCISOS = {"CF": ["LV", "XXXV", "IX", "XXIX", "II"], "CPC": ["I", "II"], "CLT": ["§ 1º-A", "§ 2º"],
           "LC64/1990": ["I, 'g'", "I, 'e'"], "CC": [], "CDC": ["§ 3º"], "CPP": [], "CPM": [], "CE": []}

DESC_ITER = [
    "julgado do {t} proferido em {a} pela relatoria de {r}",
    "precedente do {t} de {a}, da relatoria de {r}",
    "{classe} do {t}, de {a}, Rel. Min. {r}",
    "{sigla} de {a}, Rel. Min. {r}",
]
DESC_HOLDOUT = [
    "acórdão do {longo} de {a}, relatado pelo Ministro {r}",
    "decisão proferida pelo {t} em {a}, sob a relatoria do Min. {r}",
    "{t}, {a}, Rel. Min. {r}",
    "julgamento de {a} do {t} (Rel. Min. {r})",
]
TEMA_ITER = ["Tema {n} da repercussão geral"]
TEMA_HOLDOUT = ["Tema Repetitivo nº {n} do STJ", "Tema {n}/STF", "tema de repercussão geral nº {n}"]

# ------------------------------------------------------------------ texto da peça

HEADS = {
    "cível": ["EGRÉGIO SUPERIOR TRIBUNAL DE JUSTIÇA", "EXCELENTÍSSIMO SENHOR MINISTRO RELATOR",
              "SUPREMO TRIBUNAL FEDERAL"],
    "penal": ["MINISTÉRIO PÚBLICO FEDERAL\nPROCURADORIA-GERAL DA REPÚBLICA", "EGRÉGIO SUPERIOR TRIBUNAL DE JUSTIÇA",
              "DEFENSORIA PÚBLICA DA UNIÃO"],
    "trabalhista": ["EXCELENTÍSSIMO SENHOR MINISTRO RELATOR DO TRIBUNAL SUPERIOR DO TRABALHO",
                    "MINISTÉRIO PÚBLICO DO TRABALHO\nPROCURADORIA-GERAL DO TRABALHO"],
    "eleitoral": ["MINISTÉRIO PÚBLICO ELEITORAL\nPROCURADORIA-GERAL ELEITORAL", "TRIBUNAL SUPERIOR ELEITORAL"],
    "militar": ["SUPERIOR TRIBUNAL MILITAR", "MINISTÉRIO PÚBLICO MILITAR\nPROCURADORIA-GERAL DE JUSTIÇA MILITAR"],
}
TITLES = ["PARECER", "CONTRARRAZÕES", "MEMORIAL", "DECISÃO MONOCRÁTICA", "RAZÕES DE AGRAVO INTERNO",
          "MANIFESTAÇÃO", "EMBARGOS DE DECLARAÇÃO"]
PEOPLE = ["ANA LÚCIA FERREIRA", "CARLOS EDUARDO BRANDÃO", "TRANSPORTADORA RIO CLARO LTDA", "JOÃO BATISTA NUNES",
          "HELENA MARQUES DE ALMEIDA", "CONSTRUTORA HORIZONTE S.A.", "PAULO ROBERTO SIQUEIRA", "União",
          "Ministério Público Federal", "BANCO REGIONAL DO CENTRO-OESTE S.A.", "MARCOS VINÍCIUS TEIXEIRA"]
CITIES = ["Brasília", "Porto Alegre", "Recife", "Belo Horizonte", "Curitiba", "Manaus", "Goiânia", "Salvador"]
MONTHS = ["janeiro", "fevereiro", "março", "abril", "maio", "junho", "julho", "agosto", "setembro", "outubro",
          "novembro", "dezembro"]
SECTIONS = [["I — SÍNTESE DA DEMANDA", "II — FUNDAMENTAÇÃO", "III — CONCLUSÃO"],
            ["1. DOS FATOS", "2. DO MÉRITO", "3. DOS PEDIDOS"],
            ["I - RELATÓRIO", "II - DO CABIMENTO", "III - DO MÉRITO", "IV - CONCLUSÃO"],
            ["DOS FATOS", "DAS RAZÕES JURÍDICAS", "DO PEDIDO"]]
FILLERS = [
    "A questão devolvida a esta instância é exclusivamente de direito.",
    "Não há controvérsia sobre a moldura fática fixada pelas instâncias ordinárias.",
    "O acórdão recorrido enfrentou todos os pontos relevantes da controvérsia.",
    "A parte adversa não trouxe argumento capaz de infirmar essa conclusão.",
    "Registre-se que a matéria foi devidamente prequestionada.",
    "Eventual divergência de interpretação não se confunde com negativa de prestação jurisdicional.",
    "O tema exige leitura sistemática das normas aplicáveis.",
    "A tese defendida pela parte, contudo, não resiste a exame mais detido.",
    "Os documentos juntados confirmam a narrativa apresentada na inicial.",
    "A instrução processual transcorreu regularmente, sem qualquer nulidade arguida a tempo.",
    "Impõe-se, assim, a manutenção do que decidido na origem.",
    "Tal circunstância, por si só, já seria suficiente para o desfecho proposto.",
]
VAGUE = [  # referências vagas: NÃO são citação no gabarito
    "A orientação consolidada desta Corte Superior caminha no mesmo sentido.",
    "O enunciado sumular pertinente afasta a pretensão recursal.",
    "Aplica-se o dispositivo legal de regência, cuja literalidade não comporta outra leitura.",
    "Há reiterados julgados do Superior Tribunal de Justiça nessa linha.",
    "A matéria já foi decidida em sede de recurso repetitivo, cuja tese vincula os demais órgãos.",
    "A jurisprudência iterativa dos tribunais superiores respalda a conclusão.",
    "O precedente qualificado invocado pela parte não guarda pertinência com o caso.",
]
FACTS = {
    "cível": ["A autora ajuizou ação de indenização por danos materiais e morais em razão de falha na prestação "
              "do serviço, julgada parcialmente procedente em primeiro grau.",
              "O contrato de locação foi rescindido unilateralmente, com retenção indevida da caução prestada."],
    "penal": ["O paciente foi preso em flagrante e teve a prisão convertida em preventiva sem fundamentação "
              "concreta quanto ao risco de reiteração.",
              "A denúncia imputa ao acusado a prática de crime patrimonial, sem individualização da conduta."],
    "trabalhista": ["O reclamante postula horas extras e reflexos, alegando jornada superior à contratada, "
                    "além de diferenças de adicional noturno.",
                    "A reclamada sustenta a validade do banco de horas instituído por norma coletiva."],
    "eleitoral": ["O candidato teve o registro indeferido por suposta inelegibilidade decorrente de "
                  "rejeição de contas pelo órgão competente.",
                  "A representação aponta abuso do poder econômico no período de pré-campanha."],
    "militar": ["O acusado, militar da ativa, foi denunciado por deserção após ausentar-se da unidade por "
                "mais de oito dias sem autorização.",
                "A defesa sustenta a atipicidade da conduta e a ausência de dolo específico."],
}
# Frases de citação por tipo: "{o}" concorda com o gênero da citação.
FRAME_PROCESS = [
    "Nessa linha, confira-se {o} {c}, que examinou hipótese idêntica.",
    "É o que se extrai d{o} {c}, de fundamentação irretocável.",
    "A orientação foi reafirmada n{o} {c}.",
    "Não por acaso, {o} {c} rejeitou pretensão semelhante.",
    "Colhe-se d{o} {c} a mesma conclusão aqui defendida.",
    "Tal entendimento foi aplicado n{o} {c}, em caso análogo.",
    "Merece destaque {o} {c}, cujo voto condutor enfrentou a questão.",
]
FRAME_SUMULA = [
    "Incide, na espécie, a {c}.",
    "A pretensão esbarra na {c}, que se aplica por analogia.",
    "Aplica-se ao caso o entendimento cristalizado na {c}.",
    "O óbice da {c} impede o conhecimento do recurso.",
]
FRAME_ARTICLE = [
    "A controvérsia deve ser resolvida à luz do {c}.",
    "O {c} é expresso ao disciplinar a matéria.",
    "Houve violação direta ao {c}, como demonstrado.",
    "A solução decorre da leitura do {c}.",
    "Nos termos do {c}, a pretensão não prospera.",
]
FRAME_DESC = [
    "Na mesma direção, o {c} assentou premissa idêntica.",
    "Vale lembrar o {c}, em que a questão foi enfrentada.",
    "Esse foi o entendimento adotado no {c}.",
]
FRAME_TEMA = ["A tese fixada no {c} não alcança a hipótese dos autos.", "O {c} não guarda relação com o caso."]


# ------------------------------------------------------------------ geradores de citação

@dataclass
class Ctx:
    rng: random.Random
    src: Sources
    matter: str
    level: int
    holdout: bool

    def pick(self, iter_forms: list, holdout_forms: list, n2_forms: list | None = None):
        """Nível 1: formas canônicas (a primeira de cada lista); nível 2 e holdout ampliam."""
        if self.level == 1 and not (self.holdout and holdout_forms and self.rng.random() < 0.3):
            return self.rng.choice(iter_forms[:2])
        pool = list(iter_forms) + list(n2_forms or [])
        if self.holdout and holdout_forms and self.rng.random() < 0.5:
            pool = list(holdout_forms)
        return self.rng.choice(pool)


def _cnj(key, rng: random.Random, level: int, holdout: bool) -> str:
    seq, rest = key
    dv, ano, j, tr, orig = rest[:2], rest[2:6], rest[6], rest[7:9], rest[9:]
    forms = [f"{seq:07d}-{dv}.{ano}.{j}.{tr}.{orig}"]
    if level == 2:
        forms += [f"{seq}-{dv}.{ano}.{j}.{tr}.{orig}", f"{seq:07d}-{dv} {ano} {j} {tr} {orig}",
                  f"{seq:07d}-{dv}{ano}{j}{tr}{orig}", f"{seq:07d}-{dv}. {ano}.{j}.{tr}.{orig}"]
    if holdout and rng.random() < 0.4:
        forms = [f"{seq:07d}{dv}{ano}{j}{tr}{orig}", f"{seq:07d}.{dv}.{ano}.{j}.{tr}.{orig}",
                 f"{seq:07d} - {dv}.{ano}.{j}.{tr}.{orig}"]
    return rng.choice(forms)


def _short(n: int, rng: random.Random, level: int, holdout: bool) -> str:
    forms = [fmt_thousands(n)]
    if level == 2:
        forms += [str(n), fmt_thousands(n).replace(".", ". ", 1), fmt_thousands(n, " ")]
    if holdout and rng.random() < 0.3:
        forms = [fmt_thousands(n).replace(".", ","), fmt_thousands(n, " . ")]
    return rng.choice(forms)


def _chain(ctx: Ctx, chain: tuple[str, ...]) -> str | None:
    """Recursos de fora para dentro: ('AGINT', 'ED') -> "EDcl no AgInt no"."""
    if any(c not in CHAIN_FORMS for c in chain) or len(chain) > 3:
        return None
    words = [ctx.pick(*CHAIN_FORMS[c]) for c in reversed(chain)]
    joiner = " no " if ctx.level == 1 else ctx.rng.choice([" no ", " no ", " nos ", "-", " na "])
    return joiner.join(words) + (" no " if joiner != "-" else "-") if words else ""


def process(ctx: Ctx, invent: bool) -> Cite | None:
    rng, src = ctx.rng, ctx.src
    courts = MATTERS[ctx.matter]["courts"]
    court = rng.choice(sorted(courts))
    ids = [d for d in src.unique_short + src.unique_cnj
           if src.index.records[d].tribunal == court and src.index.records[d].classe in courts[court]]
    if not ids:
        return None
    doc_id = rng.choice(ids)
    record = src.index.records[doc_id]
    key = src.key_of(doc_id)
    if invent and (key := perturb_number(rng, src, key)) is None:
        return None
    number = _cnj(key, rng, ctx.level, ctx.holdout) if isinstance(key, tuple) else _short(key, rng, ctx.level,
                                                                                          ctx.holdout)
    if court == "TST":
        if any(c not in TST_CHAIN for c in record.chain):
            return None
        chain = "-".join([TST_CHAIN[c] for c in reversed(record.chain)] + [record.classe or "RR"])
        forms = [f"TST-{chain}-{number}", f"{chain}-{number}", f"processo nº TST-{chain}-{number}"]
        if ctx.level == 2:
            forms += [f"Processo n° TST- {chain}-{number}", f"TST-{chain}-{number}".replace("-", " - ", 1)]
        if ctx.holdout and rng.random() < 0.4:
            forms = [f"{chain} nº {number}", f"Recurso de Revista nº {number}" if chain == "RR" else f"{chain} {number}",
                     f"TST, {chain} {number}"]
        text = rng.choice(forms)
    else:
        prefix = _chain(ctx, record.chain)
        if prefix is None:
            return None
        cls = ctx.pick(*CLASS_FORMS[record.classe], CLASS_FORMS_N2.get(record.classe))
        marker = rng.choice(MARKERS_N1 if ctx.level == 1 else MARKERS_N2)
        uf = ""
        if record.uf and rng.random() < 0.85:
            uf = ctx.pick(UF_N1, UF_HOLDOUT, UF_N2).format(uf=record.uf)
        text = f"{prefix}{cls} {marker}{number}{uf}"
        if ctx.holdout and rng.random() < 0.25:
            text += rng.choice(COURT_SUFFIX_HOLDOUT).format(t=court)
    gender = "a" if record.classe in FEMININE and not text.startswith(("Ag", "AG", "ED", "Emb", "EDs")) else "o"
    return Cite(text, "jurisprudencia", "inventada" if invent else "real", "" if invent else doc_id), gender


def sumula(ctx: Ctx, invent: bool) -> Cite | None:
    rng, src = ctx.rng, ctx.src
    courts = MATTERS[ctx.matter]["sumulas"]
    if not courts:
        return None
    court = rng.choice(courts)
    real = [s for s in SUMULA_REAL if s[0] == court]
    if real and not invent:
        trib, num, vinc = rng.choice(real)
        doc_id = src.index.sumulas[(trib, num, vinc)]
    else:
        trib, vinc = court, court == "STF" and rng.random() < 0.4
        existing = {n for (t, n, v) in src.index.sumulas if t == trib and v == vinc}
        num = rng.choice([n for n in range(1, 700) if n not in existing])
        doc_id = ""
    if vinc:
        text = ctx.pick(SV_ITER, SV_HOLDOUT).format(n=num)
    else:
        form = ctx.pick(SUMULA_ITER, SUMULA_HOLDOUT)
        if trib == "TST" and num == 331 and ctx.holdout and rng.random() < 0.4:
            form = "Súmula {n}, IV, do {t}"
        text = form.format(n=num, t=trib, longo=COURT_LONG[trib])
    return Cite(text, "jurisprudencia", "real" if doc_id else "inventada", doc_id), "a"


def article(ctx: Ctx, invent: bool) -> Cite | None:
    rng, src = ctx.rng, ctx.src
    cfg = MATTERS[ctx.matter]
    if invent and rng.random() < 0.35:  # lei fora da cobertura: identificável, sem registro
        lei, apelido = rng.choice(cfg["leis_fora"])
        name = apelido if (apelido and ctx.holdout and rng.random() < 0.5) else lei
        num = rng.randint(2, 180)
        word = rng.choice(["art.", "artigo"] + (["art"] if ctx.level == 2 else []))
        return Cite(f"{word} {num}{'º' if num < 10 else ''} da {name}", "lei", "inventada"), "o"
    diploma = rng.choice(cfg["diplomas"])
    real = sorted((d, n) for d, n in src.index.dispositivos if d == diploma)
    if real and not invent:
        _, num = rng.choice(real)
        doc_id = src.index.dispositivos[(diploma, num)]
    else:
        existing = {n for _, n in real}
        num = rng.choice([n for n in range(2, 400) if n not in existing])
        doc_id = ""
    name = ctx.pick(DIPLOMA_ITER[diploma], DIPLOMA_HOLDOUT[diploma])
    art = f"{num}º" if num < 10 else fmt_thousands(num)
    if ctx.level == 2 and num < 10 and rng.random() < 0.3:
        art = f"{num}°"
    inciso = ""
    if INCISOS[diploma] and rng.random() < 0.5:
        inciso = ", " + rng.choice(INCISOS[diploma])
        if ctx.holdout and rng.random() < 0.3:
            inciso = inciso.replace(", ", ", inciso ", 1) if not inciso.startswith(", §") else inciso
    if ctx.holdout and not inciso and rng.random() < 0.2:
        inciso = ", caput,"
    prep = "da" if name.startswith(FEM_DIPLOMA) else "do"
    word = rng.choice(["art.", "artigo"] + (["art", "Art."] if ctx.level == 2 else []))
    sep = "," if inciso and not inciso.endswith(",") else ""
    text = f"{word} {art}{inciso}{sep} {prep} {name}"
    if ctx.holdout and rng.random() < 0.15:
        text = f"{name}, art. {art}{inciso.rstrip(',')}"
    return Cite(text, "lei", "real" if doc_id else "inventada", doc_id), "o"


def descriptive(ctx: Ctx, invent: bool) -> Cite | None:
    rng, src = ctx.rng, ctx.src
    courts = set(MATTERS[ctx.matter]["courts"])
    groups = [(g, recs) for g, recs in src.desc_groups if g[0] in courts]
    if not groups:
        return None
    (tribunal, ano, relator), _ = rng.choice(groups)
    full = clean_relator(relator)
    parts = full.split()
    name = rng.choice([full, full.upper(), full.title()])
    if len(parts) > 2 and rng.random() < 0.4:  # nome curto, como se cita: primeiro + último
        name = f"{parts[0]} {parts[-1]}".title()
    form = ctx.pick(DESC_ITER, DESC_HOLDOUT)
    classe = None
    if "{classe}" in form or "{sigla}" in form:
        records = [r for r in src.by_court_year[(tribunal, ano)] if r.classe in MATTERS[ctx.matter]["courts"][tribunal]]
        if not records:
            return None
        classe = rng.choice(records).classe
        long_name = CLASS_FORMS[classe][0][-1] if classe in CLASS_FORMS else classe
    text = form.format(t=tribunal, a=ano, r=name, longo=COURT_LONG[tribunal],
                       classe=long_name if classe else "", sigla=classe or "")
    matches = src.same_relator(tribunal, ano, name, classe)
    if not matches:
        return None
    if len(matches) == 1:
        return Cite(text, "jurisprudencia", "real", matches[0].doc_id), "o"
    return Cite(text, "jurisprudencia", "incompleta"), "o"


def tema(ctx: Ctx, invent: bool) -> Cite | None:
    n = ctx.rng.randint(100, 1300)
    num = fmt_thousands(n) if ctx.rng.random() < 0.6 else str(n)
    return Cite(ctx.pick(TEMA_ITER, TEMA_HOLDOUT).format(n=num), "jurisprudencia", "inventada"), "o"


KINDS = [(process, 0.44, FRAME_PROCESS), (article, 0.2, FRAME_ARTICLE), (sumula, 0.12, FRAME_SUMULA),
         (descriptive, 0.2, FRAME_DESC), (tema, 0.04, FRAME_TEMA)]


# ------------------------------------------------------------------ ruído (nível 2)

OCR = {"0": "O", "1": "l", "5": "S", "m": "rn", "e": "c", "a": "ã", "i": "l", "6": "G", "9": "g", "8": "B",
       "o": "0", "S": "5", "u": "n", "n": "ri", "c": "e"}
# Pares que o PDF cita (0↔O, 1↔l, 5↔S, m↔rn) pesam mais dentro das citações.
CITE_OCR = {"0": "O", "1": "l", "5": "S", "m": "rn", "6": "G", "9": "g", "8": "B", "e": "c", "a": "ã"}


def ocr(rng: random.Random, text: str, rate: float, table: dict) -> str:
    out = []
    for i, ch in enumerate(text):
        # Nunca troca dígito por dígito; primeiro e último caractere ficam (bordas do span).
        if ch in table and 0 < i < len(text) - 1 and rng.random() < rate:
            out.append(table[ch])
        else:
            out.append(ch)
    return "".join(out)


def cite_noise(rng: random.Random, text: str, strength: float) -> str:
    """Ruído do nível 2 dentro da citação (PDF, §2)."""
    text = ocr(rng, text, 0.05 * strength, CITE_OCR)
    r = rng.random
    if r() < 0.25 * strength:  # quebra de linha no meio do identificador
        spots = [m.start() for m in re.finditer(r"(?<=\S) (?=\S)|(?<=\d\.)(?=\d)|(?<=\d-)(?=\d)", text)]
        if spots:
            pos = rng.choice(spots)
            if text[pos] == " ":  # entre palavras: só a quebra ("-\n" aqui não é hifenização real)
                text = text[:pos] + rng.choice(["\n", "\n "]) + text[pos + 1:]
            else:  # dentro do número, como "33.-\n474" e "2020-\n.6.14" da amostra oficial
                text = text[:pos] + rng.choice(["\n", "-\n"]) + text[pos:]
    if r() < 0.2 * strength:  # espaço duplo
        text = text.replace(" ", "  ", 1)
    if r() < 0.1 * strength:  # ponto duplicado ou perdido
        text = re.sub(r"\.(?=\d)", lambda _: rng.choice([". ", "", ".-"]), text, count=1)
    return text


def body_noise(rng: random.Random, text: str, strength: float) -> str:
    text = ocr(rng, text, 0.006 * strength, OCR)
    return re.sub(" ", lambda _: "  " if rng.random() < 0.01 * strength else " ", text)


def wrap(text: str, rng: random.Random) -> str:
    """Quebra dura a cada ~95–110 caracteres, trocando espaço por \\n (não muda offsets)."""
    out, col = list(text), 0
    limit = rng.randint(92, 110)
    for i, ch in enumerate(out):
        col = 0 if ch == "\n" else col + 1
        if ch == " " and col >= limit:
            out[i], col, limit = "\n", 0, rng.randint(92, 110)
    return "".join(out)


# ------------------------------------------------------------------ documento

def _date(rng: random.Random) -> str:
    return f"{rng.randint(1, 28)} de {rng.choice(MONTHS)} de {rng.randint(2016, 2026)}"


def _autos(rng: random.Random) -> str:
    return (f"{rng.randint(1000000, 9999999)}-{rng.randint(10, 99)}.{rng.randint(2012, 2025)}."
            f"{rng.randint(1, 9)}.{rng.randint(1, 27):02d}.{rng.randint(1, 9999):04d}")


def header(rng: random.Random, matter: str) -> str:
    a, b = rng.sample(PEOPLE, 2)
    lines = [rng.choice(HEADS[matter]), "", f"Autos nº {_autos(rng)}",
             f"{rng.choice(['Recorrente', 'Agravante', 'Impetrante', 'Requerente'])}: {a}",
             f"{rng.choice(['Recorrido', 'Agravado', 'Impetrado', 'Requerido'])}: {b}"]
    if rng.random() < 0.6:
        lines.append(f"Protocolo nº {rng.randint(2016, 2026)}.{rng.randint(1000000, 9999999)}")
    if rng.random() < 0.3:
        lines.append(f"CPF {rng.randint(100, 999)}.{rng.randint(100, 999)}.{rng.randint(100, 999)}-"
                     f"{rng.randint(10, 99)}")
    lines += ["", rng.choice(TITLES), ""]
    intro = (f"{a}, por seu advogado (OAB/{rng.choice(['SP', 'RJ', 'MG', 'DF', 'RS'])} {rng.randint(10000, 499999)}), "
             f"vem, respeitosamente, apresentar sua manifestação, pelos fundamentos a seguir expostos.")
    return "\n".join(lines) + "\n" + intro + "\n"


def closing(rng: random.Random) -> str:
    extras = [f"Valor da causa: R$ {rng.randint(1, 900)}.{rng.randint(100, 999)},{rng.randint(10, 99)}.",
              f"Documentos às fls. {rng.randint(10, 400)}/{rng.randint(401, 900)}."]
    return (f"\n{rng.choice(extras)}\n\nAnte o exposto, requer-se o acolhimento das razões apresentadas.\n\n"
            f"{rng.choice(CITIES)}, {_date(rng)}.\n")


def build(rng: random.Random, src: Sources, level: int, holdout: bool, strength: float, cites: tuple[int, int]):
    matter = rng.choice(sorted(MATTERS))
    ctx = Ctx(rng, src, matter, level, holdout)
    text = header(rng, matter)
    labels = []
    target = rng.randint(*cites)
    sections = rng.choice(SECTIONS)
    per_section = max(1, target // (len(sections) - 1))
    for s, title in enumerate(sections):
        text += f"\n{title}\n\n"
        if s == 0:
            text += wrap(rng.choice(FACTS[matter]) + " " + " ".join(rng.sample(FILLERS, 2)), rng) + "\n"
            continue
        n_here = per_section if s < len(sections) - 1 else max(0, target - len(labels))
        paragraph = []  # (texto, rótulo | None)
        for _ in range(n_here):
            funcs, weights, frames = zip(*KINDS)
            for _ in range(20):
                k = rng.choices(range(len(KINDS)), weights)[0]
                out = funcs[k](ctx, rng.random() < 0.45)
                if out:
                    break
            else:
                continue
            cite, gender = out
            frame = rng.choice(frames[k])
            before, after = frame.split("{c}")
            before = before.replace("{o}", gender)
            if rng.random() < 0.5:
                paragraph.append((rng.choice(FILLERS) + " ", None))
            if rng.random() < 0.15:
                paragraph.append((rng.choice(VAGUE) + " ", None))
            surface = cite_noise(rng, cite.text, strength) if level == 2 else cite.text
            paragraph += [(before, None), (surface, cite), (after + " ", None)]
        if not paragraph:
            continue
        chunk_start = len(text)
        pieces, spans, pos = [], [], 0
        for piece, cite in paragraph:
            if cite is None and level == 2:
                piece = body_noise(rng, piece, strength)
            if cite is not None:
                spans.append((pos, pos + len(piece), cite))
            pieces.append(piece)
            pos += len(piece)
        body = wrap("".join(pieces).rstrip(), rng)  # troca espaço por \n: offsets intactos
        text += body + "\n"
        for a, b, cite in spans:
            labels.append((chunk_start + a, chunk_start + b, cite))
    text += closing(rng)
    return text, labels


def generate(split: str, dest: Path, seed: int, src: Sources, docs: int, holdout: bool, strength: float,
             cites: tuple[int, int]) -> None:
    rng = random.Random(f"v5-{split}-{seed}")
    txt_dir = dest / "txt"
    txt_dir.mkdir(parents=True, exist_ok=True)
    for old in txt_dir.glob("*.txt"):
        old.unlink()
    rows = []
    for i in range(docs):
        level = 1 if i < docs // 2 else 2
        doc_id = f"{split}_n{level}_{i + 1:03d}"
        text, labels = build(rng, src, level, holdout, strength, cites)
        (txt_dir / f"{doc_id}.txt").write_text(text, encoding="utf-8", newline="")
        for j, (start, end, cite) in enumerate(labels, 1):
            rows.append({"nivel": level, "documento_id": doc_id, "citacao_id": f"g{j}", "inicio": start, "fim": end,
                         "trecho": text[start:end].replace("\n", "\\n"), "tipo": cite.tipo,
                         "classificacao": cite.classe, "id_canonico": cite.doc_id})
    with (dest / "goldenset.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{split}: {docs} documentos, {len(rows)} citações {dict(Counter(r['classificacao'] for r in rows))}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--docs", type=int, default=60)
    parser.add_argument("--holdout", action="store_true", help="inclui as formas exclusivas de holdout")
    parser.add_argument("--noise", type=float, default=1.0, help="intensidade do ruído do nível 2")
    parser.add_argument("--cites", type=int, nargs=2, default=(7, 11))
    parser.add_argument("--db", type=Path, default=Path("data/desafio1_bracis.db"))
    parser.add_argument("--out", type=Path, default=Path("data/synthetic"))
    args = parser.parse_args()
    generate(args.split, args.out / args.split, args.seed, Sources(args.db), args.docs, args.holdout, args.noise,
             tuple(args.cites))


if __name__ == "__main__":
    main()
