"""Identificação de spans de citação, em qualquer peça.

Três camadas, todas sobre o texto limpo e com os spans devolvidos em
codepoints do texto ORIGINAL:

  1. limpeza: remove invisíveis, junta hifenização de fim de linha e guarda o
     mapa de offsets (`clean`, `to_original`);
  2. padrões catalogados: regex das formas conhecidas, tolerantes a OCR
     (`extract_citations`);
  3. âncoras gerais: partem do que toda citação tem (número + frase jurídica,
     súmula + número, artigo + diploma, tribunal + ano + relator) e acham
     formas novas (`expand_spans`, `find_anchors`).

Aqui só se decide ONDE está cada citação e de que família ela é; a classe
(real/inventada/incompleta) é decidida em classify.py, contra a base.
"""

from __future__ import annotations

import re

from .normalize import (
    OCR_DIGITS,
    canonical_words,
    cnj_justice,
    cnj_key,
    diploma_key,
    expand_abbreviation,
    fold,
    LEGAL_VOCABULARY,
    LEXICON,
    is_number_marker,
    name_tokens,
)


# ============================================================================
# 1. Limpeza com mapa de offsets
# ============================================================================
# Limpeza de digitalização com mapa de offsets.
#
# O regex roda no texto limpo; cada posição limpa sabe de qual posição do
# texto ORIGINAL veio, então os spans entregues continuam em codepoints do
# original (exigência do contrato).
#
# Operações (todas gerais, nenhuma específica de citação):
#   * remove caracteres invisíveis (zero-width, soft hyphen, BOM);
#   * junta hifenização de fim de linha entre letras ("Recur-\nso" -> "Recurso");
#   * troca espaços estranhos (NBSP, thin space, tab) por espaço comum;
#   * colapsa pontuação duplicada (".." -> ".", ",," -> ",").

INVISIBLE = {"\u200b", "\u200c", "\u200d", "\u00ad", "\ufeff", "\u2060"}  # zero-width, soft hyphen, BOM, word joiner
ODD_SPACE = {"\u00a0", "\u2009", "\u202f", "\u2007", "\t", "\x0c"}  # NBSP, thin, narrow NBSP, figure space
# "-\n" (com espaços opcionais) entre duas letras: quebra de linha hifenizada.
HYPHEN_BREAK = re.compile(r"(?<=[A-Za-zÀ-ÿ])-[ \t]*(?:\r?\n[ \t]*){1,2}(?=[A-Za-zÀ-ÿ])")


def clean(text: str, words: frozenset[str] = frozenset()) -> tuple[str, list[int]]:
    """(texto limpo, mapa) onde mapa[i] = posição no original do caractere i.

    `words` (tokens de nomes da base, já com fold) reforça o vocabulário usado
    para juntar palavras partidas ("Anto nio") e separar nomes colados.
    """
    # 0. Mojibake (UTF-8 lido como cp1252/latin-1) volta a ser um caractere, que
    #    aponta para o início da sequência; a numeração de linha da margem sai.
    chars: list[tuple[str, int]] = []
    pos = 0
    for m in MOJIBAKE.finditer(text):
        chars += [(c, i) for i, c in enumerate(text[pos:m.start()], pos)]
        chars.append((MOJIBAKE_MAP[m.group()], m.start()))
        pos = m.end()
    chars += [(c, i) for i, c in enumerate(text[pos:], pos)]
    margin = _margin_numbers(text)
    # 1. Invisíveis saem primeiro: "Recu-\n\u200brso" só vira hifenização
    #    reconhecível depois disso.
    kept = [i for c, i in chars if c not in INVISIBLE and i not in margin]
    by_pos = dict((i, c) for c, i in chars)
    visible = "".join(by_pos[i] for i in kept)
    # 2. Hifenização de fim de linha: posições (no texto visível) a pular.
    skip = set()
    for m in HYPHEN_BREAK.finditer(visible):
        skip.update(range(m.start(), m.end()))

    out: list[str] = []
    mapping: list[int] = []
    prev = ""
    for j, ch in enumerate(visible):
        i = kept[j]
        if j in skip:
            continue
        if ch in ODD_SPACE:
            ch = " "
        if ch in ".," and prev == ch:
            continue
        out.append(ch)
        mapping.append(i)
        prev = ch
    mapping.append(len(text))  # posição de fim
    cleaned, mapping = _join_split_words("".join(out), mapping, words)
    for _ in range(3):  # um corte pode expor outro: "oREspnº" -> "o REspnº" -> "o REsp nº"
        size = len(cleaned)
        cleaned, mapping = _split_glued(cleaned, mapping, words)
        if len(cleaned) == size:
            break
    return cleaned, mapping


# UTF-8 decodificado como cp1252 ou latin-1: "ção" -> "Ã§Ã£o", "nº" -> "nÂº", "—" -> "â€”".
# As sequências começam por Â/Ã/Ä/Å/â seguidas de símbolos que não vêm depois dessas
# letras em português. Ficam de fora as que continuam com espaço, travessão ou aspas:
# "DÃ\xa0RELATORIA" é "DA" com OCR, não "à"; "IRMÃ”" é palavra entre aspas.
_MOJIBAKE_AMBIGUOUS = set("\xa0 –—‘’“”")


def _mojibake_map() -> dict[str, str]:
    table = {}
    for ch in [chr(c) for c in range(0xA0, 0x180)] + list("–—‘’“”•…€"):
        raw = ch.encode("utf-8")
        for codec in ("cp1252", "latin-1"):
            try:
                seq = raw.decode(codec)
            except UnicodeDecodeError:
                continue
            if len(seq) > 1 and not _MOJIBAKE_AMBIGUOUS & set(seq[1:]):
                table.setdefault(seq, ch)
    return table


MOJIBAKE_MAP = _mojibake_map()
MOJIBAKE = re.compile("|".join(re.escape(k) for k in sorted(MOJIBAKE_MAP, key=len, reverse=True)))

# Numeração de linha na margem ("  12  texto"): só quando o documento inteiro
# a usa (5+ linhas, números crescentes); "1.  Dos fatos" não casa (tem ponto).
_MARGIN_NUMBER = re.compile(r"(?m)^[ \t]*(\d{1,3})(?:[ \t]{2,}|\t)")


def _margin_numbers(text: str) -> set[int]:
    found = list(_MARGIN_NUMBER.finditer(text))
    if len(found) < 5:
        return set()
    numbers = [int(m.group(1)) for m in found]
    rising = sum(b > a for a, b in zip(numbers, numbers[1:]))
    if rising < 0.8 * (len(numbers) - 1):
        return set()
    return {i for m in found for i in range(m.start(1), m.end(1))}


# Marcador de número colado aos dígitos: "n233", "no233", "nº233", "numero233". Exige 2+
# dígitos: "n0 REsp" é o "no" com OCR, não o processo nº 0.
GLUED_MARKER = re.compile(r"(?<![A-Za-zÀ-ÿ])(?:n[º°o.]?|n\.º|nr\.?|nro\.?|n[uú]m\.?|n[uú]mero)(?=\d\d)", re.IGNORECASE)

# Espaço perdido na digitalização ("noARESP", "doart.", "REsp1.664", "443do").
# Cada padrão casa até o ponto onde falta o espaço; só fronteiras inequívocas:
# conector minúsculo antes de maiúscula/dígito/palavra-chave, sigla ou classe
# antes de número, número antes de conector, tribunal antes de minúscula.
_CONN1 = r"(?:o|a|e|à)"
_CONN2 = r"(?:aos|ao|os|as|nos|nas|no|na|dos|das|do|da|de|em|pelos|pelas|pelo|pela|sob)"
# Depois de conector de uma letra, só uma sigla/classe inteira conhecida ("aRcl", "oRHC"):
# "aGrG", "eSPECIAL" são palavras com a caixa trocada, não "a GrG".
_UPPER_FORMS = "|".join(sorted((
    "AR", "AI", "RE", "RO", "RR", "HC", "MS", "ED", "EDcl", "AgR", "AgRg", "AgInt", "REsp", "AREsp", "REspe",
    "RHC", "RMS", "Rcl", "Recl", "APL", "RSE", "STF", "STJ", "TST", "TSE", "STM", "CPC", "CPP", "CF", "CLT",
    "CDC", "Recurso", "Agravo", "Súmula", "Reclamação", "Lei", "Constituição", "Código", "Tema", "Embargos",
    "Habeas", "Apelação", "Suspensão", "Mandado", "Processo"), key=len, reverse=True))
_LEFT_KEYWORDS = (r"(?:art|artigos?|s[uú]mulas?|precedentes?|julgados?|ac[oó]rd[aã]os?|recursos?|agravos?"
                  r"|reclama[cç][aã]o|lei|processo|embargos|habeas|apela[cç][aã]o|tema)")
_RIGHT_KEYWORDS = (r"(?:a?resp|a?respe|agresp|are|re|rcl|recl|rhc|rms|apl|hc|ar|ai|ro|rr|arr|airr|rse|agint|agrg"
                   r"|agr|edcl|ed|eds|especial|eleitoral|extraordin[aá]rio|ordin[aá]rio|vinculante|s[uú]mulas?"
                   r"|artigo|art|lei|tema|reclama[cç][aã]o|agravo|interno|regimental|instrumento|corpus"
                   r"|seguran[cç]a|criminal|apela[cç][aã]o)")
_NOT_LETTER = r"(?![A-Za-zÀ-ÿ])"
_START = r"(?<![A-Za-zÀ-ÿ\d])"
GLUE_POINTS = [
    GLUED_MARKER,
    re.compile(rf"{_START}(?-i:{_CONN2})(?=[A-ZÀ-Ý][A-Za-zÀ-ÿ])"),                        # "noARESP", "doSTJ"
    re.compile(rf"{_START}(?-i:{_CONN1})(?=(?:{_UPPER_FORMS})(?:{_NOT_LETTER}|n[º°.]))", re.IGNORECASE),  # "aRcl"
    re.compile(rf"{_START}(?-i:{_CONN2})(?=\d)"),                                        # "de2023", "na5úmula"
    re.compile(rf"{_START}(?-i:{_CONN1}|{_CONN2})(?={_LEFT_KEYWORDS}{_NOT_LETTER})", re.IGNORECASE),  # "doart."
    # Sigla/classe antes de número com 2+ dígitos: "REsp1.664", "Vinculante10"; "Re1." é "Rel.".
    re.compile(rf"(?<![A-Za-zÀ-ÿ]){_RIGHT_KEYWORDS}"
               rf"(?=\d[\d.,]*\d(?![A-Za-zÀ-ÿ\d])|n[º°.]\s*\d)", re.IGNORECASE),
    # Número (não colado a letra) antes de conector: "443do", "2016sob"; "Reg1na" é nome.
    re.compile(rf"{_START}\d[\d.,/\-–]*\d(?=(?-i:dos|das|do|da|de|sob|em|nos|nas|no|na|pelo|pela|d0){_NOT_LETTER})"),
    re.compile(r"(?<![A-Za-zÀ-ÿ])(?-i:[S5]T[FJM]|T[S5][TE])(?=[a-zà-ÿ]{2,})"),           # "STJproferido"
]
# Palavras que, partidas por um espaço ("Suspe nsão", "julg ado"), são reunidas.
_JOIN_EXTRA = {"precedente", "acordao", "decisao", "liminar", "sentenca", "suspensao", "reclamacao"}


def _join_split_words(text: str, mapping: list[int], words: frozenset[str]) -> tuple[str, list[int]]:
    """Remove o espaço de "Suspe nsão" quando a junção é palavra jurídica ou nome da base
    e nenhuma das metades é palavra sozinha."""
    vocab = words | _JOIN_EXTRA | set(LEGAL_VOCABULARY)
    tokens = list(re.finditer(r"[A-Za-zÀ-ÿ]+", text))
    drop = set()
    for a, b in zip(tokens, tokens[1:]):
        if text[a.end():b.start()] != " " or a.start() in drop:
            continue
        fa, fb = fold(a.group()), fold(b.group())
        if len(fa) < 2 or len(fb) < 2 or fa in vocab or fb in vocab or fa in CONNECTORS or fb in CONNECTORS:
            continue
        if fa + fb in vocab:
            drop.add(a.end())
    if not drop:
        return text, mapping
    keep = [i for i in range(len(text)) if i not in drop]
    return "".join(text[i] for i in keep), [mapping[i] for i in keep] + [mapping[-1]]


def _glued_names(text: str, words: frozenset[str]) -> list[int]:
    """Pontos de corte em nomes colados da base: "CarlosFerreira", "AUGUSTOAMARAL"."""
    points = []
    if not words:
        return points
    for m in re.finditer(r"[A-Za-zÀ-ÿ]{8,}", text):
        w = fold(m.group())
        if w in words or w in LEGAL_VOCABULARY:
            continue
        for i in range(4, len(w) - 3):
            if w[:i] in words and w[i:] in words:
                points.append(m.start() + i)
                break
    return points


def _split_glued(text: str, mapping: list[int], words: frozenset[str] = frozenset()) -> tuple[str, list[int]]:
    """Insere um espaço em cada ponto de colagem; o espaço aponta para o caractere seguinte no original."""
    points = {m.end() for rx in GLUE_POINTS for m in rx.finditer(text) if 0 < m.end() < len(text)}
    points.update(_glued_names(text, words))
    out, new_map, pos = [], [], 0
    for p in sorted(points):
        if text[p - 1].isspace() or text[p].isspace():
            continue
        out.append(text[pos:p] + " ")
        new_map += mapping[pos:p] + [mapping[p]]
        pos = p
    out.append(text[pos:])
    new_map += mapping[pos:]
    return "".join(out), new_map


def to_original(mapping: list[int], start: int, end: int) -> tuple[int, int]:
    """Span no texto limpo -> span no original (fim exclusivo)."""
    return mapping[start], mapping[end - 1] + 1


# ============================================================================
# 2. Padrões catalogados (regex tolerante a OCR)
# ============================================================================
# Padrões de superfície: identificam citações, não sua validade jurídica.
#
# Os componentes são compartilhados entre famílias. Não normalizamos o texto
# inteiro: \s reconhece espaços e quebras reais sem alterar os spans.

FLAGS = re.IGNORECASE | re.VERBOSE

# Trocas típicas de OCR em palavras-chave ("5úmula", "Fedcral", "profcrido",
# "Súmulo", "Súmnla", "Milifar", "Códiqo"). Só é aplicado a literais fixos,
# nunca a nomes de relatores.
CONFUSABLE = {
    "s": "s5", "e": "eéêc", "a": "aáàâão", "i": "iíl1", "o": "oóôõ0a",
    "u": "uún", "c": "cçe", "l": "l1i", "ç": "çc", "ã": "ãaâ", "é": "éec",
    "í": "íil", "ó": "óo", "ú": "úu", "ê": "êe", "â": "âa", "õ": "õo",
    "t": "tf", "f": "ft", "g": "gq",
}
# Uma letra lida como duas (ou o contrário): "rn" por "m", "ri" por "n", "cl" por "d", "li" por "h".
SPLIT_CONFUSABLE = {"m": "(?:m|rn)", "n": "(?:[nu]|ri)", "d": "(?:d|cl)", "h": "(?:h|b|li)"}


def _ocr_char(ch: str) -> str:
    if ch.lower() in SPLIT_CONFUSABLE:
        return SPLIT_CONFUSABLE[ch.lower()]
    if ch.lower() in CONFUSABLE:
        return f"[{CONFUSABLE[ch.lower()]}]"
    return re.escape(ch)


def ocr(phrase: str) -> str:
    """Regex tolerante a OCR para uma expressão literal (espaços viram \\s+)."""
    return r"\s+".join("".join(_ocr_char(ch) for ch in word) for word in phrase.split())

# "nº" com OCR: "n°", "no", "n0"; e, só com o sinal ou o ponto, "riº" e "uº"/"u.".
NUMBER_MARKER = r"(?:(?:n(?:\.\s*[º°o0]|[º°o0.])?|(?:u|ri)(?:\.\s*[º°o0]|[º°.]))\s*)?"
UF = r"(?:AC|AL|AP|AM|BA|CE|DF|ES|GO|MA|MT|MS|MG|PA|PB|PR|PE|PI|RJ|RN|RS|RO|RR|SC|SP|SE|TO)"
# Siglas de tribunal com S lido como 5 ("T5T", "5TF").
TRIBUNAL = r"(?:[S5]T[FJM]|T[S5][TE])"
# Mantém os tamanhos dos blocos CNJ, aceitando separadores ausentes,
# repetidos ou substituídos por espaços. Não altera nenhum dígito.
CNJ_SEPARATOR = r"[\s.\-–/]*"
# Letras que o OCR troca por dígitos: as mesmas que normalize.OCR_DIGITS converte
# (os padrões rodam com IGNORECASE, então "S" também cobre "s", "B" cobre "b" etc.).
# "|" só vale sozinho: "|||" é fio de tabela ou de página, não "111".
_D = r"(?:[\dOolISgGBLDQZ]|(?<!\|)\|(?!\|))"
CNJ_NUMBER = CNJ_SEPARATOR.join((
    rf"(?<![A-Za-zÀ-ÿ])(?:[OolIGSBDQZ|]{{0,3}}\d|[lIOGSBDQZ|](?=\d)){_D}{{0,6}}", rf"{_D}{{2}}", rf"{_D}{{4}}", _D, rf"{_D}{{2}}", rf"{_D}{{4}}"
))
# Número de lei, com OCR a partir do 2º caractere ("g.504" também: 1º seguido de ".").
# Número de lei: "9.504", "l3.467", "13.-467" (hífen de quebra de linha depois do ponto).
SHORT_NUMBER = rf"(?:\d|[lIOSgQZ|](?=[\d.]))[\dOolISgGBQZ|]*(?:\s*\.[\s\-]*[\dOolISgGBQZ|]+)*"
# Espaços entre milhares são aceitos só em grupos de três dígitos.
# O formato de leis permanece separado para não ampliar essa família.
# Dígitos podem vir trocados por letras parecidas ("21737l8", "1.528.4S5");
# o primeiro caractere precisa ser um dígito real.
OCR_DIGIT = _D
# O número nunca começa colado numa letra: "n0" (OCR de "no") não é o nº 0.
PROCESS_SHORT_NUMBER = (
    rf"(?<![A-Za-zÀ-ÿ])(?:(?:\d|[lILBGSOQZ|](?=\d)|[lIL|](?=[.\s]\s*{OCR_DIGIT})){OCR_DIGIT}{{0,2}}"
    rf"(?:\s*[.\s][\s\-–]*{OCR_DIGIT}{{3}}(?:{OCR_DIGIT}{{3}})?|\s*[\-–]\s*\.\s*{OCR_DIGIT}{{3}}|,{OCR_DIGIT}{{3}}(?!\d))+"
    rf"|(?:\d|[lILBGSOQZ|](?=\d)){OCR_DIGIT}*)"
)
PROCESS_NUMBER = rf"(?:{CNJ_NUMBER}|{PROCESS_SHORT_NUMBER})"
# Não aceita um prefixo numérico se outro bloco de dígitos vem após ponto
# ou hífen. Ex.: um OCR não suportado em 1.234.56g não deve virar só 1.234.
# Também recusa parar antes de ",ddd": "87,101" não pode virar o processo 87.
# Ano com OCR nos dígitos ("201g", "20l7").
YEAR = r"(?:19|[2Z][0OD])[\dOolIgSBDQZ|]{2}(?!\w)"
# "-5P" é a UF SP com OCR, não continuação do número.
# O número continua depois de ponto/hífen com no máximo uma quebra de linha de cada lado ("33.-\n474");
# linha em branco encerra o número ("067.\n\n3. DOS PEDIDOS" é outro parágrafo).
_WS1 = r"[^\S\n]*(?:\n[^\S\n]*)?"
NUMBER_END = rf"(?!\w)(?!{_WS1}[.\-–]{_WS1}(?!(?-i:5[A-Z])\b)\d)(?!,\d)"
STATE_SUFFIX = rf"(?:\s*(?:[/–-]\s*{UF}\b|\({UF}\)))?"

def ocr_alternatives(phrases: list[str]) -> str:
    """Alternativas tolerantes a OCR, das mais longas para as mais curtas."""
    return "|".join(ocr(p) for p in sorted(phrases, key=len, reverse=True))


# Nomes por extenso recebem tolerância a OCR; siglas curtas ficam literais
# para não gerar falsos positivos.
CLASS_NAMES = [
    "Reclamação", "Apelação Criminal", "Apelação", "Habeas Corpus", "Ação Rescisória",
    "Recurso Ordinário", "Recurso em Sentido Estrito", "Agravo Interno", "Agravo Regimental",
    "Agravo em Recurso Especial", "Recurso Especial Eleitoral", "Recurso Especial",
    "Recurso Extraordinário", "Recurso em Habeas Corpus", "Recurso em Mandado de Segurança",
    "Suspensão de Liminar e de Sentença", "Agravo de Instrumento",
]
PROCESS_CLASS = rf"""(?:
    {ocr_alternatives(CLASS_NAMES)}
    | Recl | Rcl
    # "Rcl" com OCR ("Rel", "Rc1", "RecI", "Rd"), só quando um número vem em seguida
    | R(?-i:e?[ce][l1I|]|d)(?=\.?\s+(?:(?:n|u|ri)[º°.]?\s*)?(?:\d|[lIOGSBQZ|]\d)) | Rec\.\s*Esp\.
    | AREspE[Il1] | AgREsp | A\.?REsp | REspe | R\.?Esp | RHC | RMS | ARE | RE | RO
    | H\.?C | AR | AI | APL | RSE | R-Rp | Ag\.?\s*Int
)"""
APPEAL_NAMES = ["Agravo Interno", "Agravo Regimental", "Embargos de Declaração", "Embargos de Divergência"]
APPEAL_PREFIX = rf"""(?:
    Ag\.?\s*Int\.? | AgRg | AgR | EDcl | EDs | EDv | ED | PExt | AG\.REG\.?
    | {ocr_alternatives(APPEAL_NAMES)}
)"""
# Recursos repetidos do STF: "Terceiro AG.REG na Rcl", "Segundos EDcl no AgR".
# Só entra antes de um prefixo recursal; sozinho, é prosa ("segundo o STJ").
APPEAL_ORDINAL = r"(?:Primeir|Segund|Terceir|Quart|Quint|Sext)[oa]s?\s+"

PROCESSOS = re.compile(
    rf"\b(?:(?:{APPEAL_ORDINAL})?(?:{APPEAL_PREFIX}(?:\s+n[oa0ã]s?\s+|\s*-\s*))+)?"
    rf"{PROCESS_CLASS}\.?\s+{NUMBER_MARKER}{PROCESS_NUMBER}{NUMBER_END}{STATE_SUFFIX}",
    FLAGS,
)

# A identificação trabalhista usa cadeias como TST-ED-E-ED-RR-<CNJ>.
PROCESSOS_TRABALHISTAS = re.compile(
    rf"\b(?:{ocr('processo')}\s+{NUMBER_MARKER})?(?:T[S5]T\s*-\s*)?"
    rf"(?:(?:ED|E|Ag|AgR|Emb)\s*-\s*)*(?:Ag)?(?:AIRR|ARR|RRAg|RR|ROT|IRR)\s*-\s*"
    rf"{CNJ_NUMBER}{NUMBER_END}{STATE_SUFFIX}",
    FLAGS,
)

# O 1º caractere pode ser letra de OCR se um dígito real vem logo depois ("SO6", "G7B").
# "D" nunca abre número: "d0" é o "do" com OCR ("Súmula d0 STJ"), como "n0" é o "no".
SUMULA_NUMBER = rf"{NUMBER_MARKER}(?:\d|[lIBgGSOQZ|](?={OCR_DIGIT}{{0,2}}\d)){OCR_DIGIT}*(?![\w|])"
SUMULAS = re.compile(
    rf"""(?<!\w)(?:
        # "Súmula 83 do STJ", "Súmula Vinculante 10", "Súmula nº 83/STJ"
        (?:{ocr('Súmula')}|Súm\.)\s+(?:{ocr('Vinculante')}\s+)?{SUMULA_NUMBER}
        (?:\s*(?:d[oa]\s+|/){TRIBUNAL}\b)?
      | # "Enunciado 83 da Súmula do STJ"
        {ocr('Enunciado')}\s+{SUMULA_NUMBER}\s+d[ao]\s+{ocr('Súmula')}
        (?:\s+{ocr('Vinculante')})?(?:\s+d[oa]\s+{TRIBUNAL}\b)?
    )""",
    FLAGS,
)

CONSTITUICAO = f"{ocr('Constituição')}\\s+(?:{ocr('da República')}|{ocr('Federal')})"
DIPLOMA_NAMES = [
    "Código de Processo Civil", "Código de Processo Penal", "Código de Defesa do Consumidor",
    "Código Penal Militar", "Código Civil", "Código Penal", "Código Eleitoral",
    "Consolidação das Leis do Trabalho",
]
DIPLOMA = rf"""(?:
    {ocr_alternatives(DIPLOMA_NAMES)}
    | {CONSTITUICAO}
    | N?CPC | CPP | CDC | CPM | CC | CP | CLT | CF(?:/[\dB]{2})? | CR(?:FB)?(?:/[\dB]{2})?
)"""
LAW_NUMBER = (
    rf"(?:{ocr('Lei')}\s+(?:{ocr('Complementar')}\s+)?|LC\s+)"
    rf"{NUMBER_MARKER}{SHORT_NUMBER}\s*/\s*{YEAR}"
)
ARTICLE_NUMBER = (r"(?:\d|[lISLBgQZ|](?=[\dº°OolIgGSBDQZ|]))[\dOolIgGSBDQZ|]*"
                  r"(?:\.[\dOolIgGSBDQZ|]+)*(?:[º°o])?(?:-[A-Z])?")
PARAGRAPH = rf"§\s*{ARTICLE_NUMBER}|parágrafo\s+único"
INCISO = r"(?:inciso\s+)?[IVXLCDM]+\b"
ALINEA = r"(?:alínea\s+)?['\"‘’][a-z]['\"‘’]"

ARTIGOS = re.compile(
    rf"\b(?:{ocr('artigo')}|{ocr('art')}\.?)\s+{ARTICLE_NUMBER}"
    rf"(?:\s*,\s*(?:{PARAGRAPH}|{INCISO}|{ALINEA}))*"
    rf"\s*,?\s*(?:d|cl)[oaã0]\s+(?:{LAW_NUMBER}|{DIPLOMA})(?!\w)",  # "cla CLT": OCR de "da"
    FLAGS,
)


# Ordem invertida: "CLT, art. 818", "Constituição Federal, artigo 5º, LV".
ARTIGOS_INVERTIDOS = re.compile(
    rf"\b(?:{DIPLOMA}),\s*(?:{ocr('artigo')}|{ocr('art')}\.?)\s+{ARTICLE_NUMBER}"
    rf"(?:\s*,\s*(?:{PARAGRAPH}|{INCISO}|{ALINEA}))*(?!\w)",
    FLAGS,
)

# Referências processuais descritivas, sem fixar nomes de ministros.
# (?-i:...) mantém a inicial maiúscula como sinal de continuação do nome.
NAME_WORD = r"(?-i:[A-ZÀ-ÖØ-Þ][a-zA-ZÀ-ÖØ-öø-ÿ0-9]*)"
RELATOR_NAME = rf"{NAME_WORD}(?:\s+(?:(?:de|da|do|dos|das)\s+)?{NAME_WORD})*"
# Classe + ano + relator: "Rcl de 2021, Rel. Min. Rosa Weber"; vale para qualquer classe, inclusive as do
# TST ("RR de 2016, Rel. Min. …"), e para "Rel."/"Min." com OCR ("Rcl. Mln.").
DESCRIPTIVE_CLASS = rf"(?:{PROCESS_CLASS}|A?I?RR|ARR)"
PROCESSOS_DESCRITIVOS = re.compile(
    rf"\b{DESCRIPTIVE_CLASS}\.?\s+(?:{ocr('do')}\s+{TRIBUNAL}\s*,?\s*)?"
    rf"{ocr('de')}\s+{YEAR}\s*,\s*{ocr('Rel')}\.\s*M[il1í]n\.\s*{RELATOR_NAME}",
    FLAGS,
)

# Referências sem número, identificadas por tribunal, ano e relator.
# As alternativas descrevem a frase; nomes e anos não são enumerados.
JULGADOS_DESCRITIVOS = re.compile(
    rf"\b(?:{ocr('julgado')}|{ocr('acórdão')}|{ocr('precedente')}|{ocr('decisão')})\s+{ocr('do')}\s+{TRIBUNAL}\s+"
    rf"(?:(?:{ocr('proferido')}|{ocr('julgado')})\s+{ocr('em')}|{ocr('de')})\s+{YEAR}"
    rf"\s*,?\s+(?:(?:{ocr('pela')}|{ocr('da')}|{ocr('sob')})\s+)?{ocr('relatoria')}\s+(?:{ocr('de')}|{ocr('do')})"
    rf"\s+(?:M[il1í]n(?:istr[oa])?[\s.\-–]*)?{RELATOR_NAME}",
    FLAGS,
)

# Temas de repercussão geral ou repetitivos. A base não indexa temas.
TEMAS = re.compile(
    rf"\b{ocr('Tema')}\s+{NUMBER_MARKER}{PROCESS_SHORT_NUMBER}{NUMBER_END}"
    rf"(?:\s+{ocr('da repercussão geral')}|\s+{ocr('dos recursos repetitivos')})?",
    FLAGS,
)

# Registro: nome da família, tipo do contrato, expressão compilada.
PATTERNS = (
    ("processos", "jurisprudencia", PROCESSOS),
    ("processos_trabalhistas", "jurisprudencia", PROCESSOS_TRABALHISTAS),
    ("sumulas", "jurisprudencia", SUMULAS),
    ("artigos", "lei", ARTIGOS),
    ("artigos", "lei", ARTIGOS_INVERTIDOS),
    ("processos_descritivos", "jurisprudencia", PROCESSOS_DESCRITIVOS),
    ("julgados_descritivos", "jurisprudencia", JULGADOS_DESCRITIVOS),
    ("temas", "jurisprudencia", TEMAS),
)


def extract_citations(content: str) -> list[dict]:
    """Aplica os padrões e remove candidatos contidos em outro (Rcl dentro de AgInt na Rcl).

    Ocorrências da mesma referência em posições diferentes são preservadas.
    """
    candidates = [
        {"inicio": m.start(), "fim": m.end(), "trecho": m.group(), "tipo": tipo, "familia": familia}
        for familia, tipo, pattern in PATTERNS
        for m in pattern.finditer(content)
    ]
    candidates.sort(key=lambda c: (c["inicio"], -c["fim"]))
    citations: list[dict] = []
    for candidate in candidates:
        if not any(p["inicio"] <= candidate["inicio"] and candidate["fim"] <= p["fim"] for p in citations):
            citations.append(candidate)
    return citations


# ============================================================================
# 3. Âncoras gerais (formas que o regex não conhece)
# ============================================================================
# Âncoras gerais: encontram citações em formatos que o regex não conhece.
#
# O regex (seção 2) reconhece formas catalogadas. As âncoras partem de
# elementos que toda citação tem, e não de uma lista de formatos:
#
#   processo    um número com cara de processo e, colada antes dele, uma frase
#               jurídica (classe/recurso por extenso, abreviada ou sigla);
#   súmula      súmula/enunciado/verbete/SV + número (+ item, tribunal);
#   artigo      art./artigo + número (+ enumerações) + diploma classificável,
#               nas duas ordens ("art. 5º da CF", "art. 5º, CF", "CF, art. 5º");
#   descritiva  tribunal, ano e relator na mesma oração, em qualquer ordem.
#
# Tudo roda no texto limpo (seção 1). As âncoras só ACRESCENTAM citações que
# não se sobrepõem às do regex; `expand_spans` alarga spans do regex com
# tribunal, UF, item de súmula e ano de lei adjacentes. A decisão
# real/inventada continua no resolvedor, contra a base.

# ------------------------------------------------------------------ tribunais

_HONORIFIC = r"(?:(?:E|C|Col|Eg)\.\s*|(?:Col|Eg)\s+|(?:Colendo|Colenda|Egr[eé]gio|Egr[eé]gia)\s+)?"
_COURT_FORMS = {
    "STJ": [r"(?<![A-Za-z])[S5]TJ(?![A-Za-z])", ocr("Superior Tribunal de Justiça")],
    "STF": [r"(?<![A-Za-z])[S5]TF(?![A-Za-z])", ocr("Supremo Tribunal Federal"), ocr("Pretório Excelso"),
            ocr("Excelso Pretório")],
    "TST": [r"(?<![A-Za-z])T[S5]T(?![A-Za-z])", ocr("Tribunal Superior do Trabalho")],
    "TSE": [r"(?<![A-Za-z])T[S5]E(?![A-Za-z])", ocr("Tribunal Superior Eleitoral")],
    "STM": [r"(?<![A-Za-z])[S5]TM(?![A-Za-z])", ocr("Superior Tribunal Militar")],
}
# Apelidos e perífrases destilados (src/lexicon.json, gerado por tools/lexicon.py).
for _form, _sigla in LEXICON.get("tribunal", {}).items():
    if _sigla in _COURT_FORMS:
        _COURT_FORMS[_sigla].append(rf"(?<![A-Za-z]){ocr(_form)}(?![A-Za-z])")
# COURT (sem grupos nomeados) entra em outras regex; COURT_RE identifica a sigla.
COURT = _HONORIFIC + "(?:" + "|".join("|".join(forms) for forms in _COURT_FORMS.values()) + ")"
# Nome por extenso seguido da sigla: "Superior Tribunal de Justiça (STJ)".
COURT_FULL = rf"{COURT}(?:\s*\(\s*{COURT}\s*\))?"
COURT_RE = re.compile(_HONORIFIC + "(?:" + "|".join(
    f"(?P<{sigla}>{'|'.join(forms)})" for sigla, forms in _COURT_FORMS.items()
) + ")", re.IGNORECASE)
_COURT_FULL_RE = re.compile(COURT_FULL, re.IGNORECASE)


def courts_in(text: str) -> list[str]:
    """Siglas dos tribunais mencionados no texto, na ordem."""
    return [m.lastgroup for m in COURT_RE.finditer(text) if m.lastgroup]


# ------------------------------------------------------------------ números e datas

_CNJ_ANCHOR = CNJ_NUMBER.replace(r"[\s.\-–]*", r"[\s.\-–/]*")
NUMBER_RE = re.compile(rf"(?:{_CNJ_ANCHOR}|{PROCESS_SHORT_NUMBER}){NUMBER_END}", re.IGNORECASE)
_YEAR = r"(?:19|[2Z][0OD])[\dOolIgGSsBbLDQqZz|]{2}"
# Ponto final depois do ano é fim de frase; ponto + dígito é bloco de CNJ ("2015.5.03").
YEAR_ALONE = re.compile(rf"(?<![\w/.\-]){_YEAR}(?![\w/\-]|\.\d)")
# Ano dentro de data: "10/10/2020", "07-05-2020", "11.09.2010", "10/2024".
DATE = re.compile(rf"(?<![\w/.\-])(?:[\dOolIBSG]{{1,2}}[./\-]){{1,2}}{_YEAR}(?![\w/\-]|\.\d)")


def _digits(text: str) -> str:
    return re.sub(r"\D", "", text.translate(OCR_DIGITS))


def _is_process_number(m: re.Match, text: str) -> bool:
    raw = m.group()
    if len(_digits(raw)) < 4 or re.fullmatch(_YEAR, raw):
        return False  # curto demais ou ano solto (também com OCR: "2O24", "201g")
    before = text[max(0, m.start() - 4):m.start()]
    return not re.search(r"R\$\s*$|\$\s*$|fls?\.\s*$", before, re.IGNORECASE)


def _years(text: str, lo: int, hi: int) -> list[tuple[int, int]]:
    """Spans de ano (sozinho ou dentro de data) entre lo e hi."""
    dates = [(m.start(), m.end()) for m in DATE.finditer(text, lo, hi)]
    alone = [(m.start(), m.end()) for m in YEAR_ALONE.finditer(text, lo, hi)
             if not any(s <= m.start() < e for s, e in dates)]
    return sorted(dates + alone)


# ------------------------------------------------------------------ vocabulário

CONNECTORS = {"no", "na", "nos", "nas", "em", "de", "do", "da", "dos", "das", "com", "e", "o", "a", "sob"}
# Palavras que, sozinhas, fazem de uma frase a menção a um processo.
CLASS_BEARING = {
    "recurso", "agravo", "embargos", "habeas", "mandado", "reclamacao", "apelacao",
    "rescisoria", "revista", "acao",
}
# Siglas: 3+ letras sem distinção de caixa; 2 letras só como escritas.
ACRONYMS = {
    "resp", "aresp", "respe", "arespe", "arespei", "agresp", "rhc", "rms", "rcl", "recl", "apl",
    "rse", "airr", "arr", "rrag", "agint", "agrg", "edcl", "edv", "eds", "pext", "sls", "ares",
    "adi", "adpf", "rext",
}
ACRONYMS_CASED = {"RE", "AR", "AI", "RO", "RR", "HC", "MS", "ED", "AgR", "Ag", "E"}
SELF_REFERENCE = {"presente", "este", "esta", "deste", "desta", "neste", "nesta"}
_LEGAL_WORDS = set(
    """especial extraordinario ordinario instrumento criminal sentido estrito eleitoral
    interno regimental declaracao divergencia seguranca corpus processo
    liminar suspensao sentenca""".split()
)
# Classes e recursos destilados: sigla de uma palavra vira sigla; em forma de várias
# palavras, a primeira dá sentido de processo e as demais são palavras jurídicas.
for _form in list(LEXICON.get("classe", {})) + list(LEXICON.get("recurso", {})):
    _words = [w for w in _form.replace(".", " ").split() if w not in CONNECTORS]
    if len(_words) == 1 and len(_words[0]) <= 6:
        ACRONYMS.add(_words[0])
    elif _words:
        CLASS_BEARING.add(_words[0])
        _LEGAL_WORDS.update(_words[1:])
# OCR em conectores curtos ("n0", "d0", "dc", "cm") e em siglas ("R5E").
_GLUE_OCR = str.maketrans("0135c", "olese")
_ACRONYM_OCR = str.maketrans("50", "so")
_TOKEN = re.compile(r"[^\s/\-–—(),;:]+")


def _kind(token: str) -> str | None:
    """'class' (dá sentido de processo), 'legal', 'glue' (conector/marcador), 'court' ou None."""
    bare = token.strip(".,;:")
    if not bare or bare in ("º", "°"):
        return "glue"
    folded = fold(bare)
    if is_number_marker(bare) or folded in CONNECTORS or folded.translate(_GLUE_OCR) in CONNECTORS:
        return "glue"
    if COURT_RE.fullmatch(bare):
        return "court"
    undotted = bare.replace(".", "")
    acronym = fold(undotted).translate(_ACRONYM_OCR)
    # Um "c" pode ser "e" lido errado: "Rccl." é "Recl.".
    variants = {acronym} | {acronym[:i] + "e" + acronym[i + 1:] for i, ch in enumerate(acronym) if ch == "c"}
    if bare in ACRONYMS_CASED or undotted in ACRONYMS_CASED or variants & ACRONYMS:
        return "class"
    canon = canonical_words(token).strip(" .")
    if canon in CLASS_BEARING or bare in ("R", "r"):
        return "class"
    if len(undotted) >= 3 and undotted.isalpha() and expand_abbreviation(fold(undotted)) in CLASS_BEARING:
        return "class"
    if canon in _LEGAL_WORDS or (undotted.isalpha() and expand_abbreviation(fold(undotted)) in _LEGAL_WORDS):
        return "legal"
    return None


def _looks_like_class_acronym(word: str) -> bool:
    """Sigla de classe: caixa-alta curta ("REE", "AG") ou mista ("AgIn", "AgRreg").

    Palavra de cabeçalho em caixa-alta ("PARECER", "MILITAR") não conta.
    """
    upper = sum(c.isupper() for c in word)
    if not re.fullmatch(r"[A-Za-z]{2,7}", word) or upper < 2:
        return False
    return len(word) <= 4 if word.isupper() else True


def _phrase_start(text: str, number_start: int, floor: int, strong: bool = False,
                  acronyms: bool = False) -> int | None:
    """Início da frase jurídica colada antes do número, ou None.

    Para em pontuação e em qualquer palavra que não seja classe, recurso ou
    conector. Com número forte (`strong`: CNJ ou 5+ dígitos), aceita vírgula
    entre os elementos e o tribunal colado à classe ou ao número ("STJ, Recurso
    Especial, 1511083"; "RR TST 0010147-…"). Tribunal separado da classe por
    conector é de outra citação ("…Eg. TST e Agravo…") e encerra a frase.
    """
    tokens = list(_TOKEN.finditer(text, max(floor, number_start - 160), number_start))
    kept: list[tuple[re.Match, str]] = []
    right = number_start
    for tok in reversed(tokens):
        gap = text[tok.end():right]
        comma_ok = strong and re.fullmatch(r"\s*,\s*", gap) is not None
        if (re.search(r"[;:()]", gap) or ("," in gap and not comma_ok)) or "\n\n\n" in gap:
            break
        kind = _kind(tok.group())
        if kind in (None, "legal") and acronyms and "\n" not in gap and _looks_like_class_acronym(tok.group().strip(".")):
            kind = "class"  # sigla desconhecida ("REE", "AgIn") antes de CNJ de justiça coberta
        if kind is None:
            break
        if kind == "court" and not (strong and (not kept or kept[-1][1] in ("class", "legal"))):
            break
        kept.append((tok, kind))
        right = tok.start()
    kept.reverse()
    while kept and kept[0][1] == "glue":
        kept.pop(0)
    if not any(kind == "class" for _, kind in kept):
        return None
    # "o presente recurso especial nº ..." é o próprio processo, não citação.
    before = text[max(0, kept[0][0].start() - 20):kept[0][0].start()]
    if any(w in SELF_REFERENCE for w in re.findall(r"\w+", fold(before))[-2:]):
        return None
    return kept[0][0].start()


# ------------------------------------------------------------------ extensões

# UF com OCR no S e no O ("-5P", "R5").
_UF_OCR = UF.replace("S", "[S5]").replace("O", "[O0]")
_UF_SUFFIX = re.compile(
    rf"\s*(?:[/–—-]\s*(?-i:{_UF_OCR})\b|\(\s*(?-i:{_UF_OCR})\s*\)|,\s*(?-i:{_UF_OCR})\b(?!\s*[a-zà-ÿ]))",
    re.IGNORECASE,
)
_COURT_SUFFIX = re.compile(
    rf"\s*(?:\(\s*{COURT_FULL}\s*\)|[,–—-]\s*(?:d[oa]\s+)?{COURT_FULL}(?!\w)|\s+d[oa]\s+{COURT_FULL}(?!\w))",
    re.IGNORECASE,
)
_COURT_PREFIX = re.compile(rf"{COURT_FULL}\s*[,:–—-]\s*$", re.IGNORECASE)
# Incisos vão até LXXVIII; sem C e D, "CC" e "CDC" não viram numeral romano.
_ROMAN = r"(?-i:[IVXL]+)\b"
_SUMULA_TAIL = re.compile(
    rf"(?:\s*,?\s*(?:item\s+|inciso\s+)?{_ROMAN}\s*,?)?"            # item: "Súmula 331, IV, do TST"
    rf"(?:\s+d[aoã]\s+{ocr('súmula')}(?:\s+{ocr('vinculante')})?)?"
    rf"(?:\s*(?:d[oa0]\s+|/|,\s*|[–—-]\s*)?{COURT_FULL}(?!\w)|\s*\(\s*{COURT_FULL}\s*\))?"
    rf"(?:\s*,\s*(?:item\s+)?{_ROMAN})?",
    re.IGNORECASE,
)


def _extend(text: str, start: int, end: int, floor: int = 0) -> tuple[int, int]:
    """Inclui UF e tribunal colados à citação ("…/SP (STJ)", "C. STJ, …") e os recursos
    listados depois do número ("…/RS, Agravo Interno, Embargos de Declaração, STJ")."""
    for _ in range(3):
        m = _UF_SUFFIX.match(text, end) or _COURT_SUFFIX.match(text, end)
        if not m:
            break
        end = m.end()
    return _court_prefix(text, start, floor), _trailing_items(text, end)


_ITEM_SEP = re.compile(r"[ \t]*,[ \t]*")


def _trailing_items(text: str, end: int) -> int:
    """Fim dos itens ", <recurso/tribunal/UF>" colados depois da citação.

    Cada item é uma sequência de palavras de classe/recurso/tribunal/UF (com
    conectores no meio). Item com número é outra citação ("…, AgInt no REsp 2/RJ")
    e encerra a lista; palavra comum encerra o item.
    """
    while sep := _ITEM_SEP.match(text, end):
        prev, item_end = sep.end(), None
        for tok in _TOKEN.finditer(text, sep.end(), min(len(text), sep.end() + 80)):
            if tok.start() != prev and not re.fullmatch(r"[ \t]+", text[prev:tok.start()]):
                break
            if re.search(r"\d", tok.group()):
                item_end = None  # número: começa outra citação
                break
            bare = tok.group().strip(".")
            kind = "uf" if re.fullmatch(_UF_OCR, bare) else _kind(tok.group())
            if kind is None:
                break
            if kind != "glue":
                # Ponto final de frase fica fora ("…, STF."); ponto de abreviação fica ("Ag. Reg.").
                sentence_dot = tok.group().endswith(".") and (bare.isupper() or len(bare) > 5)
                item_end = tok.end() - sentence_dot
            prev = tok.end()
        if item_end is None:
            break
        end = item_end
    return end


def _court_prefix(text: str, start: int, floor: int) -> int:
    window = max(floor, start - 80)
    m = _COURT_PREFIX.search(text[window:start])
    return window + m.start() if m else start


def expand_spans(text: str, citations: list[dict]) -> list[dict]:
    """Alarga spans do regex com o que está colado a eles (tribunal, UF, item, ano)."""
    out: list[dict] = []
    for c in sorted(citations, key=lambda c: c["inicio"]):
        floor = max([o["fim"] for o in out if o["fim"] <= c["inicio"]], default=0)
        start, end = c["inicio"], c["fim"]
        if c["familia"] in ("processos", "processos_trabalhistas"):
            if _year_before_relator(text, c):
                continue  # "Resp 2019, Rel. Min. …": descritiva, não número de processo
            left = _phrase_start(text, start, floor)
            start, end = _extend(text, min(left, start) if left is not None else start, end, floor)
        elif c["familia"] == "artigos":
            year = re.match(r"(?:/\s*\d{2,4}|\s+de\s+\d{4})(?!\d)", text[end:end + 12])
            end += year.end() if year else 0
            if not fold(c["trecho"]).startswith("art"):  # "CPM, art. 290, incisos I a III"
                while (enum := _ENUM.match(text, end)) and enum.end() - c["fim"] < 80:
                    end = enum.end()
        elif c["familia"] == "sumulas":
            tail = _SUMULA_TAIL.match(text, end)
            end = tail.end() if tail else end
            start = _court_prefix(text, start, floor)
        if (start, end) != (c["inicio"], c["fim"]):
            c = c | {"inicio": start, "fim": end, "trecho": text[start:end]}
        out.append(c)
    return out


def _year_before_relator(text: str, c: dict) -> bool:
    number = re.search(r"\S+$", c["trecho"]).group()
    return bool(re.fullmatch(_YEAR, number)) and bool(
        re.match(rf"[^.;\n]{{0,40}}?(?<!\w)(?:{_REL_CUE_PATTERN})", text[c["fim"]:c["fim"] + 80], re.IGNORECASE))


# ------------------------------------------------------------------ âncoras

def _overlaps(start: int, end: int, taken: list[tuple[int, int]]) -> bool:
    return any(s < end and start < e for s, e in taken)


def _floor(taken: list[tuple[int, int]], pos: int) -> int:
    return max([e for s, e in taken if e <= pos], default=0)


def _candidate(text: str, start: int, end: int, familia: str, tipo: str) -> dict:
    return {"inicio": start, "fim": end, "trecho": text[start:end], "tipo": tipo,
            "familia": familia, "origem": "ancora"}


_SECTION_NUMBER = re.compile(r"\d{1,2}\.?\s*[.)\-–—]\s+[A-ZÀ-Ý]{2}")


def _section_number(m: re.Match, text: str) -> bool:
    """Número de título de seção ("3. DOS PEDIDOS", "2 - DO MÉRITO"): início de linha, 1–2 dígitos."""
    line_start = text.rfind("\n", 0, m.start()) + 1
    return not text[line_start:m.start()].strip() and bool(_SECTION_NUMBER.match(text, m.start()))


def _process_anchors(text: str, taken: list[tuple[int, int]]) -> list[dict]:
    found = []
    for m in NUMBER_RE.finditer(text):
        if not _is_process_number(m, text) or _overlaps(m.start(), m.end(), taken) or _section_number(m, text):
            continue
        floor = _floor(taken, m.start())
        digits = _digits(m.group())
        start = _phrase_start(text, m.start(), floor, strong=len(digits) >= 5)
        if start is None and len(digits) >= 14 and cnj_justice(cnj_key(digits) or (0, "0" * 13)) is not None:
            start = _phrase_start(text, m.start(), floor, strong=True, acronyms=True)
        if start is None:
            continue
        start, end = _extend(text, start, m.end(), floor)
        if _overlaps(start, end, taken):
            continue
        found.append(_candidate(text, start, end, "processos", "jurisprudencia"))
        taken.append((start, end))
    return found


_SUMULA_CUE = re.compile(
    rf"(?<!\w)(?:{ocr('súmula')}s?|{ocr('sumular')}|s[uú]m\.|{ocr('enunciado')}s?|{ocr('verbete')}s?"
    r"|(?-i:SV))",
    re.IGNORECASE,
)
_GAP_WORDS = {"vinculante", "sumula", "sumular", "de", "da", "do"}
_SUMULA_NUM = re.compile(r"(?<![\w.])(?:\d|[lIBgGSOQZ|](?=[\dOolISsgGBbDQqZz|]{0,3}\d))"
                         r"[\dOolISsgGBbDQqZz|]{0,3}(?![\w|])")


def _sumula_gap_ok(gap: str) -> bool:
    """Entre a pista e o número só cabem: marcador, conector, vinculante, tribunal."""
    rest = COURT_RE.sub(" ", gap)
    for tok in re.findall(r"[^\s\-–—,]+", rest):
        bare = tok.strip(".")
        if not bare or is_number_marker(tok) or bare in ("º", "°"):
            continue
        if canonical_words(tok).strip(" .") not in _GAP_WORDS:
            return False
    return True


def _sumula_anchors(text: str, taken: list[tuple[int, int]]) -> list[dict]:
    found = []
    for cue in _SUMULA_CUE.finditer(text):
        if _overlaps(cue.start(), cue.end(), taken):
            continue
        for num in _SUMULA_NUM.finditer(text, cue.end(), min(len(text), cue.end() + 80)):
            if not _sumula_gap_ok(text[cue.end():num.start()]):
                break
            tail = _SUMULA_TAIL.match(text, num.end())
            end = tail.end() if tail else num.end()
            span = text[cue.start():end]
            weak_cue = re.match(r"(?:enunciad|verbet)", fold(cue.group()))
            strong = re.search(r"s[uú]mul|vincul", canonical_words(span)) or COURT_RE.search(span)
            if weak_cue and not strong:
                break
            start = _court_prefix(text, cue.start(), _floor(taken, cue.start()))
            if not _overlaps(start, end, taken):
                found.append(_candidate(text, start, end, "sumulas", "jurisprudencia"))
                taken.append((start, end))
            break
    return found


_ART_CUE = re.compile(
    rf"(?<!\w)(?:{ocr('artigo')}s?|[aáàâão]r[tf]s?\.?)[\s.\-–]*(?:n[º°o.]*\s*)?(?P<num>{ARTICLE_NUMBER})",
    re.IGNORECASE,
)
# Enumeração depois do número: "caput", "inciso LV", "incisos I a III", "§§ 1º e 2º",
# "par. único", "alínea 'a'"/"alínea b", "I, 'a'", com vírgula, espaço, "e" ou "a" entre itens.
_ENUM_ITEM = (
    rf"(?:{ocr('caput')}"
    rf"|(?:{ocr('incisos')}|{ocr('inciso')}|incs?\.)\s*(?:{_ROMAN}|(?-i:[ivxlc]+)\b)"
    rf"|{_ROMAN}|(?-i:[IVXL]*[l|][IVXLl|]*)(?![\w|])"  # inciso com OCR: "lX", "l"
    rf"|§§?\s*[\dlIZSGB|]+[º°o]?(?:-[A-Z])?(?:\s*(?:,|e)\s*[\dlIZSGB|]+[º°o]?)*"
    rf"|(?:{ocr('parágrafo')}|{ocr('par')}\.)\s*{ocr('único')}"
    rf"|(?:{ocr('alínea')}|al\.|{ocr('letra')})\s*['\"‘’]?[a-z]['\"‘’]?(?!\w)"
    rf"|['\"‘’][a-z]['\"‘’])"
)
_ENUM = re.compile(rf"(?:\s*,\s*|\s+(?:e|[aã])\s+|\s+){_ENUM_ITEM}", re.IGNORECASE)
# Entre artigo e diploma: "do/da" (com OCR), vírgula, travessão ou só espaço.
_PREPS = [re.compile(p, re.IGNORECASE) for p in (
    r"\s*,?\s*(?:(?:d|cl)[oaã0ó]s?|n[oa])\s+", r"\s*,\s*", r"\s*[–—-]\s*", r"\s+")]  # "cla": OCR de "da"
_PHRASE_WORD = re.compile(r"[^\s,;]+")
# Palavras que podem compor o nome de um diploma (além de Maiúsculas, números
# e conectores). Radicais gerais; nada específico de catálogo de teste.
_DIPLOMA_WORDS = (
    "codigo", "lei", "leis", "decreto", "complementar", "constitu", "consolid", "trabalh",
    "celetis", "process", "civil", "penal", "militar", "eleitoral", "consumid", "defesa",
    "federal", "republica", "diploma", "magna", "maior", "carta", "inelegib", "brasileir",
    "politic", "vigente", "lc",
) + tuple(sorted({w for f in LEXICON.get("diploma", {}) for w in f.replace(".", " ").split()
                  if len(w) >= 4 and w not in CONNECTORS}))


# Número de lei com OCR, "g.504/1997", "13.467/Z017", "(8.078/90)".
_LAW_NUMBER_WORD = re.compile(r"\(?[\dlIOSGBgqZ|][\dlIOoSsGgqBbZz|.\s\-]*/\s*[\dOolIGgqSsZzBb|]{2,4}\)?")  # "13.-467/2017"


def _diploma_word(word: str) -> bool:
    bare = word.strip("().,;:'\"")
    if not bare:
        return True
    if bare[0].isupper() or bare[0].isdigit() or bare in ("nº", "n.º", "n°", "riº", "ri°", "uº", "u°", "/") or _LAW_NUMBER_WORD.fullmatch(bare):
        return True
    return _diploma_content(bare) or fold(bare).translate(_GLUE_OCR) in CONNECTORS


def _diploma_content(word: str) -> bool:
    """Palavra que por si indica diploma: radical ("constitu", "codigo") ou sigla ("clt")."""
    bare = word.strip("().,;:'\"")
    low = canonical_words(bare).strip(" .")
    return any(low.startswith(stem) for stem in _DIPLOMA_WORDS) or bool(diploma_key(bare))


def _diploma_last(word: str) -> bool:
    """Última palavra de um nome de lei: radical/sigla, número de lei, ano ou ')'."""
    bare = word.rstrip(".,;:")
    return (bare.endswith(")") or _diploma_content(bare)
            or _LAW_NUMBER_WORD.fullmatch(bare) is not None or re.fullmatch(r"(?:19|20)\d\d", bare) is not None)


def _ends_sentence(word: str) -> bool:
    """'Confiram-se:', 'autos.' e '2022.' fecham frase; 'Esp.', 'n.' e 'Lei.' não."""
    return word.endswith((":", ";")) or (word.endswith(".") and (
        word[0].isdigit() or len(word.strip(".")) > 5))


def _longest_diploma(text: str, start: int, max_words: int = 12) -> int | None:
    """Fim do maior trecho a partir de `start` que é um diploma reconhecível."""
    words = []
    prev_end = start
    for w in _PHRASE_WORD.finditer(text, start, min(len(text), start + 160)):
        if re.search(r"\n\s*\n\s*\n|[,;]", text[prev_end:w.start()]) or not _diploma_word(w.group()):
            break
        words.append(w)
        prev_end = w.end()
        if len(words) >= max_words or re.search(r"[.;]$", w.group()) and not re.search(r"\b\w{1,5}\.$", w.group()):
            break
    for i in range(len(words), 0, -1):
        end = words[i - 1].end()
        if not _diploma_last(words[i - 1].group()):
            continue  # "Constituição Federal e Súmula 331" não termina em nome de lei
        chunk = text[start:end].rstrip(".,;:")
        if chunk.count("(") > chunk.count(")") and text[end:end + 1] == ")":
            chunk += ")"
        if diploma_key(chunk):
            return start + len(chunk)
    return None


def _diploma_before(text: str, head_end: int) -> int | None:
    """Início do diploma que termina em `head_end` ("CF", "Código Civil (Lei …)")."""
    allowed = []  # sufixo contínuo de palavras que podem compor nome de lei
    right = head_end
    for w in reversed(list(_PHRASE_WORD.finditer(text, max(0, head_end - 100), head_end))):
        gap = text[w.end():right]
        if re.search(r"[,;]|\n\s*\n\s*\n", gap) or not _diploma_word(w.group()) or _ends_sentence(w.group()):
            break
        allowed.append(w)
        right = w.start()
    for w in reversed(allowed):  # do mais longo ao mais curto
        # Começa por palavra de diploma: nem conector ("e Lei …") nem Maiúscula solta ("Ver CF").
        if _diploma_content(w.group()) and diploma_key(text[w.start():head_end]):
            return w.start()
    return None


def article_diploma(trecho: str) -> str | None:
    """Chave do diploma de uma citação de artigo, nas duas ordens."""
    cue = _ART_CUE.search(trecho)
    if not cue:
        return None
    end = cue.end()
    while enum := _ENUM.match(trecho, end):
        end = enum.end()
    for prep in _PREPS:
        m = prep.match(trecho, end)
        dip_end = _longest_diploma(trecho, m.end()) if m else None
        if dip_end:
            return diploma_key(trecho[m.end():dip_end])
    head = trecho[:cue.start()].rstrip(" ,:–—-(\n\t")
    return diploma_key(head) if head else None


def _article_anchors(text: str, taken: list[tuple[int, int]]) -> list[dict]:
    found = []
    for cue in _ART_CUE.finditer(text):
        if _overlaps(cue.start(), cue.end(), taken):
            continue
        end = cue.end()
        while (enum := _ENUM.match(text, end)) and enum.end() - cue.end() < 80:
            end = enum.end()
        span = None
        for prep in _PREPS:
            m = prep.match(text, end)
            dip_end = _longest_diploma(text, m.end()) if m else None
            if dip_end:
                span = (cue.start(), dip_end)
                break
        if span is None:
            # Ordem invertida: "<diploma>, art. N" / "<diploma> - art. N".
            m = re.search(r"\s*[,:–—-]\s*$", text[max(0, cue.start() - 6):cue.start()])
            if m:
                head_end = max(0, cue.start() - 6) + m.start()
                start = _diploma_before(text, head_end)
                if start is not None:
                    span = (start, end)
        if span and not _overlaps(*span, taken):
            found.append(_candidate(text, span[0], span[1], "artigos", "lei"))
            taken.append(span)
    return found


# ------------------------------------------------------------------ descritivas

_REL_CUE_PATTERN = (
    rf"{ocr('relatoria')}|{ocr('relatado')}|{ocr('relatada')}|{ocr('relatora')}|{ocr('relator')}"
    rf"|R[ec]l(?:[.\-–]|(?=\s+[A-ZÀ-Ý]))|{ocr('Ministra')}|{ocr('Ministro')}|M[il1í]n(?:[.\-–]|(?=[ªº]|\s))"
)
_REL_CUE = re.compile(rf"(?<!\w)(?:{_REL_CUE_PATTERN})", re.IGNORECASE)
_REL_GLUE = re.compile(
    rf"(?:[ªº]|\s*:|\s*[.\-–]+|\s+(?:(?:de|do|da|d[oa0c]|pelo|pela|p[ec]lo|{ocr('Ministro')}|{ocr('Ministra')}"
    rf"|ju[ií]za?|desembargador[a]?|convocad[oa])(?!\w)"
    rf"|M[il1í]n(?:[.\-–]|(?=[ªº\s]))[ªº]?|Des\.|R[ec]l\.[ªº]?|p(?:/|\.)\s*(?:o\s+)?ac(?:[oó]rd[aã]o|\.)))*\s*",
    re.IGNORECASE,
)
_PARTICLES = {"de", "da", "do", "dos", "das"}
_NAME_WORD = r"(?-i:(?:[A-ZÀ-ÖØ-Þ]|[501](?=[a-zà-ÿ]))[A-Za-zÀ-ÖØ-öø-ÿ0-9']+)"
_NAME_WORD_LOWER = r"[a-zà-öø-ÿ][a-zà-öø-ÿ0-9']+"
_PARTICLE = r"(?:de|da|do|dos|das|De|Da|Do|Dos|Das|DE|DA|DO|DOS|DAS)"
_NAME = re.compile(rf"{_NAME_WORD}(?:\s+(?:{_PARTICLE}\s+)?{_NAME_WORD})*")
_NAME_LOWER = re.compile(rf"{_NAME_WORD_LOWER}(?:\s+(?:{_PARTICLE}\s+)?{_NAME_WORD_LOWER})*")
_NOT_NAME = {
    "relator", "relatora", "ministro", "ministra", "tribunal", "superior", "supremo", "justica",
    "federal", "eleitoral", "militar", "trabalho", "turma", "secao", "plenario", "corte", "stj",
    "stf", "tst", "tse", "stm", "egregio", "colendo", "excelso", "pretorio", "julgado", "julgamento",
    "em", "no", "na", "rel", "min",
}
_DESCRIPTOR = {
    "julgado", "julgamento", "precedente", "acordao", "decisao", "colegiada", "colegiado",
    "relatado", "relatada", "proferido", "proferida", "aresto", "decisum",
}


def relator_name(text: str) -> tuple[int, int, str] | None:
    """(início da pista, fim do nome, nome) do primeiro relator citado no texto.

    O nome é cortado na primeira palavra que não pode ser nome (título,
    tribunal, "julgado"...) e precisa de duas palavras. Em trechos todo em
    minúsculas, aceita nome em minúsculas.
    """
    for cue in _REL_CUE.finditer(text):
        glue = _REL_GLUE.match(text, cue.end())
        pos = glue.end() if glue else cue.end()
        segment = re.split(r"[,;()\n.]", text[pos:pos + 60])[0]
        lower = not re.search(r"[A-ZÀ-Ý]", segment)
        name = (_NAME_LOWER if lower else _NAME).match(text, pos)
        if not name:
            continue
        end, count = pos, 0
        for tok in re.finditer(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9']+", name.group()):
            word = fold(tok.group())
            if word in _PARTICLES:
                continue
            if word in _NOT_NAME or COURT_RE.fullmatch(tok.group()):
                break
            end, count = pos + tok.end(), count + 1
        if count >= 2:
            return cue.start(), end, text[pos:end]
    return None


# Rabo de citação numerada: ", Terceira Turma, julgado em 10/10/2020, DJe …, Rel. Min."
_ORDINAL = r"(?:\d+[ªº]|Primeir[ao]|Segund[ao]|Terceir[ao]|Quart[ao]|Quint[ao]|Sext[ao]|S[eé]tim[ao]|Oitav[ao])"
_TAIL = re.compile(
    rf"""(?:[\s,;:()\-–—]+
      | {_ORDINAL}\s+(?:Turma|Se[cç][aã]o|C[aâ]mara)
      | T\d | Turma | Se[cç][aã]o | Plen[aá]rio | Pleno | Corte\s+Especial | [OÓ]rg[aã]o\s+Especial
      | julgad[oa]s?(?:\s+em)? | julgamento(?:\s+em)? | j\. | publicad[oa]\s+em | DJe? | DJU | em | de | do | da | no | na
      | \d{{1,2}}[./-]\d{{1,2}}[./-]\d{{2,4}} | \d{{1,2}}/\d{{4}}
      | {_REL_CUE_PATTERN} | [ªº] | p/\s*ac[oó]rd[aã]o
    )*""",
    re.IGNORECASE | re.VERBOSE,
)


def _clause(text: str, pos: int) -> tuple[int, int]:
    """Limites da oração em volta de `pos` (';', quebra dupla, fim de frase real)."""
    base = max(0, pos - 220)
    right = min(len(text), pos + 220)
    left = base
    boundary = (r";|\n\s*\n|(?<=[a-zà-ÿ0-9\)]{5})\.\s+(?=[A-ZÀ-Ý])|,?\s+bem\s+como\s+"
                # " e " que abre outra citação: "…Rel. Min. X e REsp …", "… e acórdão do STJ …"
                r"|\s+e\s+(?=[A-ZÀ-Ý]|(?:o|a)?\s*(?:julgad|precedent|ac[oó]rd|arest|decis))")
    for m in re.finditer(boundary, text[base:pos]):
        left = base + m.end()
    m = re.search(boundary + r"|\.\s*$", text[pos:right])
    if m:
        right = pos + m.start()
    return left, right


def _court_mentions(text: str, lo: int, hi: int) -> list[tuple[int, int]]:
    """Tribunais citados; "Superior Tribunal de Justiça (STJ)" conta como um só."""
    return [(m.start(), m.end()) for m in _COURT_FULL_RE.finditer(text, lo, hi) if COURT_RE.search(m.group())]


def known_relator(text: str, known: list[frozenset]) -> tuple[int, int, str] | None:
    """(início, fim, nome) do primeiro nome, sem pista, que é de um relator da base.

    "TSE, 2017, Gilmar Mendes": todas as palavras do nome (2+) pertencem ao
    mesmo relator do acervo. A lista vem da base, não de um catálogo.
    """
    if not known:
        return None
    for m in _NAME.finditer(text):
        words = list(re.finditer(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9']+", m.group()))
        for i in range(len(words)):
            for j in range(len(words), i + 1, -1):
                tokens = set(name_tokens(m.group()[words[i].start():words[j - 1].end()]))
                if len(tokens) >= 2 and any(tokens <= rel for rel in known):
                    s, e = m.start() + words[i].start(), m.start() + words[j - 1].end()
                    return s, e, text[s:e]
    return None


def _relators(text: str, known: list[frozenset] = ()) -> list[tuple[int, int]]:
    """Spans "pista + nome" de todos os relatores citados ("Min. Rel. Fulano (relator)"),
    mais nomes sem pista que são de relatores da base."""
    spans = []
    for cue in _REL_CUE.finditer(text):
        if spans and cue.start() < spans[-1][1]:
            continue  # pista dentro do relator anterior ("Min. Rel.")
        rel = relator_name(text[cue.start():cue.start() + 200])
        if not rel or rel[0] != 0:
            continue
        end = cue.start() + rel[1]
        paren = re.match(rf"\s*\(\s*(?:{ocr('relatora')}|{ocr('relator')})\s*\)", text[end:])
        spans.append((cue.start(), end + (paren.end() if paren else 0)))
    pos = 0
    while known and (hit := known_relator(text[pos:], known)):
        s, e = pos + hit[0], pos + hit[1]
        if not any(a < e and s < b for a, b in spans):
            spans.append((s, e))
        pos = e
    return sorted(spans)


def _descriptive_anchors(text: str, taken: list[tuple[int, int]], known: list[frozenset] = ()) -> list[dict]:
    found = []
    relators = _relators(text, known)
    for rel_start, rel_end in relators:
        if _overlaps(rel_start, rel_end, taken):
            continue
        lo, hi = _clause(text, rel_start)
        # Elementos da MESMA citação: não atravessam outro relator nem outra citação.
        lo = max([lo] + [e for s, e in relators if e <= rel_start])
        hi = min([hi] + [s for s, e in relators if s >= rel_end])

        def free(spans):
            return [s for s in spans if not _overlaps(*s, taken)]

        courts = free(_court_mentions(text, lo, hi))
        years = free(_years(text, lo, hi))
        if not courts or not years:
            continue

        def dist(s):
            return max(0, rel_start - s[1], s[0] - rel_end)

        court, year = min(courts, key=dist), min(years, key=dist)
        start = min(court[0], year[0], rel_start)
        end = max(court[1], year[1], rel_end)
        if end - start > 220:
            continue
        # Número de processo dentro do trecho: é atributo de citação numerada.
        if any(_is_process_number(n, text) for n in NUMBER_RE.finditer(text, start, end)):
            continue
        # Rabo de citação numerada ("REsp 1/SP, Terceira Turma do STJ, Rel. Min. …").
        prev = [e for s, e in taken if lo <= e <= start]
        if prev and _TAIL.fullmatch(text, max(prev), start):
            continue
        start = _descriptor_start(text, start, lo)
        if text[start:end].count("(") > text[start:end].count(")") and text[end:end + 1] == ")":
            end += 1
        if _overlaps(start, end, taken):
            continue
        found.append(_candidate(text, start, end, "julgados_descritivos", "jurisprudencia"))
        taken.append((start, end))
    return found


def _descriptor_start(text: str, start: int, floor: int) -> int:
    """Inclui 'julgado do', 'Reclamação (', 'aresto relatado pelo' à esquerda."""
    tokens = list(_TOKEN.finditer(text, max(floor, start - 60), start))
    new_start = start
    for tok in reversed(tokens):
        if re.search(r"[,;:)]", text[tok.end():new_start]):
            break
        word = canonical_words(tok.group()).strip(" .")
        if word in CONNECTORS or word in _DESCRIPTOR or _kind(tok.group()) in ("class", "legal"):
            new_start = tok.start()
            continue
        break
    # Não começa por conector.
    while True:
        m = re.match(r"(\w+)\s+", text[new_start:start + 1])
        if m and fold(m.group(1)) in CONNECTORS and new_start + m.end() <= start:
            new_start += m.end()
        else:
            break
    return new_start


def find_anchors(text: str, citations: list[dict], known_relators: list[frozenset] = ()) -> list[dict]:
    """Citações novas (não sobrepostas às do regex), em ordem de posição.

    `known_relators` (tokens dos relatores da base) permite achar descritivas
    sem pista de relatoria.
    """
    taken = [(c["inicio"], c["fim"]) for c in citations]
    new = []
    new += _sumula_anchors(text, taken)
    new += _article_anchors(text, taken)
    new += _process_anchors(text, taken)
    new += _descriptive_anchors(text, taken, known_relators)
    return sorted(new, key=lambda c: c["inicio"])
