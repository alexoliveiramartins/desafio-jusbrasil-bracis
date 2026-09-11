r"""Padrões de superfície: identificam citações, não sua validade jurídica.

Os componentes são compartilhados entre famílias. Não normalizamos o texto
inteiro: \s reconhece espaços e quebras reais sem alterar os spans.
"""

import re


FLAGS = re.IGNORECASE | re.VERBOSE
NUMBER_MARKER = r"(?:n[º°o.]?\s*)?"
UF = r"(?:AC|AL|AP|AM|BA|CE|DF|ES|GO|MA|MT|MS|MG|PA|PB|PR|PE|PI|RJ|RN|RS|RO|RR|SC|SP|SE|TO)"
TRIBUNAL = r"(?:STF|STJ|TST|TSE|STM)"
# Mantém os tamanhos dos blocos CNJ, aceitando separadores ausentes,
# repetidos ou substituídos por espaços. Não altera nenhum dígito.
CNJ_SEPARATOR = r"[\s.\-–]*"
CNJ_NUMBER = CNJ_SEPARATOR.join((r"\d{1,7}", r"\d{2}", r"\d{4}", r"\d", r"\d{2}", r"\d{4}"))
SHORT_NUMBER = r"\d+(?:\s*\.\s*\d+)*"
# Espaços entre milhares são aceitos só em grupos de três dígitos.
# O formato de leis permanece separado para não ampliar essa família.
PROCESS_SHORT_NUMBER = r"(?:\d{1,3}(?:[.\s]+\d{3})+|\d+)"
PROCESS_NUMBER = rf"(?:{CNJ_NUMBER}|{PROCESS_SHORT_NUMBER})"
# Não aceita um prefixo numérico se outro bloco de dígitos vem após ponto
# ou hífen. Ex.: um OCR não suportado em 1.234.56g não deve virar só 1.234.
NUMBER_END = r"(?!\w)(?!\s*[.\-–]\s*\d)"
STATE_SUFFIX = rf"(?:\s*(?:[/–-]\s*{UF}\b|\({UF}\)))?"

PROCESS_CLASS = r"""(?:
    Reclamação | Recl | Rcl
    | Agravo\s+em\s+Recurso\s+Especial
    | Recurso\s+Especial\s+Eleitoral
    | Recurso\s+Especial | Rec\.\s*Esp\.
    | Recurso\s+Extraordinário
    | Recurso\s+em\s+Habeas\s+Corpus
    | Recurso\s+em\s+Mandado\s+de\s+Segurança
    | Suspensão\s+de\s+Liminar\s+e\s+de\s+Sentença
    | Agravo\s+de\s+Instrumento
    | AREspEI | AgREsp | A\.?REsp | REspe | R\.?Esp | RHC | RMS | RE
    | H\.?C | AR | AI | APL | RSE | R-Rp | Ag\.?\s*Int
)"""
APPEAL_PREFIX = r"""(?:
    Ag\.?\s*Int\.? | AgRg | AgR | EDcl | EDs | AG\.REG\.?
    | Agravo\s+Interno | Agravo\s+Regimental
    | Embargos\s+de\s+Declaração
)"""

PROCESSOS = re.compile(
    rf"\b(?:{APPEAL_PREFIX}(?:\s+n[oa]s?\s+|\s*-\s*))*"
    rf"{PROCESS_CLASS}\.?\s+{NUMBER_MARKER}{PROCESS_NUMBER}{NUMBER_END}{STATE_SUFFIX}",
    FLAGS,
)

# A identificação trabalhista usa cadeias como TST-ED-E-ED-RR-<CNJ>.
PROCESSOS_TRABALHISTAS = re.compile(
    rf"\b(?:processo\s+{NUMBER_MARKER})?(?:TST\s*-\s*)?"
    rf"(?:(?:ED|E|Ag)\s*-\s*)*(?:Ag)?(?:ARR|RR)\s*-\s*"
    rf"{CNJ_NUMBER}{NUMBER_END}{STATE_SUFFIX}",
    FLAGS,
)

SUMULAS = re.compile(
    rf"\b(?:Súmula|Súm\.)\s+(?:Vinculante\s+)?"
    rf"{NUMBER_MARKER}\d+\b(?:\s+d[oa]\s+{TRIBUNAL}\b)?",
    FLAGS,
)

DIPLOMA = r"""(?:
    Código\s+de\s+Processo\s+Civil
    | Código\s+de\s+Processo\s+Penal
    | Código\s+de\s+Defesa\s+do\s+Consumidor
    | Código\s+Penal\s+Militar
    | Código\s+Civil | Código\s+Penal | Código\s+Eleitoral
    | Constituição\s+da\s+República | Constituição\s+Federal
    | Consolidação\s+das\s+Leis\s+do\s+Trabalho
    | CPC | CPP | CDC | CPM | CC | CP | CLT | CF(?:/88)?
)"""
LAW_NUMBER = rf"Lei\s+(?:Complementar\s+)?{NUMBER_MARKER}{SHORT_NUMBER}\s*/\s*\d{{4}}"
ARTICLE_NUMBER = r"\d+(?:\.\d+)*(?:[º°o])?(?:-[A-Z])?"
PARAGRAPH = rf"§\s*{ARTICLE_NUMBER}|parágrafo\s+único"
INCISO = r"(?:inciso\s+)?[IVXLCDM]+\b"
ALINEA = r"(?:alínea\s+)?['\"‘’][a-z]['\"‘’]"

ARTIGOS = re.compile(
    rf"\b(?:artigo|art\.?)\s+{ARTICLE_NUMBER}"
    rf"(?:\s*,\s*(?:{PARAGRAPH}|{INCISO}|{ALINEA}))*"
    rf"\s*,?\s*d[oa]\s+(?:{LAW_NUMBER}|{DIPLOMA})(?!\w)",
    FLAGS,
)

# Referências processuais descritivas, sem fixar nomes de ministros.
# (?-i:...) mantém a inicial maiúscula como sinal de continuação do nome.
NAME_WORD = r"(?-i:[A-ZÀ-ÖØ-Þ][a-zA-ZÀ-ÖØ-öø-ÿ]*)"
RELATOR_NAME = rf"{NAME_WORD}(?:\s+(?:(?:de|da|do|dos|das)\s+)?{NAME_WORD})*"
PROCESSOS_DESCRITIVOS = re.compile(
    rf"\b{PROCESS_CLASS}\.?\s+(?:do\s+{TRIBUNAL}\s*,?\s*)?"
    rf"de\s+\d{{4}}\s*,\s*Rel\.\s*Min\.\s*{RELATOR_NAME}",
    FLAGS,
)

# Referências sem número, identificadas por tribunal, ano e relator.
# As alternativas descrevem a frase; nomes e anos não são enumerados.
JULGADOS_DESCRITIVOS = re.compile(
    rf"\b(?:julgado|acórdão|precedente)\s+do\s+{TRIBUNAL}\s+"
    rf"(?:(?:proferido|julgado)\s+em|de)\s+\d{{4}}"
    rf"\s*,?\s+(?:pela|da|sob)\s+relatoria\s+de\s+{RELATOR_NAME}",
    FLAGS,
)

# Registro: nome da família, tipo do contrato, expressão compilada.
PATTERNS = (
    ("processos", "jurisprudencia", PROCESSOS),
    ("processos_trabalhistas", "jurisprudencia", PROCESSOS_TRABALHISTAS),
    ("sumulas", "jurisprudencia", SUMULAS),
    ("artigos", "lei", ARTIGOS),
    ("processos_descritivos", "jurisprudencia", PROCESSOS_DESCRITIVOS),
    ("julgados_descritivos", "jurisprudencia", JULGADOS_DESCRITIVOS),
)
