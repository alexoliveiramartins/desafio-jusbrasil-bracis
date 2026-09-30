"""Camada de NLP: modelos abertos (Hugging Face, GGUF) revisam o que as regras acharam.

Uso::

    python -m src.main --input data/txt --output resultados --nlp
    python -m src.main ... --nlp --nlp-extra-model hf.co/Qwen/Qwen3-8B-GGUF:Q4_K_M

Cada modelo roda num servidor local (Ollama, API nativa) e lê a peça em trechos, devolvendo em
JSON as citações que encontrou: o trecho literal e os campos lidos (tribunal, classe, número,
UF, diploma, artigo, ano, relator), já com o OCR corrigido. Com isso a camada faz duas coisas:

- normalização: uma citação que as regras deixaram ``inventada`` ou ``incompleta`` é consultada
  de novo na base, pelos campos lidos;
- recall: uma citação que as regras não extraíram entra como candidata.

Com dois modelos (conjunto), as citações novas são a união das leituras, e a normalização só
vale se as leituras não levarem a registros diferentes; em conflito, fica a resposta das regras.

Notes
-----
O modelo nunca decide a classe: a consulta à base continua determinística
(``classify.resolve`` sobre uma forma canônica montada com os campos). Guardas contra o erro
grave (``inventada`` -> ``real``):

- o número lido precisa ser um dos números do trecho, inteiro, trocando só letra por dígito
  parecido (O->0, l->1, S->5...), nunca dígito por dígito;
- tribunal, classe e diploma precisam de evidência no trecho;
- números depois de palavras que não abrem citação ("Resolução nº", "nº de ordem", "fls.") e
  o número dos autos do próprio documento são recusados;
- nada que as regras resolveram como ``real`` é renormalizado.

Sem servidor (ou com erro), a camada se desliga e o pipeline segue só com as regras. Só
biblioteca padrão.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

from .classify import INCOMPLETA, INVENTADA, REAL, CanonicalIndex, Resolution, resolve
from .normalize import OCR_DIGITS, UFS, class_code, diploma_key, fold, name_tokens, number_group_spans, number_groups
from .spans import _ART_CUE, _ENUM, _PREPS, _SUMULA_CUE, _longest_diploma, clean, courts_in, to_original

DEFAULT_MODEL = "hf.co/unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M"
DEFAULT_URL = "http://127.0.0.1:11434"

# Confiança das decisões da camada, medida em conjuntos sintéticos rotulados pela base (28/09/2026, ~7.000 citações):
# real_nlp 9/9, inventada_nlp 11/11, citações novas 59/59. Com margem: o sintético é mais fácil que o cego.
NLP_CONFIDENCE = {"real_nlp": 0.92, "inventada_nlp": 0.92, "incompleta_nlp": 0.85, "novo": 0.92}

SYSTEM = ("Você anota citações em peças jurídicas brasileiras para um sistema de verificação. "
          "Responda somente com JSON.")
PROMPT = """Liste TODAS as citações de jurisprudência e de legislação do texto, na ordem em que aparecem.

Conta como citação:
- processo identificado por número, com ou sem classe ("AgInt no REsp 1.234.567/SP", "RR-1000-10.2015.5.02.0001");
- súmula ("Súmula 83 do STJ", "Súmula Vinculante 10");
- artigo de lei ou de código ("art. 5º, LV, da Constituição Federal", "art. 373, I, do CPC");
- julgado citado sem número, por tribunal, ano e relator ("precedente do STF de 2024, da relatoria de Fulano");
- tema de repercussão geral ou de recurso repetitivo.
NÃO conta: número dos autos do próprio documento, protocolo, OAB, CPF, folhas, valores, datas, lei citada sem
artigo e referências vagas ("jurisprudência pacífica", "dispositivo legal de regência", "súmula aplicável").

"trecho" é SÓ a citação, copiada exatamente como está no texto (com erros de OCR, espaços e quebras de linha),
sem a frase em volta: "AgInt no REsp 1.234/SP", e não "Colhe-se do AgInt no REsp 1.234/SP a mesma conclusão".
Os outros campos trazem a leitura corrigida (OCR: O->0, l->1, S->5, rn->m, c->e), ou null se não estiverem
na citação; não deduza tribunal pelo contexto.
Campos: "tipo" (processo, sumula, artigo, julgado, tema); "tribunal" (STF, STJ, TST, TSE, STM); "classe" (sigla:
REsp, AREsp, RE, ARE, Rcl, HC, RHC, RMS, MS, AR, AI, RO, REspe, APL, RSE, RR, AIRR, ARR, AgInt); "recursos"
(recursos antes da classe: AgInt, AgRg, EDcl, EDv, E, Ag); "numero" (só os dígitos, completo; CNJ tem 20
dígitos); "uf"; "ano" e "relator" (julgados sem número); "vinculante"; "diploma" (CF, CC, CPC, CDC, CPP, CLT, CE,
LC64/1990, CPM ou OUTRA); "lei" (número da lei, se o artigo é de lei citada pelo número); "artigo".

Exemplo:
Texto: Colhe-se do EDcl no AgRg no Rec. Esp. nº 1. 57O.531 – CE a mesma conclusão. Incide a Súm. 2l1 do STJ.
Nos termos do art 93, IX, da Constituição
Federal, a pretensão não prospera. Vale lembrar o precedente do TSE de 2016, da relatoria de Henrique Neves.
Autos nº 0001234-56.2020.8.26.0100. Ver ainda o RR - 1000-10.2015.5.O2.0001.
Resposta: {{"citacoes": [
{{"trecho": "EDcl no AgRg no Rec. Esp. nº 1. 57O.531 – CE", "tipo": "processo", "tribunal": null, "classe": "REsp", "recursos": ["AgRg", "EDcl"], "numero": "1570531", "uf": "CE", "ano": null, "relator": null, "diploma": null, "artigo": null}},
{{"trecho": "Súm. 2l1 do STJ", "tipo": "sumula", "tribunal": "STJ", "classe": null, "recursos": [], "numero": "211", "uf": null, "ano": null, "relator": null, "diploma": null, "artigo": null}},
{{"trecho": "art 93, IX, da Constituição\nFederal", "tipo": "artigo", "tribunal": null, "classe": null, "recursos": [], "numero": null, "uf": null, "ano": null, "relator": null, "diploma": "CF", "artigo": "93"}},
{{"trecho": "precedente do TSE de 2016, da relatoria de Henrique Neves", "tipo": "julgado", "tribunal": "TSE", "classe": null, "recursos": [], "numero": null, "uf": null, "ano": 2016, "relator": "Henrique Neves", "diploma": null, "artigo": null}},
{{"trecho": "RR - 1000-10.2015.5.O2.0001", "tipo": "processo", "tribunal": "TST", "classe": "RR", "recursos": [], "numero": "00010001020155020001", "uf": null, "ano": null, "relator": null, "diploma": null, "artigo": null}}
]}}

Texto:
<<<
{texto}
>>>
Resposta:"""

_NULLABLE_STR = {"type": ["string", "null"]}
SCHEMA = {
    "type": "object",
    "properties": {"citacoes": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "trecho": {"type": "string"},
            "tipo": {"type": "string", "enum": ["processo", "sumula", "artigo", "julgado", "tema"]},
            "tribunal": {"type": ["string", "null"], "enum": ["STF", "STJ", "TST", "TSE", "STM", None]},
            "classe": _NULLABLE_STR, "recursos": {"type": "array", "items": {"type": "string"}},
            "numero": _NULLABLE_STR, "uf": _NULLABLE_STR, "ano": {"type": ["integer", "null"]},
            "relator": _NULLABLE_STR, "vinculante": {"type": ["boolean", "null"]},
            "diploma": {"type": ["string", "null"],
                        "enum": ["CF", "CC", "CPC", "CDC", "CPP", "CLT", "CE", "LC64/1990", "CPM", "OUTRA", None]},
            "lei": _NULLABLE_STR, "artigo": _NULLABLE_STR,
        },
        "required": ["trecho", "tipo", "tribunal", "classe", "recursos", "numero", "uf", "ano", "relator",
                     "diploma", "artigo"],
    }}},
    "required": ["citacoes"],
}

FAMILY = {"processo": "processos", "sumula": "sumulas", "artigo": "artigos", "julgado": "julgados_descritivos",
          "tema": "temas"}
# Famílias equivalentes (as regras separam processo trabalhista e dois tipos de descritiva).
GROUP = {"processos": "processo", "processos_trabalhistas": "processo", "sumulas": "sumula", "artigos": "artigo",
         "processos_descritivos": "julgado", "julgados_descritivos": "julgado", "temas": "tema"}
DIPLOMA_TEXT = {"CF": "da Constituição Federal", "CC": "do Código Civil", "CPC": "do Código de Processo Civil",
                "CDC": "do Código de Defesa do Consumidor", "CPP": "do Código de Processo Penal",
                "CLT": "da CLT", "CE": "do Código Eleitoral", "LC64/1990": "da Lei Complementar nº 64/1990",
                "CPM": "do Código Penal Militar"}
CHAIN_TEXT = {"AGINT": "AgInt", "AGRG": "AgRg", "AGR": "AgRg", "EDCL": "EDcl", "ED": "EDcl", "EDV": "EDv",
              "E": "E", "AG": "Ag"}
TST_CLASSES = {"RR", "AIRR", "ARR"}
# Linhas de cabeçalho cujo número nunca é citação (PDF do desafio, "Distratores").
DISTRACTOR_LINE = re.compile(r"^\s*(?:autos|processo\s+de\s+origem|protocolo|oab|cpf|cnpj|fls?\b|valor)", re.I)


# ------------------------------------------------------------------ cliente

LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}


class OllamaClient:
    """Cliente mínimo da API nativa do Ollama (``/api/chat``), com cache opcional em disco.

    Só aceita servidor local (regra da competição: nenhuma chamada externa na execução).

    Parameters
    ----------
    url : str, default DEFAULT_URL
        Endereço do servidor; precisa ser local (127.0.0.1, localhost ou ::1).
    model : str, default DEFAULT_MODEL
        Nome do modelo no servidor.
    timeout_s : float, default 120.0
        Tempo máximo de cada pedido, em segundos.
    cache_dir : pathlib.Path, optional
        Pasta de cache das respostas (só para medições de laboratório); a chave é o pedido inteiro,
        sem o teto de tokens.
    num_ctx : int, default 8192
        Tamanho do contexto do modelo.
    num_predict : int, default 4096
        Teto de tokens gerados, contra laço de repetição.

    Raises
    ------
    ValueError
        Se o servidor não for local.
    """

    def __init__(self, url: str = DEFAULT_URL, model: str = DEFAULT_MODEL, timeout_s: float = 120.0,
                 cache_dir: Path | None = None, num_ctx: int = 8192, num_predict: int = 4096):
        if urlparse(url).hostname not in LOCAL_HOSTS:
            raise ValueError(f"servidor do modelo precisa ser local (127.0.0.1): {url}")
        self.url, self.model, self.timeout_s, self.cache_dir = url.rstrip("/"), model, timeout_s, cache_dir
        self.num_ctx, self.num_predict = num_ctx, num_predict
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def available(self) -> bool:
        """Diz se o servidor responde e tem o modelo.

        Returns
        -------
        bool
            ``True`` se ``/api/tags`` lista o modelo.
        """
        try:
            with urllib.request.urlopen(f"{self.url}/api/tags", timeout=5) as response:
                models = {m["name"] for m in json.loads(response.read())["models"]}
        except (urllib.error.URLError, OSError, ValueError, KeyError):
            return False
        return self.model in models or f"{self.model}:latest" in models

    def chat_json(self, system: str, user: str, schema: dict) -> dict | None:
        # Decodificação determinística: gulosa (temperature 0) e semente fixa; um pedido por vez no servidor.
        """Faz um pedido ao modelo com saída JSON restrita por schema.

        A decodificação é determinística: gulosa (temperatura 0), semente 42 e sem raciocínio
        ("think" desligado); o servidor atende um pedido por vez.

        Parameters
        ----------
        system : str
            Mensagem de sistema.
        user : str
            Mensagem do usuário (o prompt com o trecho).
        schema : dict
            JSON Schema da resposta (``format`` do Ollama).

        Returns
        -------
        dict or None
            A resposta decodificada; ``None`` se o servidor falhar ou a resposta não for um objeto JSON.
        """
        payload = {"model": self.model, "stream": False, "think": False, "format": schema,
                   "options": {"temperature": 0, "seed": 42, "num_ctx": self.num_ctx},
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        # Sem sort_keys: a ordem das propriedades do schema é a ordem em que o modelo escreve (trecho primeiro).
        key = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        payload["options"]["num_predict"] = self.num_predict  # teto contra laço de repetição (fora da chave)
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        cache = self.cache_dir / f"{hashlib.sha256(key).hexdigest()}.json" if self.cache_dir else None
        if cache and cache.exists():
            content = cache.read_text(encoding="utf-8")
        else:
            request = urllib.request.Request(f"{self.url}/api/chat", data=body,
                                             headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                    content = json.loads(response.read())["message"]["content"]
            except (urllib.error.URLError, OSError, ValueError, KeyError) as error:
                print(f"[nlp] falha no servidor: {error}", file=sys.stderr)
                return None
            if cache:
                cache.write_text(content, encoding="utf-8")
        try:
            data = json.loads(content)
        except ValueError:
            return None
        return data if isinstance(data, dict) else None


# ------------------------------------------------------------------ alinhamento tolerante a OCR

_SKELETON = {"o": "0", "q": "6", "g": "6", "9": "6", "l": "1", "i": "1", "|": "1", "j": "1", "s": "5", "b": "8",
             "z": "2", "c": "e", "u": "n"}


def skeleton(text: str) -> tuple[str, list[int]]:
    """Reduz o texto a um esqueleto para comparação tolerante a OCR.

    Só letras e dígitos, sem acento, em minúsculas e com as confusões de OCR unificadas
    (o/0, l/i/1, s/5, rn -> m, ri -> n...).

    Parameters
    ----------
    text : str
        Texto de entrada.

    Returns
    -------
    sk : str
        O esqueleto.
    index : list of int
        Para cada caractere do esqueleto, o índice no texto original.
    """
    out, index = [], []
    i, n = 0, len(text)
    while i < n:
        base = unicodedata.normalize("NFD", text[i])[0].lower()
        nxt = unicodedata.normalize("NFD", text[i + 1])[0].lower() if i + 1 < n else ""
        if base == "r" and nxt in ("n", "i"):  # "rn" lido por "m"; "ri" lido por "n" ("Viriculante")
            out.append("m" if nxt == "n" else "n")
            index.append(i)
            i += 2
            continue
        if base.isalnum() or base == "|":
            out.append(_SKELETON.get(base, base))
            index.append(i)
        i += 1
    return "".join(out), index


def align(text_sk: str, text_index: list[int], trecho: str, min_ratio: float = 0.86) -> list[tuple[int, int]]:
    """Localiza no documento todas as ocorrências do trecho lido pelo modelo.

    Compara pelos esqueletos (:func:`skeleton`); sem ocorrência exata, procura a mais parecida a
    partir de âncoras no começo ou no fim do trecho.

    Parameters
    ----------
    text_sk : str
        Esqueleto do documento.
    text_index : list of int
        Índices do esqueleto no texto original.
    trecho : str
        Trecho copiado pelo modelo.
    min_ratio : float, default 0.86
        Semelhança mínima para aceitar uma ocorrência aproximada.

    Returns
    -------
    list of tuple of (int, int)
        Spans (início, fim) no texto original; vazia se o trecho não estiver no documento (o que o
        modelo inventar é descartado aqui).
    """
    t_sk, _ = skeleton(trecho)
    if len(t_sk) < 5:
        return []
    spans, pos = [], text_sk.find(t_sk)
    while pos >= 0:
        spans.append((text_index[pos], text_index[pos + len(t_sk) - 1] + 1))
        pos = text_sk.find(t_sk, pos + 1)
    if spans:
        return spans
    # Aproximado: âncora no começo ou no fim do trecho e razão de semelhança no entorno.
    best = None
    k = min(5, len(t_sk) // 2)
    starts = {m.start() for m in re.finditer(re.escape(t_sk[:k]), text_sk)}
    starts |= {m.end() - len(t_sk) for m in re.finditer(re.escape(t_sk[-k:]), text_sk)}
    for start in starts:
        for length in range(max(5, len(t_sk) - 3), len(t_sk) + 4):
            s, e = max(0, start), min(len(text_sk), start + length)
            if e - s < 5:
                continue
            ratio = difflib.SequenceMatcher(None, t_sk, text_sk[s:e], autojunk=False).ratio()
            if ratio >= min_ratio and (best is None or ratio > best[0]):
                best = (ratio, s, e)
    if best is None:
        return []
    return [(text_index[best[1]], text_index[best[2] - 1] + 1)]


# ------------------------------------------------------------------ guardas

_LETTER_NUMBER = re.compile(r"(?<![\w.\-/])[OoQDlIi|SsgqGbBZzL]{2,}(?![\w.\-/])")  # sem dígito real


def digits_in_span(digits: str, span: str) -> bool:
    """Confere se o número lido pelo modelo é um dos números do trecho.

    O número precisa estar inteiro no trecho, só com troca letra -> dígito parecido (nunca dígito
    por dígito, nem sobrando dígito antes ou depois). Um número todo em letras parecidas
    ("Súmula Vinculante lBB" = 188) também vale, se for uma palavra isolada.

    Parameters
    ----------
    digits : str
        Dígitos lidos pelo modelo.
    span : str
        Trecho da citação.

    Returns
    -------
    bool
        ``True`` se o número está no trecho.
    """
    if not digits:
        return False
    wanted = {digits, digits.lstrip("0") or "0"}
    if any(g in wanted or (g.lstrip("0") or "0") in wanted for g in number_groups(span)):
        return True
    # Número inteiro em letras parecidas ("Súmula Vinculante lBB" = 188): palavra isolada, idêntica.
    return any(t.translate(OCR_DIGITS) in wanted for t in _LETTER_NUMBER.findall(span))


def _name_in_span(relator: str, span: str) -> bool:
    """Confere se cada token do nome lido pelo modelo aparece no trecho, tolerando OCR.

    Parameters
    ----------
    relator : str
        Nome lido pelo modelo.
    span : str
        Trecho da citação.

    Returns
    -------
    bool
        ``True`` se todos os tokens do nome aparecem.
    """
    tokens = name_tokens(relator)
    if not tokens:
        return False
    span = clean(_nfc(span)[0])[0]  # "Gil-\nmar" -> "Gilmar"; acento decomposto, invisíveis
    span_tokens = [skeleton(t)[0] for t in re.findall(r"[^\W\d_]+", span)]
    for token in tokens:
        sk = skeleton(token)[0]
        if not any(sk == s or difflib.SequenceMatcher(None, sk, s).ratio() >= 0.8 for s in span_tokens):
            return False
    return True


_COURT_WORDS = re.compile(r"tribunal|corte|pret[oó]rio|supremo|superior|excels|cidad|castrense|eleitoral|trabalh",
                          re.IGNORECASE)


# Evidência da classe no trecho: siglas curtas como palavra; nomes por extenso tolerando OCR.
CLASS_EVIDENCE = {
    "REsp": ["resp", "especial"], "AREsp": ["aresp", "agravo em recurso especial", "agravo em resp"],
    "RE": ["re", "extraordinario"], "ARE": ["are", "extraordinario com agravo"], "Rcl": ["rcl", "recl", "reclama"],
    "HC": ["hc", "habeas"], "RHC": ["rhc", "recurso em habeas", "ordinario em habeas"],
    "RMS": ["rms", "mandado de seguranca"], "MS": ["ms", "mandado de seguranca"], "AR": ["ar", "rescisoria"],
    "AI": ["ai", "agravo de instrumento"], "RO": ["ro", "recurso ordinario"], "REspe": ["respe", "respel",
                                                                                      "especial eleitoral"],
    "APL": ["apl", "apel", "ap", "apelacao"], "RSE": ["rse", "sentido estrito"], "AgInt": ["agint", "agravo interno"],
    "RR": ["rr", "recurso de revista"], "AIRR": ["airr"], "ARR": ["arr"],
}


def _has_evidence(stems: list[str], span: str) -> bool:
    """Diz se algum radical aparece no trecho.

    Siglas curtas contam como palavra inteira; nomes, como substring. As comparações usam o
    esqueleto: "RE5P" é "resp", "Rc1" é "rcl".

    Parameters
    ----------
    stems : list of str
        Radicais procurados.
    span : str
        Trecho da citação.

    Returns
    -------
    bool
        ``True`` se há evidência de algum radical.
    """
    return _evidence_length(stems, span) > 0


def _evidence_length(stems: list[str], span: str) -> int:
    """Mede o maior radical com evidência no trecho.

    Parameters
    ----------
    stems : list of str
        Radicais procurados.
    span : str
        Trecho da citação.

    Returns
    -------
    int
        Tamanho (no esqueleto) do maior radical encontrado; 0 se nenhum.
    """
    words = {skeleton(w)[0] for w in re.split(r"[^\w.]+", span) if w}
    words |= {skeleton(w.replace(".", ""))[0] for w in re.split(r"\s+", span) if "." in w}  # "R.Esp." -> resp
    span_sk = skeleton(span)[0]
    best = 0
    for stem in stems:
        stem_sk = skeleton(stem)[0]
        if len(stem) <= 4:
            # Sigla: palavra inteira; com 3+ letras, tolera uma troca/inversão de OCR ("Rcel." ~ "recl").
            # Com 4 letras, tolera uma troca/inversão de OCR sem mudar o tamanho ("Rcel." ~ "recl"; "apelo" ≠ "apel").
            if stem_sk in words or (len(stem_sk) == 4 and any(
                    len(w) == 4 and w[0] == stem_sk[0] and difflib.SequenceMatcher(None, stem_sk, w).ratio() >= 0.75
                    for w in words)):
                best = max(best, len(stem_sk))
        elif stem_sk in span_sk or (len(stem_sk) >= 9 and _fuzzy_in(stem_sk, span_sk)):
            best = max(best, len(stem_sk))
    return best


# Palavras que podem vir logo antes do número de um processo (classe, recurso, tribunal), além dos radicais
# de CLASS_EVIDENCE; marcadores e conectivos entre elas e o número são ignorados.
_CITATION_WORDS = ["agravo", "agint", "agrg", "agr", "edcl", "eds", "embargos", "declaracao", "divergencia", "interno",
                   "regimental", "recurso", "corpus", "especial", "extraordinario", "eleitoral", "revista",
                   "instrumento", "seguranca", "estrito", "ordinario", "rescisoria", "reclamacao", "apelacao",
                   "criminal", "liminar", "sentenca", "processo", "proc", "feito", "stf", "stj", "tst", "tse", "stm"]
_MARKERS = {"n", "no", "nº", "n°", "nr", "nro", "num", "numero", "número", "de", "do", "da", "dos", "das", "na",
            "nos", "nas", "em", "e"}


# Palavras que, logo antes de um número, dizem que ele não é de processo.
_BLOCKERS = {"ordem", "resolucao", "portaria", "ato", "instrucao", "provimento", "decreto", "pauta", "folha",
             "folhas", "fls", "fl", "pagina", "paginas", "protocolo", "autos", "sessao", "edital", "oficio", "item",
             "lei", "emenda", "rs", "reais", "valor", "cpf", "cnpj", "oab", "cep", "matricula", "inscricao"}


def _number_after_citation_words(span: str, digits: str) -> bool:
    """Confere se o número vem depois de classe, recurso ou tribunal, e não de outra coisa.

    Volta pelas palavras antes do número: palavra de bloqueio ("Resolução", "ordem", "fls.")
    recusa; vocabulário de citação ("REsp", "Reclamação Constitucional") aceita; até 3 palavras
    desconhecidas ("Cível") no caminho. Um CNJ completo se identifica sozinho.

    Parameters
    ----------
    span : str
        Trecho da citação.
    digits : str
        Número lido pelo modelo.

    Returns
    -------
    bool
        ``True`` se o número é de um processo citado.
    """
    groups = [g for g in number_group_spans(span) if not (len(g[0]) == 4 and g[0][:2] in ("19", "20"))]
    if any(len(g[0]) >= 18 for g in groups):
        return True
    target = [g for g in groups if digits and g[0].lstrip("0") == digits.lstrip("0")] or (groups if len(groups) == 1 else [])
    if not target:
        return False
    stems = _CITATION_WORDS + [stem for stems in CLASS_EVIDENCE.values() for stem in stems]
    unknown = 0
    for token in reversed(re.findall(r"[^\W\d_]+", span[:target[0][1]])):
        folded = fold(token)
        if folded in _MARKERS:
            continue
        if folded in _BLOCKERS:
            return False
        token_sk = skeleton(token)[0]
        if _evidence_length(stems, token) > 0 or any(len(stem) >= 4 and token_sk.endswith(skeleton(stem)[0])
                                                      for stem in stems):  # "uoREsp"
            return True
        unknown += 1
        if unknown > 3:
            return False
    return False


def _best_evidence_diploma(span: str) -> str | None:
    """Identifica o único diploma da base com evidência no trecho.

    Parameters
    ----------
    span : str
        Trecho da citação.

    Returns
    -------
    str or None
        Chave do diploma, se houver exatamente um com evidência.
    """
    found = [d for d in DIPLOMA_TEXT if _diploma_evidence(d, span)]
    return found[0] if len(found) == 1 else None


def _best_evidence_class(span: str) -> str | None:
    """Identifica a classe com o radical mais longo presente no trecho.

    "recurso em habeas" vence "habeas"; empate entre classes diferentes não decide.

    Parameters
    ----------
    span : str
        Trecho da citação.

    Returns
    -------
    str or None
        Sigla da classe; ``None`` sem evidência ou em empate.
    """
    scored = sorted(((_evidence_length(stems, span), cls) for cls, stems in CLASS_EVIDENCE.items()), reverse=True)
    if not scored or scored[0][0] == 0 or (len(scored) > 1 and scored[1][0] == scored[0][0]):
        return None
    return scored[0][1]


def _fuzzy_in(needle: str, hay: str, cutoff: float = 0.8) -> bool:
    """Procura um nome longo com OCR pesado no texto ("Reccurso em Hqbcas" ~ "recurso em habeas").

    Parameters
    ----------
    needle : str
        Esqueleto do nome procurado.
    hay : str
        Esqueleto do texto.
    cutoff : float, default 0.8
        Semelhança mínima.

    Returns
    -------
    bool
        ``True`` se alguma janela do texto é parecida o bastante.
    """
    n = len(needle)
    for start in range(0, max(1, len(hay) - n + 3)):
        for size in (n - 2, n, n + 2):
            if 0 < size <= len(hay) - start and difflib.SequenceMatcher(None, needle, hay[start:start + size],
                                                                       autojunk=False).ratio() >= cutoff:
                return True
    return False


def _class_evidence(classe: str | None, span: str) -> bool:
    """Diz se a classe tem evidência no trecho.

    Parameters
    ----------
    classe : str or None
        Sigla da classe.
    span : str
        Trecho da citação.

    Returns
    -------
    bool
        ``True`` se algum radical da classe aparece.
    """
    return _has_evidence(CLASS_EVIDENCE.get(classe or "", []), span)


# Evidência do diploma no trecho (inclui apelidos e o número da lei que o instituiu).
DIPLOMA_EVIDENCE = {
    "CF": ["constituicao", "cf", "crfb", "carta", "lei maior"], "CC": ["codigo civil", "cc", "10406"],
    "CPC": ["processo civil", "cpc", "13105"], "CDC": ["consumidor", "cdc", "8078"],
    "CPP": ["processo penal", "cpp", "3689"], "CLT": ["clt", "consolidacao", "5452", "diploma consolidado"],
    "CE": ["codigo eleitoral", "4737"], "LC64/1990": ["complementar", "lc", "inelegibilidade"],
    "CPM": ["penal militar", "cpm", "1001"],
}


def _diploma_evidence(diploma: str | None, span: str) -> bool:
    """Diz se o diploma tem evidência no trecho (nome, sigla, apelido ou número da lei).

    Parameters
    ----------
    diploma : str or None
        Chave do diploma.
    span : str
        Trecho da citação.

    Returns
    -------
    bool
        ``True`` se algum radical do diploma aparece.
    """
    return _has_evidence(DIPLOMA_EVIDENCE.get(diploma or "", []), span)


_OCR_NUM = r"[\dOolI|SsgqGbBZz]"
_LAW_IN_SPAN = re.compile(rf"L[eco][il1](?![a-z])\D{{0,25}}?({_OCR_NUM}(?:[\s.\-\u2060]*{_OCR_NUM})*)\s*/\s*({_OCR_NUM}{{2,4}})",
                          re.IGNORECASE)


_GLUED_COURT = re.compile(r"(?<![A-Z0-9])((?:[S5]T[FJM])|(?:T[S5][TE]))(?![A-Z0-9])")


def _is_year(group: str) -> bool:
    """Diz se o grupo de dígitos é um ano (19xx ou 20xx).

    Parameters
    ----------
    group : str
        Dígitos.

    Returns
    -------
    bool
        ``True`` para anos de 4 dígitos.
    """
    return len(group) == 4 and group[:2] in ("19", "20")


_TEMA_CUE = re.compile(r"(?<![^\W\d_])t[eéc]m[aã]s?(?![^\W\d_])", re.IGNORECASE)


_SUMULA_SK = "5nmn1a"  # esqueleto de "súmula"


def _sumula_cue_end(plain: str) -> int | None:
    r"""Localiza o fim da pista de súmula no trecho.

    Aceita "Súmula", "Súm.", "verbete", "enunciado", "SV" e a palavra "súmula" corrompida por OCR
    ("Súmuula", "Súmla", "5umulla", "Suu\nmla", "aSúm.").

    Parameters
    ----------
    plain : str
        Trecho já limpo.

    Returns
    -------
    int or None
        Posição logo depois da pista; ``None`` se não houver.
    """
    if cue := _SUMULA_CUE.search(plain) or re.search(r"(?-i:\bSV\b)", plain):
        return cue.end()
    words = list(re.finditer(r"[^\W\d_]+", plain))
    for i, w in enumerate(words):
        candidates = [(w.group(), w.end())]
        if i + 1 < len(words) and re.fullmatch(r"[\s\-\u00ad]*", plain[w.end():words[i + 1].start()]):
            candidates.append((w.group() + words[i + 1].group(), words[i + 1].end()))  # "Suu\nmla"
        for text, end in candidates:
            sk = skeleton(text)[0]
            if difflib.SequenceMatcher(None, sk, _SUMULA_SK).ratio() >= 0.75 and len(sk) <= 9:
                return end
            if sk.endswith("5nm") and plain[end:end + 1] == ".":  # "aSúm."
                return end + 1
    return None


def _tribunal(item: dict, span: str) -> str | None:
    """Decide o tribunal da citação.

    Vale o tribunal escrito no trecho (sigla ou extenso, também colado: "doSTJ"); o do modelo só
    vale se o trecho fala de um tribunal por apelido ("Corte Cidadã").

    Parameters
    ----------
    item : dict
        Citação lida pelo modelo.
    span : str
        Trecho (ou trecho com entorno curto).

    Returns
    -------
    str or None
        Sigla do tribunal.
    """
    courts = courts_in(span)
    if courts:
        return courts[0]
    if glued := _GLUED_COURT.search(span):  # "doSTJ", "noTSE"
        return glued.group(1).replace("5", "S")
    return item.get("tribunal") if _COURT_WORDS.search(span) else None


def _digits(value) -> str:
    """Extrai os dígitos de um campo do modelo, convertendo letras parecidas ("1BB" -> 188).

    Parameters
    ----------
    value : object
        Campo lido pelo modelo (texto, número ou ``None``).

    Returns
    -------
    str
        Só os dígitos.
    """
    return re.sub(r"\D", "", str(value or "").translate(OCR_DIGITS))


def _fmt_number(digits: str) -> str:
    """Formata um número de processo para a forma canônica.

    Parameters
    ----------
    digits : str
        Dígitos do número.

    Returns
    -------
    str
        CNJ ``NNNNNNN-DD.AAAA.J.TR.OOOO`` (20 dígitos) ou número curto com ponto de milhar.
    """
    if len(digits) == 20:
        return f"{digits[:7]}-{digits[7:9]}.{digits[9:13]}.{digits[13]}.{digits[14:16]}.{digits[16:]}"
    return f"{int(digits):,}".replace(",", ".")


def canonical(item: dict, span: str, context: str | None = None) -> tuple[str, list[str]] | None:
    """Monta a forma canônica da citação com os campos do modelo, depois das guardas.

    Parameters
    ----------
    item : dict
        Citação lida pelo modelo (``tipo``, ``numero``, ``classe``, ``tribunal``...).
    span : str
        Trecho da citação no documento; os dígitos são conferidos só nele.
    context : str, optional
        O trecho com um entorno curto; serve de evidência de tribunal, classe e diploma.

    Returns
    -------
    tuple of (str, list of str) or None
        A família e as formas canônicas; ``None`` se as guardas recusarem.

    Notes
    -----
    Há mais de uma forma quando modelo e trecho divergem (ex.: na classe); quem chama só aceita se
    todas levarem à mesma resposta (:func:`resolve_forms`). Se o número lido não estiver no
    trecho, vale o único número candidato do próprio trecho.
    """
    tipo = item.get("tipo")
    evidence = context or span
    if tipo == "tema":
        # "Tema 1.046 da repercussão geral": a palavra "tema" e o número logo depois dela, no próprio trecho
        # (o modelo às vezes chama de tema uma lei ou resolução: "Lei nº 8.112/90", "Resolução nº 639").
        digits = _digits(item.get("numero"))
        plain = clean(_nfc(span)[0])[0]
        cue = _TEMA_CUE.search(plain)
        if not digits or not cue or not digits_in_span(digits, plain[cue.end():cue.end() + 50]):
            return None
        return "temas", ["Tema"]
    if tipo == "processo":
        digits = _digits(item.get("numero"))
        if 14 <= len(digits) < 20:
            digits = digits.zfill(20)
        if not digits_in_span(digits, span):
            # O modelo leu mal o número (só o começo de um CNJ, um zero a mais...). Os dígitos passam a
            # vir do próprio trecho, e só se houver um único candidato: CNJ (14 a 20 dígitos) ou número curto.
            groups = [g for g in number_groups(span) if not _is_year(g)]
            cnj = [g.zfill(20) for g in groups if 14 <= len(g) <= 20]
            short = [g for g in groups if 3 <= len(g) <= 8]
            digits = cnj[0] if len(cnj) == 1 else short[0] if not cnj and len(short) == 1 else ""
        if len(digits) < 3:
            return None
        tribunal = _tribunal(item, evidence)
        # Classe: vale a leitura que tem evidência no trecho (do modelo ou das regras); sem evidência
        # para nenhuma, as duas precisam levar ao mesmo registro.
        readings = list(dict.fromkeys(c for c in (item.get("classe"), class_code(evidence)) if c))
        classes = [c for c in readings if _class_evidence(c, evidence)]
        if not classes and (by_evidence := _best_evidence_class(evidence)):
            classes = [by_evidence]  # nem o modelo nem as regras leram a classe que o trecho mostra
        classes = classes or readings or [None]
        if tribunal == "TST" or set(classes) & TST_CLASSES:
            if len(digits) != 20:
                return None
            classe = next((c for c in classes if c in TST_CLASSES), "RR")
            return "processos_trabalhistas", [f"TST-{classe}-{_fmt_number(digits)}"]
        chain = " no ".join(CHAIN_TEXT[r.upper().replace(".", "")] for r in reversed(item.get("recursos") or [])
                            if r and r.upper().replace(".", "") in CHAIN_TEXT)
        uf = (item.get("uf") or "").upper()
        uf = uf if uf in UFS else ""
        number = f"nº {_fmt_number(digits)}" + (f"/{uf}" if len(uf) == 2 else "")
        forms = [" ".join(p for p in (f"{chain} no" if chain else "", classe or "", number) if p)
                 + (f" ({tribunal})" if tribunal else "") for classe in classes]
        return "processos", forms
    if tipo == "sumula":
        # Pista de súmula no trecho e o número logo depois dela (o modelo às vezes chama de súmula uma
        # "Resolução nº 866").
        digits = _digits(item.get("numero"))
        plain = clean(_nfc(span)[0])[0]  # "5Ãºmula" (mojibake) e acento decomposto viram "5úmula"/"Súmula"
        cue_end = _sumula_cue_end(plain)
        if cue_end is None:
            return None
        window = plain[cue_end:cue_end + 50]
        if not digits_in_span(digits, window):
            # O modelo leu mal o número ("q79"): vale o primeiro número depois da pista, lido do texto.
            groups = [g for g in number_groups(window) if len(g) <= 4]
            digits = groups[0] if groups else ""
        if not digits:
            return None
        if item.get("vinculante") or re.search(r"(?-i:\bSV\b)", evidence) or _has_evidence(["vinculante"], evidence):
            return "sumulas", [f"Súmula Vinculante {int(digits)}"]
        courts = courts_in(evidence)
        tribunal = courts[-1] if courts else _tribunal(item, evidence)
        return ("sumulas", [f"Súmula {int(digits)} do {tribunal}"]) if tribunal else None
    if tipo == "artigo":
        digits = _digits(item.get("artigo"))
        unordinal = re.sub(r"(?<=\d)[oO°º](?![^\W\d_])", "", span)  # "7o" é "7º", não 70
        one = digits == "1" and re.search(r"\bart(?:igo)?\.?\s*[lI|](?![\w|])", span, re.I)  # "art. l" = art. 1
        if not digits or not (digits_in_span(digits, span) or digits_in_span(digits, unordinal) or one):
            return None
        llm_diploma = item.get("diploma") if _diploma_evidence(item.get("diploma"), evidence) else None
        diplomas = list(dict.fromkeys(d for d in (llm_diploma, diploma_key(evidence)) if d in DIPLOMA_TEXT))
        if not diplomas and (by_evidence := _best_evidence_diploma(evidence)):
            diplomas = [by_evidence]  # "diploma consolidado" -> CLT
        if diplomas:
            return "artigos", [f"art. {int(digits)} {DIPLOMA_TEXT[d]}" for d in diplomas]
        # Lei citada pelo número: o número vem do trecho (com OCR), não do modelo, que confunde as leis.
        if m := _LAW_IN_SPAN.search(evidence):
            number, year = m.group(1).translate(OCR_DIGITS), m.group(2).translate(OCR_DIGITS)
            number, year = re.sub(r"\D", "", number), re.sub(r"\D", "", year)
            if number and year:
                return "artigos", [f"art. {int(digits)} da Lei nº {_fmt_number(number)}/{year}"]
        # Lei por apelido ("Lei do Inquilinato"): vale o número que o modelo conhece, mas se ele cair num diploma
        # da base, o trecho tem de mostrar esse diploma (o modelo não transforma inventada em real).
        lei = str(item.get("lei") or "")
        if re.search(r"\d", lei) and re.search(r"\b(?:lei|estatuto|c[oó]digo|diploma)\b", evidence, re.I):
            key = diploma_key(f"Lei nº {lei}")
            if key and (key not in DIPLOMA_TEXT or _diploma_evidence(key, evidence)):
                return "artigos", [f"art. {int(digits)} da Lei nº {lei}"]
        return None
    if tipo == "julgado":
        ano, relator = item.get("ano"), item.get("relator")
        if not ano or not relator or not digits_in_span(str(ano), span) or not _name_in_span(relator, span):
            return None
        tribunal = _tribunal(item, evidence)
        classe = class_code(span.split(str(ano))[0])
        if not tribunal:
            return None
        head = f"{classe} do {tribunal}, de {ano}, Rel. Min. {relator}" if classe else \
            f"julgado do {tribunal} proferido em {ano} pela relatoria de {relator}"
        return "julgados_descritivos", [head]
    return None


def resolve_forms(family: str, forms: list[str], citation: dict, index: CanonicalIndex) -> Resolution | None:
    """Consulta a base com cada forma canônica e só responde se todas concordarem.

    Parameters
    ----------
    family : str
        Família da citação.
    forms : list of str
        Formas canônicas de :func:`canonical`.
    citation : dict
        Citação original (dela vêm ``contexto`` e ``contexto_depois``).
    index : CanonicalIndex
        Índice da base.

    Returns
    -------
    Resolution or None
        O resultado comum; ``None`` se as formas levarem a classes ou registros diferentes.
    """
    results = [resolve({"familia": family, "trecho": text, "contexto": citation.get("contexto", ""),
                        "contexto_depois": citation.get("contexto_depois", "")}, index) for text in forms]
    if len({(r.classificacao, r.id_canonico) for r in results}) == 1:
        return results[0]
    return None


# ------------------------------------------------------------------ camada

class NLPLayer:
    """Revisão das regras pelos modelos, dentro de um orçamento de tempo.

    Parameters
    ----------
    client : OllamaClient
        Modelo principal.
    normalize : bool, default True
        Renormaliza citações que as regras deixaram ``inventada`` ou ``incompleta``.
    recall : bool, default True
        Acrescenta citações que só os modelos viram.
    chunk_chars : int, default 3000
        Tamanho máximo de cada trecho enviado ao modelo.
    workers : int, default 1
        Pedidos em paralelo por documento.
    budget_doc_s : float, default 40.0
        Média máxima de segundos por documento.
    budget_total_s : float, default 10800.0
        Tempo total máximo da camada, em segundos (3 h).
    warmup_docs : int, default 3
        Documentos antes de conferir a média.
    extra_clients : tuple of OllamaClient, optional
        Modelos do conjunto: união para citações novas; normalização só se as leituras não levarem
        a registros diferentes.

    Notes
    -----
    O envelope oficial foi de média <= 60 s por documento e 4 h no total. Se a média por documento
    (sem o primeiro, que inclui carregar os modelos na GPU) passar de ``budget_doc_s`` depois de
    ``warmup_docs`` documentos, ou o total passar de ``budget_total_s``, a camada se desliga e o
    restante sai só com as regras (ex.: contêiner sem GPU, modelo em CPU).
    """

    def __init__(self, client: OllamaClient, *, normalize: bool = True, recall: bool = True,
                 chunk_chars: int = 3000, workers: int = 1, budget_doc_s: float = 40.0,
                 budget_total_s: float = 3 * 3600.0, warmup_docs: int = 3,
                 extra_clients: tuple[OllamaClient, ...] = ()):
        self.client, self.normalize, self.recall = client, normalize, recall
        self.chunk_chars, self.workers = chunk_chars, workers
        # Modelos extras: conjunto (união para citações novas; normalização só se todos os modelos
        # levarem ao mesmo registro).
        self.extra_clients = tuple(extra_clients)
        self.budget_doc_s, self.budget_total_s, self.warmup_docs = budget_doc_s, budget_total_s, warmup_docs
        self.spent_s, self.docs, self.disabled, self.first_s = 0.0, 0, False, 0.0

    def _charge(self, seconds: float) -> None:
        """Contabiliza o tempo de um documento e desliga a camada se o orçamento estourar.

        Parameters
        ----------
        seconds : float
            Tempo gasto pela camada no documento.
        """
        self.spent_s += seconds
        self.docs += 1
        if self.docs == 1:
            # O 1º documento inclui carregar os modelos na GPU (partida a frio, 100 s ou mais num disco
            # lento): conta para o total, mas fica fora da média, senão desliga a camada no começo.
            self.first_s = seconds
        average = (self.spent_s - self.first_s) / (self.docs - 1) if self.docs > 1 else 0.0
        if (self.docs > self.warmup_docs and average > self.budget_doc_s) or self.spent_s > self.budget_total_s:
            self.disabled = True
            print(f"[nlp] orçamento de tempo excedido ({self.docs} docs, média {average:.1f} s, total "
                  f"{self.spent_s:.0f} s): o restante sai só com as regras", file=sys.stderr)

    # -- chamadas ao modelo

    def chunks(self, content: str) -> list[tuple[int, str]]:
        """Corta o documento em trechos de até ``chunk_chars`` caracteres, em fim de parágrafo.

        Parameters
        ----------
        content : str
            Texto do documento.

        Returns
        -------
        list of tuple of (int, str)
            Offset de início e texto de cada trecho.
        """
        out, start = [], 0
        while start < len(content):
            end = min(len(content), start + self.chunk_chars)
            if end < len(content):
                cut = content.rfind("\n\n", start + self.chunk_chars // 2, end)
                cut = cut if cut > 0 else content.rfind("\n", start + self.chunk_chars // 2, end)
                end = cut + 1 if cut > 0 else end
            out.append((start, content[start:end]))
            start = end
        return out

    def read(self, content: str, client: OllamaClient | None = None) -> list[tuple[int, dict]]:
        """Pede a um modelo as citações de todos os trechos do documento.

        Parameters
        ----------
        content : str
            Texto do documento.
        client : OllamaClient, optional
            Modelo a usar; padrão: o principal.

        Returns
        -------
        list of tuple of (int, dict)
            Offset do trecho e citação lida, para cada citação válida (com ``trecho`` em texto).
        """
        items = []
        chunks = self.chunks(content)
        client = client or self.client

        def call(chunk: tuple[int, str]):
            """Pede as citações de um trecho.

            Parameters
            ----------
            chunk : tuple of (int, str)
                Offset e texto do trecho.

            Returns
            -------
            tuple of (int, list)
                O offset e as citações da resposta (vazia se o modelo falhar).
            """
            data = client.chat_json(SYSTEM, PROMPT.format(texto=chunk[1]), SCHEMA)
            return chunk[0], (data or {}).get("citacoes") or []

        if self.workers > 1 and len(chunks) > 1:
            with ThreadPoolExecutor(self.workers) as pool:
                results = list(pool.map(call, chunks))
        else:
            results = [call(c) for c in chunks]
        for offset, cites in results:
            items += [(offset, c) for c in cites if isinstance(c, dict) and isinstance(c.get("trecho"), str)]
        return items

    # -- revisão

    def refine(self, content: str, citations: list[dict], index: CanonicalIndex) -> list[dict]:
        """Revisa as citações das regras e acrescenta as que só os modelos viram.

        Parameters
        ----------
        content : str
            Texto do documento.
        citations : list of dict
            Citações das regras (``inicio``, ``fim``, ``trecho``, ``familia``, ``resolution``).
        index : CanonicalIndex
            Índice da base.

        Returns
        -------
        list of dict
            As citações revisadas e as novas, em ordem de posição. Sem mudança se a camada estiver
            desligada pelo orçamento.
        """
        if self.disabled:
            return citations
        clock = time.monotonic()
        items = [it for client in (self.client, *self.extra_clients) for it in self.read(content, client)]
        state = _DocState(content)
        added: list[dict] = []
        self._absorb(items, citations, added, state, index)
        self._charge(time.monotonic() - clock)
        for c in citations + added:
            if c.pop("_conflito", False):
                c["resolution"] = c.pop("_original")  # modelos levaram a registros diferentes: fica o das regras
            c.pop("_original", None)
            c.pop("_propostas", None)
        return sorted(citations + added, key=lambda c: c["inicio"])

    def _absorb(self, items: list[tuple[int, dict]], citations: list[dict], added: list[dict], state: "_DocState",
                index: CanonicalIndex) -> None:
        """Incorpora as leituras dos modelos às citações do documento.

        Cada citação lida é localizada no texto; se coincidir com uma das regras (sobreposição de ao
        menos 50% da menor), renormaliza-a; senão, depois de aparar o span e passar pelos filtros de
        precisão, entra como citação nova.

        Parameters
        ----------
        items : list of tuple of (int, dict)
            Leituras de :meth:`read` (de todos os modelos).
        citations : list of dict
            Citações das regras (alteradas no lugar).
        added : list of dict
            Citações novas (acrescentadas no lugar).
        state : _DocState
            O que a revisão do documento reaproveita.
        index : CanonicalIndex
            Índice da base.
        """
        content = state.content
        for _, item in items:
            # Todas as ocorrências no documento: a mesma citação repetida é anotada em cada lugar.
            for start, end in align(state.text_sk, state.text_index, item["trecho"]):
                # Mesma citação só com sobreposição substancial: um span longo das regras que só encosta
                # na citação do modelo ("… STF" + "STF, 2026, EDSON FACHIN") não a esconde.
                overlap = [c for c in citations + added if _same_citation((start, end), (c["inicio"], c["fim"]))]
                if overlap:
                    if self.normalize:
                        for c in overlap:
                            self._shrink_overlong(c, start, end, content, index)
                            # O span das regras pode vir truncado ("… do Codigo Penal" sem o "Militar"): tribunal,
                            # classe e diploma podem vir do entorno que o modelo alinhou, se for curto (o trecho do
                            # modelo pode juntar duas citações: "Súmula 3 e Súmula 211"). Dígitos, nunca.
                            union = content[min(start, c["inicio"]):max(end, c["fim"])]
                            compact = len(union) - (c["fim"] - c["inicio"]) <= 40
                            self._renormalize(c, item, index, union if compact else None)
                    continue
                if not self.recall:
                    continue
                if item.get("tipo") == "processo":
                    start, end = trim_process(content, start, end, _digits(item.get("numero")))
                elif item.get("tipo") == "artigo":
                    start, end = trim_article(content, start, end, _digits(item.get("artigo")))
                elif item.get("tipo") == "sumula":
                    start, end = trim_sumula(content, start, end, _digits(item.get("numero")))
                if any(_same_citation((start, end), (c["inicio"], c["fim"])) for c in citations + added):
                    continue  # depois de aparada, é uma citação que já existe
                if not self._acceptable(content, start, end, item, state.header_end, state.own_numbers):
                    continue
                new = self._classify_new(content, start, end, item, index)
                if new:
                    added.append(new)

    def _shrink_overlong(self, citation: dict, start: int, end: int, content: str, index: CanonicalIndex) -> None:
        """Troca um span das regras longo demais pelo span do modelo contido nele.

        Sob ruído extremo, as regras às vezes capturam um parágrafo inteiro. Se o span tiver mais de
        150 caracteres e mais de 3 vezes o do modelo, passa a ser o do modelo e é resolvido de novo
        pelas regras (não pela leitura do modelo).

        Parameters
        ----------
        citation : dict
            Citação das regras (alterada no lugar).
        start, end : int
            Span do modelo no texto.
        content : str
            Texto do documento.
        index : CanonicalIndex
            Índice da base.
        """
        length = citation["fim"] - citation["inicio"]
        if not (length > 150 and length > 3 * (end - start) and citation["inicio"] <= start and end <= citation["fim"]):
            return
        citation.update(inicio=start, fim=end, trecho=content[start:end])
        try:
            resolution = resolve(citation | {"trecho_norm": " ".join(citation["trecho"].split())}, index)
        except Exception:  # noqa: BLE001
            return
        citation["resolution"] = resolution

    def _renormalize(self, citation: dict, item: dict, index: CanonicalIndex, evidence: str | None = None) -> None:
        """Propõe uma nova classe para uma citação das regras a partir da leitura do modelo.

        Só para citações que as regras não resolveram como ``real`` e da mesma família que o modelo
        leu. A forma canônica (:func:`canonical`) é consultada na base; se der ``real``, vira proposta
        ``real_nlp``; se der ``inventada`` e as regras não tinham achado o número, ``inventada_nlp``.
        Propostas diferentes de modelos diferentes marcam conflito, e aí fica a resposta das regras.

        Parameters
        ----------
        citation : dict
            Citação das regras (alterada no lugar).
        item : dict
            Citação lida pelo modelo.
        index : CanonicalIndex
            Índice da base.
        evidence : str, optional
            Trecho com entorno curto, para evidência de tribunal, classe e diploma.
        """
        resolution: Resolution = citation.get("_original") or citation["resolution"]
        if resolution.classificacao == REAL:
            return  # o que as regras resolveram não muda
        span = citation["trecho"]
        if GROUP.get(FAMILY.get(item.get("tipo"))) != GROUP.get(citation["familia"]):
            return  # o modelo leu outra coisa neste lugar
        form = canonical(item, span, evidence)
        if not form:
            return
        family, forms = form
        new = resolve_forms(family, forms, citation, index)
        if new is None:
            return  # modelo e trecho divergem
        if new.classificacao == REAL:
            proposal = Resolution(REAL, new.id_canonico, NLP_CONFIDENCE["real_nlp"], "real_nlp")
        elif (new.classificacao == INVENTADA and resolution.regra.startswith("incompleta_sem_numero")
              and family != "julgados_descritivos"):
            # As regras não acharam o número; o modelo achou e a base não tem.
            proposal = Resolution(INVENTADA, None, NLP_CONFIDENCE["inventada_nlp"], "inventada_nlp")
        else:
            return
        proposals = citation.setdefault("_propostas", set())
        proposals.add((proposal.classificacao, proposal.id_canonico))
        citation.setdefault("_original", resolution)
        if len(proposals) > 1:
            citation["_conflito"] = True  # leituras levaram a registros diferentes: não decide
        citation["resolution"] = proposal

    def _acceptable(self, content: str, start: int, end: int, item: dict, header_end: int,
                    own_numbers: set[str]) -> bool:
        """Aplica os filtros de precisão a uma citação que só o modelo viu.

        Recusa distratores (linhas de autos, protocolo, OAB, fls., valor), números no cabeçalho, o
        número dos autos do próprio documento, números que não vêm depois de palavras de citação e
        leituras que não passam pelas guardas de :func:`canonical`.

        Parameters
        ----------
        content : str
            Texto do documento.
        start, end : int
            Span da citação.
        item : dict
            Citação lida pelo modelo.
        header_end : int
            Fim do cabeçalho do documento.
        own_numbers : set of str
            Números longos do cabeçalho (os autos do próprio documento).

        Returns
        -------
        bool
            ``True`` se a citação pode entrar.
        """
        line_start = content.rfind("\n", 0, start) + 1
        if DISTRACTOR_LINE.match(content[line_start:start]) or DISTRACTOR_LINE.match(content[line_start:end]):
            return False
        span = content[start:end]
        digits = _digits(span.translate(OCR_DIGITS))
        if start < header_end and item.get("tipo") == "processo":
            return False
        if _OWN_AUTOS.search(content[max(0, start - 30):end]):
            return False  # "Autos nº …": número do próprio processo (distrator do PDF), mesmo com ruído na linha
        if item.get("tipo") == "processo" and digits and any(digits in n or n in digits for n in own_numbers if n):
            return False  # número dos autos do próprio documento
        if item.get("tipo") == "processo" and not _number_after_citation_words(span, _digits(item.get("numero"))):
            return False  # número solto ("fls. 580/845", "Resolução nº 188", "nº de ordem 295") não é processo
        if canonical(item, span) is not None:
            return True
        return item.get("tipo") == "julgado" and _plausible_julgado(item, span)

    def _classify_new(self, content: str, start: int, end: int, item: dict, index: CanonicalIndex) -> dict | None:
        """Classifica uma citação nova pelas regras e, se não for ``real``, tenta a renormalização.

        Parameters
        ----------
        content : str
            Texto do documento.
        start, end : int
            Span da citação.
        item : dict
            Citação lida pelo modelo.
        index : CanonicalIndex
            Índice da base.

        Returns
        -------
        dict or None
            A citação, com confiança limitada à de citação nova.
        """
        span = content[start:end]
        family = FAMILY[item["tipo"]]
        citation = {"inicio": start, "fim": end, "trecho": span, "familia": family,
                    "tipo": "lei" if family == "artigos" else "jurisprudencia",
                    "contexto": content[max(0, start - 400):end + 400], "contexto_depois": content[end:end + 160]}
        if family == "processos" and re.search(r"\bTST\b|(?<![A-Za-z])(?:A?IRR|ARR|RR)\b", span):
            family = citation["familia"] = "processos_trabalhistas"
        try:
            resolution = resolve(citation | {"trecho_norm": " ".join(span.split())}, index)
        except Exception:  # noqa: BLE001 — a camada nunca derruba o documento
            resolution = Resolution(INCOMPLETA, None, 0.4, "erro_interno")
        citation["resolution"] = Resolution(resolution.classificacao, resolution.id_canonico,
                                            min(resolution.confianca, NLP_CONFIDENCE["novo"]), resolution.regra + "+novo")
        if resolution.classificacao != REAL:
            self._renormalize(citation, item, index)
        return citation


# Palavras que abrem uma citação de processo (classe, recurso, "processo"), comparadas pelo esqueleto.
_OPENERS = ["agravo", "agint", "agrg", "agr", "edcl", "eds", "emb", "embargos", "recurso", "rec", "resp", "aresp",
            "respe", "hc", "habeas", "rhc", "rms", "rcl", "recl", "reclamacao", "re", "are", "ai", "ro", "apl",
            "apelacao", "rse", "rr", "airr", "arr", "tst", "stj", "stf", "tse", "stm", "processo", "proc", "acao",
            "mandado", "ms"]
_OPENER_SK = sorted({skeleton(o)[0] for o in _OPENERS}, key=len, reverse=True)
_UF_TAIL = re.compile(r"\s*(?:[/\-–(]\s*)?([A-Z]{2})\)?(?![A-Za-z])")


def _trim_process(content: str, start: int, end: int, digits: str) -> tuple[int, int]:
    """Recorta o núcleo de uma citação de processo dentro de um trecho longo demais.

    Fim: fim do número (mais a UF). Começo: primeira palavra que abre citação (classe, recurso)
    nos 60 caracteres antes do número; sem ela, o próprio número. Trecho já compacto não muda.

    Parameters
    ----------
    content : str
        Texto (já limpo, ver :func:`_on_clean`).
    start, end : int
        Span do trecho.
    digits : str
        Número lido pelo modelo; se não estiver no trecho, vale o único candidato.

    Returns
    -------
    tuple of (int, int)
        O span recortado.
    """
    span = content[start:end]
    groups = number_group_spans(span)
    found = [(a, b) for g, a, b in groups if digits and g.lstrip("0") == digits.lstrip("0")]
    if not found:  # o modelo leu mal o número: vale o único candidato do trecho (como em canonical)
        found = [(a, b) for g, a, b in groups if len(g) >= 3 and not _is_year(g)]
        if len(found) != 1:
            return start, end
    num_start, num_end = found[0]
    if (uf := _UF_TAIL.match(span, num_end)) and uf.group(1) in UFS:
        num_end = uf.end()
    head = span[max(0, num_start - 60):num_start]
    offset = max(0, num_start - 60)
    core_start = num_start
    for m in re.finditer(r"[^\W_]+", head):
        word = skeleton(m.group())[0]
        # Sigla curta ("RO", "RE", "HC") só em maiúsculas: "ro" pode ser o fim de "regist-ro".
        if any(word.startswith(o) and (len(o) > 3 or (word == o and sum(c.isupper() for c in m.group()) >= 2))
               for o in _OPENER_SK):
            core_start = offset + m.start()
            break
    if (end - start) <= 1.4 * (num_end - core_start):
        return start, end
    return start + core_start, start + num_end


_THOUSANDS = re.compile(r"[ .\u00a0]+[\dOolI|SsgqGbBZz]{3}(?![\w])")


def _trim_article(content: str, start: int, end: int, digits: str) -> tuple[int, int]:
    """Recorta o núcleo de uma citação de artigo dentro de um trecho longo demais.

    Vai da pista ("art.", "artigo") com o número lido pelo modelo até o fim do nome do diploma, com
    as mesmas peças que as regras usam.

    Parameters
    ----------
    content : str
        Texto (já limpo).
    start, end : int
        Span do trecho.
    digits : str
        Número do artigo lido pelo modelo.

    Returns
    -------
    tuple of (int, int)
        O span recortado; o original se não houver recorte seguro.
    """
    span = content[start:end]
    if len(span) <= 120:
        return start, end
    cues = []
    for cue in _ART_CUE.finditer(span):
        stop = cue.end()
        while thousands := _THOUSANDS.match(span, stop):  # "art. 1 105", "art. 1.105"
            stop = thousands.end()
        if not digits or digits_in_span(digits, span[cue.start():stop]):
            cues.append((cue, stop))
    for cue, stop in reversed(cues):  # frase inteira: a citação costuma vir no fim
        while enum := _ENUM.match(span, stop):
            stop = enum.end()
        for prep in _PREPS:
            m = prep.match(span, stop)
            if m and (dip_end := _longest_diploma(span, m.end())):
                return start + cue.start(), start + dip_end
    return start, end


_SUMULA_TAIL = re.compile(r"[\s,]*(?:(?:d|cl)[oa]\s*)?(?:[S5]T[FJM]|T[S5][TE])(?![A-Za-z])")


def _trim_sumula(content: str, start: int, end: int, digits: str) -> tuple[int, int]:
    """Recorta o núcleo de uma súmula dentro de um trecho longo demais.

    Vai da pista ("Súmula", "Súm.", "verbete") até o número lido pelo modelo e, se vier logo
    depois, o tribunal ("… 331 do TST").

    Parameters
    ----------
    content : str
        Texto (já limpo).
    start, end : int
        Span do trecho.
    digits : str
        Número da súmula lido pelo modelo.

    Returns
    -------
    tuple of (int, int)
        O span recortado; o original se não houver recorte seguro.
    """
    span = content[start:end]
    if len(span) <= 40 or not digits:
        return start, end
    for cue in _SUMULA_CUE.finditer(span):
        for number in number_group_spans(span[cue.end():cue.end() + 40]):
            if number[0].lstrip("0") != digits.lstrip("0"):
                continue
            stop = cue.end() + number[2]
            if tail := _SUMULA_TAIL.match(span, stop):
                stop = tail.end()
            return start + cue.start(), start + stop
    return start, end


def _on_clean(core, content: str, start: int, end: int, digits: str) -> tuple[int, int]:
    """Aplica um recortador ao trecho limpo pelas regras e devolve offsets do texto original.

    A limpeza trata acento decomposto, mojibake, caracteres invisíveis e numeração de margem.

    Parameters
    ----------
    core : callable
        Recortador (:func:`_trim_process`, :func:`_trim_article` ou :func:`_trim_sumula`).
    content : str
        Texto do documento.
    start, end : int
        Span do trecho no texto original.
    digits : str
        Número lido pelo modelo.

    Returns
    -------
    tuple of (int, int)
        O span recortado, no texto original.
    """
    composed, nfc_map = _nfc(content[start:end])
    cleaned, mapping = clean(composed)
    a, b = core(cleaned, 0, len(cleaned), digits)
    if (a, b) == (0, len(cleaned)):
        return start, end
    a, b = to_original(mapping, a, b)
    return start + nfc_map[a], start + (nfc_map[b] if b < len(nfc_map) else end - start)


def _nfc(text: str) -> tuple[str, list[int]]:
    r"""Recompõe acentos decompostos ("Su\u0301mula" -> "Súmula").

    Parameters
    ----------
    text : str
        Texto de entrada.

    Returns
    -------
    composed : str
        O texto em NFC.
    index : list of int
        Para cada caractere recomposto, o índice no texto original.
    """
    out, index, i = [], [], 0
    while i < len(text):
        j = i + 1
        while j < len(text) and unicodedata.combining(text[j]):
            j += 1
        for ch in unicodedata.normalize("NFC", text[i:j]):
            out.append(ch)
            index.append(i)
        i = j
    return "".join(out), index


def trim_process(content: str, start: int, end: int, digits: str) -> tuple[int, int]:
    """Recorta o núcleo de uma citação de processo (ver :func:`_trim_process`).

    Parameters
    ----------
    content : str
        Texto do documento.
    start, end : int
        Span do trecho.
    digits : str
        Número lido pelo modelo.

    Returns
    -------
    tuple of (int, int)
        O span recortado, no texto original.
    """
    return _on_clean(_trim_process, content, start, end, digits)


def trim_article(content: str, start: int, end: int, digits: str) -> tuple[int, int]:
    """Recorta o núcleo de uma citação de artigo (ver :func:`_trim_article`).

    Parameters
    ----------
    content : str
        Texto do documento.
    start, end : int
        Span do trecho.
    digits : str
        Número do artigo lido pelo modelo.

    Returns
    -------
    tuple of (int, int)
        O span recortado, no texto original.
    """
    return _on_clean(_trim_article, content, start, end, digits)


def trim_sumula(content: str, start: int, end: int, digits: str) -> tuple[int, int]:
    """Recorta o núcleo de uma súmula (ver :func:`_trim_sumula`).

    Parameters
    ----------
    content : str
        Texto do documento.
    start, end : int
        Span do trecho.
    digits : str
        Número da súmula lido pelo modelo.

    Returns
    -------
    tuple of (int, int)
        O span recortado, no texto original.
    """
    return _on_clean(_trim_sumula, content, start, end, digits)


def _plausible_julgado(item: dict, span: str) -> bool:
    """Aceita um julgado descrito que a guarda estrita não confirmou (OCR pesado no nome).

    Exige ano, tribunal (ou classe) e um nome parecido com o lido pelo modelo no trecho. A classe
    continua vindo das regras; isto só decide se o trecho entra.

    Parameters
    ----------
    item : dict
        Citação lida pelo modelo.
    span : str
        Trecho da citação.

    Returns
    -------
    bool
        ``True`` se o trecho é plausível.
    """
    if not any(len(g) == 4 and g[:2] in ("19", "20") for g in number_groups(span)):
        return False
    if not (courts_in(span) or _COURT_WORDS.search(span) or class_code(span)):
        return False
    span_tokens = [skeleton(t)[0] for t in re.findall(r"[^\W\d_]{3,}", span)]
    return any(difflib.SequenceMatcher(None, skeleton(t)[0], s).ratio() >= 0.75
               for t in name_tokens(item.get("relator") or "") for s in span_tokens)


_OWN_AUTOS = re.compile(r"(?<![^\W\d_])[aá][uvn]t[o0][s5](?![^\W\d_])", re.IGNORECASE)


def _same_citation(a: tuple[int, int], b: tuple[int, int]) -> bool:
    """Diz se dois spans são a mesma citação: sobreposição de ao menos 50% do menor.

    Parameters
    ----------
    a, b : tuple of (int, int)
        Spans (início, fim).

    Returns
    -------
    bool
        ``True`` se os spans se sobrepõem o bastante.
    """
    shared = min(a[1], b[1]) - max(a[0], b[0])
    return shared > 0 and shared >= 0.5 * min(a[1] - a[0], b[1] - b[0])


class _DocState:
    """O que a revisão de um documento reaproveita entre as leituras dos modelos.

    Parameters
    ----------
    content : str
        Texto do documento.

    Attributes
    ----------
    content : str
        Texto do documento.
    text_sk : str
        Esqueleto do texto (:func:`skeleton`).
    text_index : list of int
        Índices do esqueleto no texto original.
    header_end : int
        Fim do cabeçalho (:func:`_header_end`).
    own_numbers : set of str
        Números longos do cabeçalho (os autos do próprio documento).
    """

    def __init__(self, content: str):
        self.content = content
        self.text_sk, self.text_index = skeleton(content)
        self.header_end = _header_end(content)
        self.own_numbers = {_digits(m.group()) for m in re.finditer(r"[\d.\-]{15,}", content[:self.header_end])}


def _header_end(content: str) -> int:
    """Localiza o fim do cabeçalho: o início da primeira linha de prosa.

    O cabeçalho só tem linhas curtas; procura até 1.500 caracteres.

    Parameters
    ----------
    content : str
        Texto do documento.

    Returns
    -------
    int
        Posição do fim do cabeçalho; 0 se não achar.
    """
    pos = 0
    for line in content.splitlines(keepends=True):
        if len(line.strip()) >= 70:
            return pos
        pos += len(line)
        if pos > 1500:
            break
    return 0


def from_args(enabled: bool, url: str, model: str, cache: Path | None, *, normalize: bool = True,
              recall: bool = True, workers: int = 1, budget_doc_s: float = 40.0,
              budget_total_s: float = 3 * 3600.0, extra_models: tuple[str, ...] = ()) -> NLPLayer | None:
    """Monta a camada pronta para uso, com os modelos disponíveis.

    Parameters
    ----------
    enabled : bool
        Se a camada deve ser ligada.
    url : str
        Servidor Ollama (local).
    model : str
        Modelo principal.
    cache : pathlib.Path or None
        Pasta de cache das respostas.
    normalize : bool, default True
        Renormaliza citações das regras.
    recall : bool, default True
        Acrescenta citações novas.
    workers : int, default 1
        Pedidos em paralelo por documento.
    budget_doc_s : float, default 40.0
        Média máxima de segundos por documento.
    budget_total_s : float, default 10800.0
        Tempo total máximo, em segundos.
    extra_models : tuple of str, optional
        Modelos do conjunto; os indisponíveis ficam de fora, com aviso.

    Returns
    -------
    NLPLayer or None
        A camada; ``None`` (com aviso) se desligada, se o servidor não for local ou se não houver
        servidor ou modelo principal.
    """
    if not enabled:
        return None
    try:
        client = OllamaClient(url, model, cache_dir=cache)
    except ValueError as error:
        print(f"[nlp] {error}: seguindo só com as regras", file=sys.stderr)
        return None
    if not client.available():
        print(f"[nlp] servidor {url} indisponível ou sem o modelo {model}: seguindo só com as regras",
              file=sys.stderr)
        return None
    extras = tuple(OllamaClient(url, m, cache_dir=cache) for m in extra_models)
    missing = [c.model for c in extras if not c.available()]
    if missing:
        print(f"[nlp] modelos extras indisponíveis ({', '.join(missing)}): seguindo só com {model}", file=sys.stderr)
        extras = tuple(c for c in extras if c.model not in missing)
    return NLPLayer(client, normalize=normalize, recall=recall, workers=workers, budget_doc_s=budget_doc_s,
                    budget_total_s=budget_total_s, extra_clients=extras)
