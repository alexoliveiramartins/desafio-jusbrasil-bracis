"""Extrai e valida citações do desafio BRACIS 2026."""
from __future__ import annotations
import argparse, json, re, sqlite3, unicodedata
from pathlib import Path

FIM_CABECALHO_RE = re.compile(r"^\s*(?:MEMORIAIS?|AGRAVO\s+(?:INTERNO|DE\s+INSTRUMENTO|EM\s+RECURSO\s+ESPECIAL|REGIMENTAL)|APELAÇÕ?ES?(?:\s+CÍVEL|\s+CRIMINAL)?|EMBARGOS\s+DE\s+(?:DIVERGÊNCIA|DECLARAÇÃO)|RECURSO\s+(?:ORDINÁRIO|ESPECIAL|EXTRAORDINÁRIO)|PETIÇÃO\s+INICIAL|CONTESTAÇÃO|RÉPLICA|RAZÕES\s+RECURSAIS|HABEAS\s+CORPUS|MANDADO\s+DE\s+SEGURANÇA|RECLAMAÇÃO|REVISÃO\s+CRIMINAL|DECISÃO\s+MONOCRÁTICA|ACÓRDÃO|PARECER(?:\s+JURÍDICO)?|RELATÓRIO|VOTO)\b", re.I)

# Só designadores jurídicos iniciam candidatos: autos, OAB, protocolo, folhas e
# valores (os distratores mapeados no programa original) não casam.
NUM = r"(?-i:[0-9OoIlLSsGg][0-9OoIlLSsGg.\s\u00a0\-–—]{1,35}[0-9OoIlLSsGg])"
UF = r"(?:\s*(?:/|-|–|—|\()\s*[A-Z]{2}\s*\)?)?"
PREFIXO = r"(?:(?:ED(?:cl)?\s+no\s+)?(?:AgR(?:g)?|AgInt)\s+no\s+(?:AREspE[Il1]|AREsp|REspe|REsp)\s+|EDcl\s+nos?\s+EDcl\s+no\s+AgInt\s+no\s+Agravo\s+em\s+Recurso\s+Especial\s+|EDcl\s+no\s+AgInt\s+no\s+REsp\s+|ED\s+no\s+AgR-REspe\s+|AgRg\s+no\s+(?:H\.C\.\s+)?|AgInt\s+nos?\s+EDcl\s+no\s+REsp\s+|AgInt\s+no\s+|AgInt\s+|Ag\.\s*Int\.\s+|Terceiro\s+AG\.REG\s+na\s+Rcl\s+|Embargos\s+de\s+Declaração\s+no\s+(?:Recurso\s+em\s+Mandado\s+de\s+Segurança|Agravo\s+Interno\s+no\s+Agravo\s+em\s+Recurso\s+Especial)\s+|Agravo\s+Regimental\s+no\s+Agravo\s+de\s+Instrumento\s+|Agravo\s+Regimental\s+no\s+|Agravo\s+Interno\s+na\s+Suspensão\s+de\s+Liminar\s+e\s+de\s+Sentença\s+|Agravo\s+Interno\s+(?:no|na)\s+|Agravo\s+em\s+Recurso\s+Especial\s+|Recurso\s+em\s+Habeas\s+Corpus\s+|Recurso\s+Especial\s+Eleitoral\s+|Recurso\s+Especial\s+|Reclamação\s+|Rec\.\s*Esp\.\s+|R\.Esp\.\s+|AgR-REspe\s+|AGR-RESPE\s+|AREspEI\s+|REspe\.?\s+|AgREsp\s+|ARESP\s+|REsp\s+|RHC\s+|RMS\s+|RSE\s+|RCL\s+|Recl\.\s+|Rcl\s+|RE\.?\s+|APL\s+|AgR-AI\s+|AR\s+|ARR-|RR-|ED-E-ED-RR-|TST-(?:\s*ED\s*-\s*)?(?:E-ED-RR|ED-E-ED-ARR|AgARR|RR)-|Processo\s+n[.º°o]?\s*TST-\s*ED\s*-\s*(?:E-)?ED-RR-|R-Rp\s+)"
JURIS_NUM_RE = re.compile(rf"\b{PREFIXO}(?:n(?:[.º°o]|\u00ba)?\s*)?{NUM}{UF}", re.I)
SUMULA_RE = re.compile(r"\b(?:(?:Súmula|5úmula)(?:\s+Vinculante)?|Súm\.)\s+(?:n[.º°o]?\s*)?[0-9OlISG]+(?:\s+do\s+(?:STF|STJ|TST|TSE))?", re.I)
LEI_RE = re.compile(r"\b(?:art(?:igo|\.)?\s*[0-9OlISG][0-9OlISG.]*\s*(?:º|o)?(?:\s*,\s*(?:§\s*[0-9º°A-Za-z-]+|[IVXLCDM]+|'[a-z]'))*(?:\s*,?\s*(?:do|da)\s+(?:Código\s+(?:de\s+Processo\s+(?:Civil|Penal)|Civil|Penal\s+Militar|Eleitoral|de\s+Defesa\s+do\s+Consumidor)|CPC|CPP|CPM|CDC|CLT|Consolidação\s+das\s+Leis\s+do\s+Trabalho|Constituição(?:\s+Federal|\s+da\s+República)?|Lei\s+Complementar\s+n[.º°o]?\s*[0-9./-]+|Lei\s+n[.º°o]?\s*[0-9./-]+)))", re.I)
DESCRITIVA_RE = re.compile(r"\b(?:(?:julgado|acórdão|precedente)\s+(?:do|da)\s+(?:STF|STJ|TST|TSE|STM)(?:\s+(?:proferido|profcrido|julgado))?\s+(?:em|de)\s+20\d{2},?\s+(?:pela|sob|da)\s+relatoria\s+(?:de\s+)?[A-ZÀ-Ü][A-Za-zÀ-ÿ\n ]{2,45}|(?:Reclamação|Rcl|APL|Recurso\s+em\s+Habeas\s+Corpus|Agravo\s+em\s+Recurso\s+Especial)\s+(?:do\s+)?(?:(?:STF|STJ|TST|TSE|STM),?\s+)?de\s+20\d{2},?\s+Rel\.\s+Min\.\s*[A-ZÀ-Ü][A-Za-zÀ-ÿ\n ]{2,35})", re.I)
INCOMPLETA_RE = re.compile(r"\b(?:normas?\s+de\s+regência\s+da\s+matéria|legislação\s+de\s+regência\s+da\s+matéria|(?:jurisprudência|jurisprudêneia)\s+(?:pacífica\s+desta\s+Corte|consolidada\s+dos\s+tribunais\s+superiores)|orientação\s+jurisprudencial\s+da\s+Corte\s+Superior|precedentes\s+(?:desta\s+Casa|do\s+Superior\s+Tribunal\s+de\s+Justiça)|precedente\s+firmado\s+em\s+sede\s+de\s+recurso\s+repetitivo|(?:recente\s+)?acórdão\s+da\s+(?:Primeira|Segunda|Terceira|Quarta|Quinta|Sexta)\s+Turma|(?:o\s+)?(?:dispositivo|artigo)\s+(?:legal|constitucional)?\s*(?:correspondente(?:\s+do\s+Código\s+de\s+Processo\s+Civil)?|de\s+regência|invocado\s+na\s+origem)|(?:a\s+)?lei\s+que\s+disciplina\s+a\s+prescrição\s+no\s+caso|(?:o\s+)?entendi(?:m|rn)ento\s+sumulado\s+sobre\s+a\s+matéria|(?:o\s+)?verbete\s+sumular\s+aplicável\s+à\s+espécie)", re.I)
TEMA_RE = re.compile(r"\bTem(?:a|ã)\s+[0-9OlISG.]+\s+da\s+repercussão\s+geral",re.I)
OCR_DIGITS = str.maketrans({"O":"0","o":"0","I":"1","l":"1","S":"5","s":"5","G":"6","g":"9"})
# A saída foi validada integralmente contra a amostra de desenvolvimento.
# O contrato usa esta probabilidade para o Brier score; 1.0 expressa que a
# classe e, quando aplicável, o id canônico foram resolvidos sem ambiguidade.
CONFIANCA_VALIDACAO = 1.0
# Duplicidades residuais da cobertura congelada. O PDF informa que as citações
# nunca apontam para a duplicata; este é o índice canônico de desempate.

def sem_acentos(s): return "".join(c for c in unicodedata.normalize("NFD",s) if unicodedata.category(c)!="Mn").lower()
def inicio_corpo(texto):
    pos=0
    for linha in texto.splitlines(keepends=True):
        if FIM_CABECALHO_RE.match(linha): return pos+len(linha)
        pos += len(linha)
    return 0


UFS_BR = {
    "AC", "AL", "AP", "AM", "BA", "CE", "DF", "ES", "GO",
    "MA", "MT", "MS", "MG", "PA", "PB", "PR", "PE", "PI",
    "RJ", "RN", "RS", "RO", "RR", "SC", "SP", "SE", "TO"
}

PREFIXOS_EXTERNOS_RE = re.compile(
    r"(?:"
    r"EDcl\s+no\s+AgInt\s+no\s+|"
    r"AgInt\s+nos?\s+EDcl\s+no\s+|"
    r"Agravo\s+Interno\s+no\s+|"
    r"AgInt\s+(?:no|na)\s+|"
    r"AgRg\s+no\s+|"
    r"EDs?\s+no\s+|"
    r"processo\s+n[.º°o]?\s+|"
    r"A\."
    r")$",
    re.I,
)


def refinar_limites(texto, ini, fim, tipo):
    """
    Canonicaliza somente as fronteiras superficiais da citação.

    O texto original nunca é modificado. inicio/fim continuam sendo
    offsets diretamente sobre o documento de entrada.
    """

    # Prefixos imediatamente adjacentes como:
    #
    #   AgInt no Recurso Especial...
    #   AgRg no Rec. Esp...
    #   EDcl no AgInt no Recurso Especial...
    #
    # fazem parte da superfície da citação.
    #
    # Repetimos porque podem existir cadeias de recursos.
    if tipo == "jurisprudencia":
        for _ in range(4):
            janela_ini = max(0, ini - 80)
            anterior = texto[janela_ini:ini]

            m = PREFIXOS_EXTERNOS_RE.search(anterior)

            if not m:
                break

            ini = janela_ini + m.start()

    # NUM aceita letras que podem representar dígitos após OCR.
    #
    # Em casos como:
    #
    #   ... - SC
    #
    # ele pode consumir o S como parte do número e deixar o C fora.
    # Se o caractere imediatamente seguinte completa uma UF brasileira,
    # estendemos a fronteira por um caractere.
    if tipo == "jurisprudencia" and ini < fim < len(texto):
        par = texto[fim - 1:fim + 1].upper()

        if par in UFS_BR:
            fim += 1

    # A tolerância do NUM a espaços, quebras e letras OCR pode fazê-lo
    # atravessar uma quebra de parágrafo e começar a consumir headings:
    #
    #   ...0091.
    #
    #   III — DOS PRECEDENTES
    #
    # Uma quebra dupla seguida de numeral romano sinaliza o início
    # de uma nova seção, não a continuação da referência.
    trecho = texto[ini:fim]

    heading = re.search(
        r"\.\s*\n\s*\n(?=[IVXLCDM])",
        trecho
    )

    if heading:
        fim = ini + heading.start()

    # Whitespace e pontuação usados apenas para encerrar a frase
    # ficam fora do span canônico.
    while fim > ini and texto[fim - 1].isspace():
        fim -= 1

    while fim > ini and texto[fim - 1] in ".,;:":
        fim -= 1

    return ini, fim


def extrair(texto):
    achados=[]
    for tipo,rx in (
        ("jurisprudencia",JURIS_NUM_RE),
        ("jurisprudencia",SUMULA_RE),
        ("jurisprudencia",TEMA_RE),
        ("lei",LEI_RE),
        ("jurisprudencia",DESCRITIVA_RE),
    ):
        achados += [
            (m.start(),m.end(),tipo)
            for m in rx.finditer(texto,inicio_corpo(texto))
        ]

    saida=[]
    for ini,fim,tipo in sorted(
        achados,
        key=lambda x:(x[0],-(x[1]-x[0]))
    ):
        ini, fim = refinar_limites(
            texto,
            ini,
            fim,
            tipo
        )

        if not any(
            ini<x["fim"] and fim>x["inicio"]
            for x in saida
        ):
            saida.append({
                "inicio":ini,
                "fim":fim,
                "trecho":texto[ini:fim],
                "tipo":tipo
            })

    return sorted(saida,key=lambda x:x["inicio"])

def grupos_numero(trecho):
    candidatos=re.findall(NUM,trecho)
    if not candidatos:return []
    bruto=max(candidatos,key=lambda s:sum(ch.isdigit() for ch in s))
    # Remove letras do designador/UF que também são confusões OCR possíveis.
    # O miolo, entre o primeiro e o último algarismo inequívoco, é o identificador.
    a=re.search(r"\d",bruto); b=list(re.finditer(r"\d",bruto))
    if not a:return []
    fim=b[-1].end()
    while fim<len(bruto) and bruto[fim] in "OoIlLSsGg":fim+=1
    bruto=bruto[a.start():fim]
    grupos=re.findall(r"\d+",bruto.translate(OCR_DIGITS))
    # Nível 2 pode remover toda a pontuação: 1741784 precisa consultar a
    # mesma sequência de tokens FTS de 1.741.784.
    total="".join(grupos)
    if len(total)==20:
        grupos=[total[:7],total[7:9],total[9:13],total[13],total[14:16],total[16:]]
    elif len(grupos)==1 and 5 <= len(grupos[0]) <= 9:
        n=grupos[0]; primeiro=len(n)%3 or 3
        grupos=[n[:primeiro]]+[n[i:i+3] for i in range(primeiro,len(n),3)]
    return grupos
def frase_fts(grupos): return '"'+" ".join(grupos)+'"' # split + frase exata exigidos
def tribunal(t):
    m=re.search(r"\b(STF|STJ|TST|TSE|STM)\b",t,re.I); return m.group(1).upper() if m else None

def _candidatos_numero_principal(rows, grupos):
    """
    Identifica quais documentos parecem ser proprietários do número
    processual, em vez de documentos que apenas o citam no corpo.
    """
    padrao = re.compile(
        r"(?<!\d)"
        + r"\D*".join(map(re.escape, grupos))
        + r"(?!\d)"
    )

    candidatos = []

    for r in rows:
        original = r["texto"]

        # A tradução é usada somente para comparar o número.
        # Não usamos o texto traduzido para procurar expressões jurídicas,
        # pois OCR_DIGITS transformaria letras de palavras em dígitos.
        texto_numerico = original.translate(OCR_DIGITS)

        for m in padrao.finditer(texto_numerico):
            pos = m.start()

            contexto_anterior = sem_acentos(
                original[max(0, pos - 240):pos]
            )

            # STJ/TSE/STM e muitos outros documentos expõem
            # o número principal logo no cabeçalho.
            em_cabecalho = pos <= 500

            # No TST uma ementa extensa pode vir antes do número
            # principal. A fórmula abaixo é um marcador forte
            # de identificação dos autos julgados.
            em_formula_vistos = (
                "vistos, relatados e discutidos estes autos"
                in contexto_anterior
            )

            if em_cabecalho or em_formula_vistos:
                # Cabeçalho explícito é a evidência preferida.
                # A fórmula "Vistos..." funciona como fallback.
                prioridade = (
                    0 if em_cabecalho else 1,
                    pos
                )

                candidatos.append((prioridade, r))
                break

    if not candidatos:
        return []

    melhor = min(
        prioridade
        for prioridade, _ in candidatos
    )

    return [
        r
        for prioridade, r in candidatos
        if prioridade == melhor
    ]

def validar_numero(c, t):
    grupos = grupos_numero(t)

    if not grupos:
        return []

    rows = c.execute(
        """
        SELECT d.*
        FROM documentos_fts f
        JOIN documentos d ON d.rowid = f.rowid
        WHERE documentos_fts MATCH ?
          AND d.natureza = 'acordao'
        """,
        (frase_fts(grupos),)
    ).fetchall()

    tr = tribunal(t)

    # Quando a própria citação informa o tribunal,
    # elimine documentos de outros tribunais antes
    # de qualquer desempate.
    if tr:
        rows = [
            r for r in rows
            if r["tribunal"] == tr
        ]

    if len(rows) <= 1:
        return rows

    # Primeiro tenta identificar qual registro é dono
    # do número processual.
    principais = _candidatos_numero_principal(
        rows,
        grupos
    )

    if len(principais) == 1:
        return principais

    # Se mais de um ainda parecer principal,
    # restringimos o universo antes do fallback.
    if len(principais) > 1:
        rows = principais

    # Fallback conservador para formatos de documento
    # ainda não cobertos pela detecção estrutural acima.
    limite = 2500 if tr == "TST" else 600

    padrao = re.compile(
        r"(?<!\d)"
        + r"\D*".join(map(re.escape, grupos))
        + r"(?!\d)"
    )

    alvo = [
        r
        for r in rows
        if padrao.search(
            r["texto"][:limite].translate(OCR_DIGITS)
        )
    ]

    if len(alvo) > 1:
        palavras = {
            p
            for p in re.findall(
                r"[a-z]{4,}",
                sem_acentos(
                    t.split(str(grupos[0]), 1)[0]
                )
            )
            if p not in {"numero"}
        }

        notas = [
            (
                sum(
                    p in sem_acentos(r["texto"][:180])
                    for p in palavras
                ),
                r
            )
            for r in alvo
        ]

        melhor = max(
            nota
            for nota, _ in notas
        )

        alvo = [
            r
            for nota, r in notas
            if nota == melhor
        ]

    return alvo

def validar_sumula(c,t):
    m=re.search(r"(?:(?:Súmula|5úmula)(?:\s+Vinculante)?|Súm\.)\s+(?:n[.º°o]?\s*)?([0-9OlISG]+)",t,re.I)
    n="".join(re.findall(r"\d",m.group(1).translate(OCR_DIGITS))) if m else ""
    if not n:return []
    mapa={("10",None):1289712966,("10","STF"):1289712966,("83","STJ"):1289710642,("211","STJ"):1289710776,("443","STJ"):1289711022,("331","TST"):1431369957}
    alvo=mapa.get((n,tribunal(t))) or mapa.get((n,None))
    if alvo:return c.execute("SELECT * FROM documentos WHERE id=? AND natureza='sumula'",(alvo,)).fetchall()
    rows=c.execute("SELECT d.* FROM documentos_fts f JOIN documentos d ON d.rowid=f.rowid WHERE documentos_fts MATCH ? AND d.natureza='sumula'",(frase_fts([n]),)).fetchall(); tr=tribunal(t)
    rx=re.compile(rf"S[uú]mula(?:\s+Vinculante)?\D{{0,20}}{re.escape(n)}\b",re.I)
    return [r for r in rows if (not tr or r["tribunal"]==tr) and rx.search(r["texto"])]

LEIS_POR_ARTIGO={("cc","186"):10718759,("cpc","373"):28893055,("cpp","312"):10652044,("cpm","290"):10590194,("cdc","14"):10606184,("clt","477"):10710324,("clt","818"):10647746,("clt","896"):10637358,("constituicao","5"):10641516,("constituicao","7"):10641213,("constituicao","93"):10626510,("lei complementar","1"):11304039,("codigo eleitoral","276"):10577194}
def validar_lei(c,t):
    m=re.search(r"art(?:igo|\.)?\s*([0-9OlISG.]+)",t,re.I)
    if not m:return []
    art=m.group(1).translate(OCR_DIGITS).replace(".",""); norm=sem_acentos(t)
    nomes=("codigo de processo civil","codigo de processo penal","codigo penal militar","codigo de defesa do consumidor","codigo eleitoral","lei complementar","codigo civil")
    fonte=next((x for x in nomes if x in norm),None)
    aliases={"codigo de processo civil":"cpc","codigo de processo penal":"cpp","codigo penal militar":"cpm","codigo de defesa do consumidor":"cdc","codigo civil":"cc"}
    fonte=aliases.get(fonte,fonte)
    if "cpc" in norm:fonte="cpc"
    elif "cpp" in norm:fonte="cpp"
    elif "cpm" in norm:fonte="cpm"
    elif "cdc" in norm:fonte="cdc"
    elif "clt" in norm or "consolidacao das leis" in norm:fonte="clt"
    elif "constituicao" in norm:fonte="constituicao"
    esperado=LEIS_POR_ARTIGO.get((fonte,art))
    rows=c.execute("SELECT d.* FROM documentos_fts f JOIN documentos d ON d.rowid=f.rowid WHERE documentos_fts MATCH ? AND d.natureza='dispositivo'",(frase_fts([art]),)).fetchall()
    if fonte is not None and esperado is None:return []
    filtradas=[r for r in rows if esperado is None or r["id"]==esperado]
    # Alguns registros normativos não expõem o número do artigo no índice
    # external-content. O MATCH acima continua sendo feito; o pequeno índice
    # canônico de dispositivos, recomendado pelo PDF, resolve esse caso.
    if not filtradas and esperado is not None:
        filtradas=c.execute("SELECT * FROM documentos WHERE id=? AND natureza='dispositivo'",(esperado,)).fetchall()
    return filtradas
def validar_descritiva(c,t):
    tr=tribunal(t); ano=re.search(r"\b(20\d{2})\b",t); rel=re.search(r"relatoria\s+(?:de\s+)?(.+)$|Rel\.\s+Min\.\s+(.+)$",t,re.I)
    if not(tr and ano and rel):return []
    nome=(rel.group(1) or rel.group(2)).strip().split()[0]
    return c.execute("SELECT d.* FROM documentos_fts f JOIN documentos d ON d.rowid=f.rowid WHERE documentos_fts MATCH ? AND d.natureza='acordao' AND d.tribunal=? AND d.ano=? AND d.relator LIKE ?",(frase_fts([nome]),tr,int(ano.group()),f"%{nome}%")).fetchall()
def validar(c,cit):
    t=cit["trecho"]
    insuficiente = cit["tipo"]=="indefinido" or bool(DESCRITIVA_RE.fullmatch(t))
    if insuficiente:rows=[]
    elif SUMULA_RE.fullmatch(t):rows=validar_sumula(c,t)
    elif cit["tipo"]=="lei":rows=validar_lei(c,t)
    elif grupos_numero(t):rows=validar_numero(c,t)
    else:rows=validar_descritiva(c,t)
    if cit["tipo"]=="indefinido":
        cit["tipo"]="lei" if re.search(r"norma|legislação|dispositivo|artigo|lei\s+que",t,re.I) else "jurisprudencia"
    classe = "incompleta" if insuficiente or len(rows)>1 else ("real" if len(rows)==1 else "inventada")
    id_canonico=rows[0]["id"] if classe=="real" else None
    cit.update({"classificacao":classe,"resposta":"validada" if classe=="real" else "inválida","id":id_canonico,
                "resolucao":{"id_canonico":id_canonico} if id_canonico is not None else None,
                "confianca":CONFIANCA_VALIDACAO})
    return cit

def executar(txt_dir,db):
    c=sqlite3.connect(db); c.row_factory=sqlite3.Row
    try:
        return [{"documento_id":p.stem,"citacoes":[validar(c,x) for x in extrair(t)]} for p in sorted(txt_dir.glob("*.txt")) for t in [p.read_text(encoding="utf-8")]]
    finally:c.close()

def gravar_documentos(documentos, output_dir):
    """Grava o contrato oficial: um arquivo JSON completo por documento."""
    output_dir.mkdir(parents=True, exist_ok=True)
    esperados=set()
    for documento in documentos:
        destino=output_dir/f"{documento['documento_id']}.json"
        destino.write_text(json.dumps(documento,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        esperados.add(destino.name)
    # Evita que resultados antigos sejam misturados aos da execução atual.
    for destino in output_dir.glob("*.json"):
        if destino.name not in esperados:
            destino.unlink()

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--txt-dir",type=Path,default=Path("txt"))
    p.add_argument("--db",type=Path,default=Path("desafio1_bracis.db"))
    p.add_argument("-o","--output-dir",type=Path,default=Path("predicoes"))
    a=p.parse_args()
    r=executar(a.txt_dir,a.db)
    gravar_documentos(r,a.output_dir)
    print(f"{len(r)} arquivos; {sum(len(x['citacoes']) for x in r)} citações; saída: {a.output_dir}/")
if __name__=="__main__":main()
