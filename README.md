# desafio-jusbrasil-bracis

Experimento opcional: [fuzzy regex](docs/ExperimentoFuzzy.md), com comparação
entre tolerância global e tolerância restrita a palavras.

- Rodar o extrator:

```bash
python3 -m src.extractor
```

Os padrões e seus tipos ficam em `src/re_patterns.py`. O `src/extractor.py`
aplica os padrões ao texto original, remove capturas contidas em outras e grava
um JSON por documento. Também é possível executar `python3 src/extractor.py`.
As pastas de entrada (`data/txt`) e saída (`resultados`) ficam definidas em `main()`.

Famílias atuais: processos numerados (incluindo trabalhistas), súmulas, artigos
de lei e processos/julgados descritivos por ano/relator. Há tolerância a espaços e
quebras de linha, mas a normalização de OCR do nível 2 ainda não está implementada.
Processos aceitam variantes como `AgREsp`, `H.C.`, `A.REsp`, `R.Esp.` e `Recl.`,
milhares separados por espaços e blocos CNJ com separadores ausentes ou repetidos.
Os dígitos e o trecho original são preservados.

**Classificação provisória:** todas as citações ainda saem como `inventada`,
com `resolucao: null`. Isso mantém o fluxo de conversão existente; não representa
uma consulta à base canônica. A pontuação completa não mede só a extração.

- Verificar o extrator:

```bash
python3 -m unittest discover -s tests
```

- Medir a extração:

Antes de avaliar a classificação, você pode medir apenas a extração:

```bash
python3 -m src.benchmark_extractor
```

Esse comando executa o extrator atual nos documentos do goldenset e mostra
acertos, extrações sem correspondência e citações faltantes, no total e por nível.
Também lista os trechos faltantes e sem correspondência separados por nível e salva os detalhes em
`relatorios/extracao.json`, sem modificar os JSONs de submissão.
O pareamento usa IoU ≥ 0,5, um para um, ignorando tipo, classificação e ID.
Toda extração sem par é contada nesse diagnóstico, inclusive componentes extras
que a métrica oficial pode tolerar. O resultado mede cobertura do desenvolvimento,
não desempenho em dados inéditos. Só documentos presentes no goldenset são avaliados.

- Converter os JSONs para submissão:

```bash
python3 json_to_submission.py resultados/ submission.csv
```

- Rodar o avaliador:

```bash
python3 -m src.evaluator {arquivo_de_submissao}.csv 
```
