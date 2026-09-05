import json
import sys

import pandas as pd

from kaggle_metric import avaliar


# Lemos tudo como texto para preservar os IDs e evitar valores como "123.0".
gold = pd.read_csv(
    "data/goldenset.csv",
    dtype=str,
    keep_default_na=False,
)

# O goldenset tem uma linha por citação.
# O avaliador exige uma linha por documento.
linhas = []

for (documento_id, nivel), grupo in gold.groupby(
    ["documento_id", "nivel"], sort=False
):
    citacoes = []

    for registro in grupo.to_dict("records"):
        campos = [
            registro["inicio"],
            registro["fim"],
            registro["classificacao"],
            registro["id_canonico"] or "-",
        ]
        citacoes.append(",".join(campos))

    linhas.append({
        "documento_id": documento_id,
        "nivel": int(nivel),
        "citacoes": "|".join(citacoes),
    })

solution = pd.DataFrame(linhas)

# O caminho da sua resposta é passado no terminal.
submission = pd.read_csv(
    sys.argv[1],
    dtype=str,
    keep_default_na=False,
)

resultado = avaliar(solution, submission)

print(json.dumps(resultado, indent=2, ensure_ascii=False))