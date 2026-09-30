"""Ferramentas que rodam fora do contêiner e não entram na imagem.

``download_models`` baixa os pesos do manifesto (usado pelo ``run.sh`` antes da execução),
``validate_output`` confere as saídas contra o contrato (usado pelo ``run.sh`` depois dela) e
``verify_official_data`` confere se ``data/`` é o pacote oficial.
"""
