"""Baixa os pesos declarados em model_manifest.json (HF id + revisão fixa) e confere o sha256.

Baixa o modelo principal e os do conjunto; com --main-only, só o principal (CACA_ENSEMBLE=0).

    python3 -m tools.download_models                 # -> modelos/<arquivo>
    python3 -m tools.download_models --dest /models

Roda ANTES do contêiner, com rede. O contêiner em si não acessa a rede: recebe a pasta
montada em /models. Só biblioteca padrão.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

MANIFEST = Path(__file__).resolve().parent.parent / "model_manifest.json"


def sha256(path: Path) -> str:
    """Calcula o sha256 de um arquivo, lendo em blocos.

    Parameters
    ----------
    path : pathlib.Path
        Arquivo a conferir.

    Returns
    -------
    str
        Resumo sha256 em hexadecimal.
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    """Baixa os pesos do manifesto que faltam ou não conferem.

    Returns
    -------
    int
        0 se todos os pesos estão presentes e conferidos; 1 se algum sha256 diverge.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dest", type=Path, default=Path("modelos"))
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--main-only", action="store_true", help="só o modelo principal (sem o conjunto)")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    models = [manifest["modelo"]] + ([] if args.main_only else manifest.get("conjunto", []))
    return max(download(model, args.dest) for model in models)


def download(model: dict, dest: Path) -> int:
    """Baixa um GGUF do manifesto e confere o sha256.

    O download vai para ``<arquivo>.part`` e só é renomeado depois de conferido; um arquivo que
    já existe e confere não é baixado de novo.

    Parameters
    ----------
    model : dict
        Entrada do manifesto (``arquivo``, ``url``, ``hf_repo``, ``revisao``, ``sha256`` e
        ``bytes``).
    dest : pathlib.Path
        Pasta de destino.

    Returns
    -------
    int
        0 se o arquivo confere; 1 se o sha256 diverge (o parcial é apagado).

    Raises
    ------
    urllib.error.URLError
        Se o download falhar (o ``run.sh`` segue só com as regras).
    """
    target = dest / model["arquivo"]
    if target.exists() and sha256(target) == model["sha256"]:
        print(f"{target} já existe e confere com o manifesto")
        return 0
    dest.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    print(f"baixando {model['hf_repo']}@{model['revisao'][:12]} -> {target}")
    with urllib.request.urlopen(model["url"], timeout=60) as response, partial.open("wb") as out:
        total, done = int(response.headers.get("Content-Length") or model["bytes"]), 0
        for block in iter(lambda: response.read(1 << 22), b""):
            out.write(block)
            done += len(block)
            print(f"\r{done / total:6.1%}", end="", file=sys.stderr)
    print(file=sys.stderr)
    got = sha256(partial)
    if got != model["sha256"]:
        partial.unlink()
        print(f"sha256 divergente: {got} != {model['sha256']}", file=sys.stderr)
        return 1
    partial.rename(target)
    print(f"ok: {target} ({model['sha256'][:12]}…)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
