r"""Ponto de entrada da imagem Docker com a camada de NLP.

Sobe o servidor Ollama dentro do contêiner, confere e registra os pesos montados e roda o
pipeline (``src.main``) com a camada de NLP ligada. Normalmente é chamado pelo ``run.sh``;
direto pelo Docker::

    docker run --rm --network none --gpus all \
      -v "$PWD/data/desafio1_bracis.db:/data/ref/desafio1_bracis.db:ro" \
      -v "$PWD/modelos:/models:ro" -v "$PWD/data/txt:/data/in:ro" -v "$PWD/saida:/data/out" \
      caca-alucinacoes --input /data/in --output /data/out

Notes
-----
Nenhuma chamada externa: o servidor escuta só em 127.0.0.1 dentro do contêiner, e os pesos
vêm de ``/models`` (baixados antes, da revisão declarada em ``model_manifest.json``; o sha256 é
conferido aqui). Qualquer falha (sem pesos, hash divergente, servidor que não sobe) faz a
execução seguir só com as regras: o contêiner sempre produz as saídas. Só biblioteca padrão.

Variáveis de ambiente: ``CACA_MANIFEST`` (manifesto), ``CACA_MODEL_DIR`` (pesos, padrão
``/models``), ``CACA_ENSEMBLE=0`` (só o modelo principal), ``CACA_MAX_LOADED_MODELS``
(modelos carregados juntos, padrão 2), ``CACA_VERIFY_SHA=0`` (pula o sha256) e
``CACA_OLLAMA_LOG`` (log do servidor, padrão ``/tmp/ollama.log``).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = Path(os.environ.get("CACA_MANIFEST") or ROOT / "model_manifest.json")
HOST = "127.0.0.1:11434"
URL = f"http://{HOST}"


class LauncherError(RuntimeError):
    """Falha ao preparar a camada de NLP; quem chama segue só com as regras."""


def log(message: str) -> None:
    """Escreve uma mensagem do launcher no stderr.

    Parameters
    ----------
    message : str
        Texto da mensagem; recebe o prefixo ``[launcher]``.
    """
    print(f"[launcher] {message}", file=sys.stderr, flush=True)


def sha256(path: Path) -> str:
    """Calcula o sha256 de um arquivo, lendo em blocos de 4 MiB.

    Parameters
    ----------
    path : pathlib.Path
        Arquivo a conferir (os GGUF têm alguns GB).

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


def server_env() -> dict[str, str]:
    """Monta o ambiente do servidor Ollama.

    Returns
    -------
    dict of str to str
        Ambiente do processo atual acrescido das variáveis do Ollama: escuta só em loopback, um
        pedido por vez (o resultado não depende de lote), até ``CACA_MAX_LOADED_MODELS`` modelos
        carregados juntos (padrão 2: principal e conjunto), modelos sempre carregados e
        armazenamento em ``OLLAMA_MODELS`` (padrão ``/tmp/ollama-models``).
    """
    return os.environ | {
        "OLLAMA_HOST": HOST,                 # só loopback
        "OLLAMA_NUM_PARALLEL": "1",          # um pedido por vez: resultado não depende de lote
        "OLLAMA_MAX_LOADED_MODELS": os.environ.get("CACA_MAX_LOADED_MODELS", "2"),  # principal + conjunto
        "OLLAMA_KEEP_ALIVE": "-1",
        "OLLAMA_MODELS": os.environ.get("OLLAMA_MODELS", "/tmp/ollama-models"),
    }


def wait_server(process: subprocess.Popen, timeout_s: float = 120.0) -> None:
    """Espera o servidor responder em ``/api/version``.

    Parameters
    ----------
    process : subprocess.Popen
        Processo do ``ollama serve``.
    timeout_s : float, default 120.0
        Tempo máximo de espera, em segundos.

    Raises
    ------
    LauncherError
        Se o processo terminar antes de responder ou se o tempo acabar.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise LauncherError(f"servidor encerrou com código {process.returncode}")
        try:
            with urllib.request.urlopen(f"{URL}/api/version", timeout=2):
                return
        except (urllib.error.URLError, OSError):
            time.sleep(0.5)
    raise LauncherError("servidor não respondeu a tempo")


def find_weights(model: dict) -> Path:
    """Localiza o GGUF declarado no manifesto dentro de ``CACA_MODEL_DIR`` (padrão ``/models``).

    Aceita qualquer layout de download: o arquivo solto, ``huggingface-cli download --local-dir``
    ou o cache do Hugging Face (``snapshots/<revisão>/``). Entre vários achados, prefere o que
    está na pasta da revisão declarada.

    Parameters
    ----------
    model : dict
        Entrada do manifesto, com as chaves ``arquivo`` e ``revisao``.

    Returns
    -------
    pathlib.Path
        Caminho do GGUF.

    Raises
    ------
    LauncherError
        Se o arquivo não for encontrado.
    """
    root = Path(os.environ.get("CACA_MODEL_DIR", "/models"))
    direct = root / model["arquivo"]
    if direct.is_file():
        return direct
    found = sorted(root.rglob(model["arquivo"]), key=lambda p: model["revisao"] not in str(p)) if root.is_dir() else []
    if not found:
        raise LauncherError(f"pesos {model['arquivo']} não encontrados em {root} "
                            f"(baixe com tools/download_models.py e monte em /models)")
    return found[0]


def report_backend() -> None:
    """Informa, pelo log do servidor, qual acelerador o Ollama encontrou.

    Lê as linhas ``inference compute`` do log (``CACA_OLLAMA_LOG``) e registra a biblioteca e o
    nome da GPU (CUDA/NVIDIA ou ROCm/AMD). Sem GPU visível, avisa que o modelo rodará em CPU e que
    o orçamento de tempo tende a desligar a camada.
    """
    log_path = Path(os.environ.get("CACA_OLLAMA_LOG", "/tmp/ollama.log"))
    time.sleep(1)
    found = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines() if log_path.exists() else []:
        if "inference compute" in line:
            library = re.search(r"\blibrary=(\w+)", line)
            name = re.search(r'\b(?:name|description)="([^"]*)"', line) or re.search(r"\bname=(\S+)", line)
            found.append(f"{library.group(1) if library else '?'} {name.group(1) if name else ''}".strip())
    if not found or all(f.startswith("cpu") for f in found):
        log("nenhuma GPU visível no contêiner: o modelo roda em CPU (lento); o orçamento de tempo desliga a "
            "camada se passar de 40 s/doc. NVIDIA: --gpus all; AMD: --device /dev/kfd --device /dev/dri")
    else:
        log("acelerador: " + "; ".join(found))


def register(model: dict, modelfile: Path, name: str) -> None:
    """Confere os pesos montados e registra o modelo no servidor.

    Parameters
    ----------
    model : dict
        Entrada do manifesto (``arquivo``, ``revisao``, ``sha256`` e ``hf_repo``).
    modelfile : pathlib.Path
        Modelfile do Ollama (template e parâmetros), sem a linha ``FROM``, que é acrescentada aqui
        apontando para o GGUF montado.
    name : str
        Nome local do modelo no servidor (ex.: ``caca-nlp``).

    Raises
    ------
    LauncherError
        Se os pesos faltarem, se o sha256 divergir do manifesto (a conferência é desligável com
        ``CACA_VERIFY_SHA=0``) ou se o ``ollama create`` falhar.
    """
    gguf = find_weights(model)
    if os.environ.get("CACA_VERIFY_SHA", "1") != "0":
        start = time.monotonic()
        if (got := sha256(gguf)) != model["sha256"]:
            raise LauncherError(f"sha256 de {gguf.name} ({got[:12]}…) difere do manifesto ({model['sha256'][:12]}…)")
        log(f"pesos conferidos: {model['hf_repo']}@{model['revisao'][:12]} ({time.monotonic() - start:.0f} s)")
    with tempfile.NamedTemporaryFile("w", suffix=".Modelfile", delete=False, encoding="utf-8") as tmp:
        tmp.write(f"FROM {gguf}\n{modelfile.read_text(encoding='utf-8')}")
    created = subprocess.run(["ollama", "create", name, "-f", tmp.name], env=server_env(),
                             capture_output=True, text=True, timeout=900)
    if created.returncode != 0:
        raise LauncherError(f"ollama create {name} falhou: {created.stderr.strip()[-300:]}")


def prepare(manifest: dict) -> tuple[subprocess.Popen, str, list[str]]:
    """Sobe o servidor e registra o modelo principal e os do conjunto.

    Parameters
    ----------
    manifest : dict
        Conteúdo de ``model_manifest.json``.

    Returns
    -------
    process : subprocess.Popen
        Processo do servidor Ollama.
    principal : str
        Nome local do modelo principal.
    extras : list of str
        Nomes locais dos modelos do conjunto que subiram (vazia com ``CACA_ENSEMBLE=0``).

    Raises
    ------
    LauncherError
        Se o modelo principal não puder ser preparado. Um modelo do conjunto que falhe só fica de
        fora, com aviso.
    """
    model, runtime = manifest["modelo"], manifest["runtime"]
    find_weights(model)  # falha cedo, antes de subir o servidor
    process = subprocess.Popen(["ollama", "serve"], env=server_env(), stdout=subprocess.DEVNULL,
                               stderr=open(os.environ.get("CACA_OLLAMA_LOG", "/tmp/ollama.log"), "w"))
    try:
        wait_server(process)
        report_backend()
        register(model, ROOT / runtime["modelfile"], runtime["nome_local"])
    except (LauncherError, OSError, subprocess.SubprocessError) as error:
        process.terminate()
        raise LauncherError(str(error)) from error
    extras = []
    if os.environ.get("CACA_ENSEMBLE", "1") != "0":
        for extra in manifest.get("conjunto", []):
            try:
                register(extra, ROOT / extra["modelfile"], extra["nome_local"])
                extras.append(extra["nome_local"])
            except (LauncherError, OSError, subprocess.SubprocessError) as error:
                log(f"modelo do conjunto {extra['hf_repo']} indisponível ({error}): seguindo sem ele")
    return process, runtime["nome_local"], extras


def main(argv: list[str] | None = None) -> int:
    """Prepara a camada de NLP e roda o pipeline.

    Parameters
    ----------
    argv : list of str, optional
        Argumentos do pipeline (``--input``, ``--output``...). Padrão: ``sys.argv[1:]``.

    Returns
    -------
    int
        Código de saída do pipeline. Se a camada não puder ser preparada, o pipeline roda só com
        as regras.
    """
    from .main import main as pipeline

    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        process, name, extras = prepare(manifest)
    except (LauncherError, OSError, ValueError, KeyError) as error:
        log(f"{error}: seguindo só com as regras")
        return pipeline(argv)
    log(f"camada de NLP ligada com {name}" + (f" + {', '.join(extras)} (conjunto)" if extras else ""))
    try:
        return pipeline(argv + ["--nlp", "--nlp-model", name, "--nlp-url", URL]
                        + [arg for extra in extras for arg in ("--nlp-extra-model", extra)])
    finally:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()


if __name__ == "__main__":
    sys.exit(main())
