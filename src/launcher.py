"""Entrypoint da imagem com NLP: sobe o Ollama local, registra o GGUF montado e roda o pipeline.

    docker run --rm --network none --gpus all \\
      -v "$PWD/data/desafio1_bracis.db:/data/ref/desafio1_bracis.db:ro" -v "$PWD/modelos:/models:ro" \\
      -v "$PWD/data/txt:/data/in:ro" -v "$PWD/saida:/data/out" \\
      caca-alucinacoes-nlp --input /data/in --output /data/out

Nenhuma chamada externa: o servidor escuta só em 127.0.0.1 dentro do contêiner, e os pesos vêm
de /models (baixados antes, da revisão declarada em model_manifest.json; o sha256 é conferido
aqui). Qualquer falha (sem pesos, hash divergente, servidor que não sobe) cai para a versão só com
regras: o contêiner sempre produz as saídas. Só biblioteca padrão.
"""

from __future__ import annotations

import hashlib
import json
import os
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
    pass


def log(message: str) -> None:
    print(f"[launcher] {message}", file=sys.stderr, flush=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def server_env() -> dict[str, str]:
    return os.environ | {
        "OLLAMA_HOST": HOST,                 # só loopback
        "OLLAMA_NUM_PARALLEL": "1",          # um pedido por vez: resultado não depende de lote
        "OLLAMA_MAX_LOADED_MODELS": "1",
        "OLLAMA_KEEP_ALIVE": "-1",
        "OLLAMA_MODELS": os.environ.get("OLLAMA_MODELS", "/tmp/ollama-models"),
    }


def wait_server(process: subprocess.Popen, timeout_s: float = 120.0) -> None:
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
    """O GGUF declarado dentro de CACA_MODEL_DIR (padrão /models), em qualquer layout de download:
    pasta simples, `huggingface-cli download --local-dir` ou cache do HF (snapshots/<revisão>/)."""
    root = Path(os.environ.get("CACA_MODEL_DIR", "/models"))
    direct = root / model["arquivo"]
    if direct.is_file():
        return direct
    found = sorted(root.rglob(model["arquivo"]), key=lambda p: model["revisao"] not in str(p)) if root.is_dir() else []
    if not found:
        raise LauncherError(f"pesos {model['arquivo']} não encontrados em {root} "
                            f"(baixe com tools/baixar_modelo.py e monte em /models)")
    return found[0]


def prepare(manifest: dict) -> tuple[subprocess.Popen, str]:
    """Confere os pesos, sobe o servidor e registra o modelo. Devolve (processo, nome local)."""
    model, runtime = manifest["modelo"], manifest["runtime"]
    gguf = find_weights(model)
    if os.environ.get("CACA_VERIFY_SHA", "1") != "0":
        start = time.monotonic()
        if (got := sha256(gguf)) != model["sha256"]:
            raise LauncherError(f"sha256 dos pesos ({got[:12]}…) difere do manifesto ({model['sha256'][:12]}…)")
        log(f"pesos conferidos: {model['hf_repo']}@{model['revisao'][:12]} ({time.monotonic() - start:.0f} s)")
    process = subprocess.Popen(["ollama", "serve"], env=server_env(), stdout=subprocess.DEVNULL,
                               stderr=open(os.environ.get("CACA_OLLAMA_LOG", "/tmp/ollama.log"), "w"))
    try:
        wait_server(process)
        modelfile = (ROOT / runtime["modelfile"]).read_text(encoding="utf-8")
        with tempfile.NamedTemporaryFile("w", suffix=".Modelfile", delete=False, encoding="utf-8") as tmp:
            tmp.write(f"FROM {gguf}\n{modelfile}")
        created = subprocess.run(["ollama", "create", runtime["nome_local"], "-f", tmp.name], env=server_env(),
                                 capture_output=True, text=True, timeout=900)
        if created.returncode != 0:
            raise LauncherError(f"ollama create falhou: {created.stderr.strip()[-300:]}")
    except (LauncherError, OSError, subprocess.SubprocessError) as error:
        process.terminate()
        raise LauncherError(str(error)) from error
    return process, runtime["nome_local"]


def main(argv: list[str] | None = None) -> int:
    from .main import main as pipeline

    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        process, name = prepare(manifest)
    except (LauncherError, OSError, ValueError, KeyError) as error:
        log(f"{error}: seguindo só com as regras")
        return pipeline(argv)
    log(f"camada de NLP ligada com {name}")
    try:
        return pipeline(argv + ["--nlp", "--nlp-model", name, "--nlp-url", URL])
    finally:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()


if __name__ == "__main__":
    sys.exit(main())
