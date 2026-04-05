import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], cwd: Path | None = None) -> None:
    printable = " ".join(cmd)
    print(f"\n[cmd] {printable}", flush=True)
    subprocess.run(cmd, check=True, cwd=cwd)


def run_with_fallback(primary: list[str], fallback: list[str], cwd: Path) -> None:
    try:
        run(primary, cwd=cwd)
    except subprocess.CalledProcessError:
        print("[gpu-ci] Primary install command failed, retrying with fallback.", flush=True)
        run(fallback, cwd=cwd)


def _torch_has_cuda() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False




def detect_cuda_index_url() -> str | None:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=cuda_version", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.STDOUT,
        )
        cuda_version = output.strip().splitlines()[0].strip()
    except Exception:
        return None

    if cuda_version.startswith("12.4"):
        return "https://download.pytorch.org/whl/cu124"
    if cuda_version.startswith("12.1") or cuda_version.startswith("12.2") or cuda_version.startswith("12.3"):
        return "https://download.pytorch.org/whl/cu121"
    if cuda_version.startswith("11.8"):
        return "https://download.pytorch.org/whl/cu118"

    return None

def ensure_torch_with_cuda() -> None:
    if _torch_has_cuda():
        print("[gpu-ci] torch + CUDA check: OK", flush=True)
        return

    index_url = detect_cuda_index_url()
    if index_url:
        print(f"[gpu-ci] torch missing or CUDA unavailable; installing torch from {index_url}.", flush=True)
        run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--upgrade",
                "torch",
                "--index-url",
                index_url,
            ]
        )
    else:
        print("[gpu-ci] CUDA version not detected from nvidia-smi; installing default torch wheel.", flush=True)
        run([sys.executable, "-m", "pip", "install", "--upgrade", "torch"])

    if not _torch_has_cuda():
        raise RuntimeError("[gpu-ci] torch is installed but CUDA is still unavailable after fallback install.")


REPO = "__STACKFORMER_REPO__"
REF = "__STACKFORMER_REF__"
WORKDIR = Path("/kaggle/working")
SRC = WORKDIR / "Stackformer"

print("[gpu-ci] ===== Stackformer GPU test execution started =====", flush=True)
run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
run(["git", "clone", "--depth", "1", REPO, str(SRC)])
run(["git", "fetch", "--depth", "1", "origin", REF], cwd=SRC)
run(["git", "checkout", REF], cwd=SRC)

run_with_fallback(
    [sys.executable, "-m", "pip", "install", "-e", ".[dev]"],
    [sys.executable, "-m", "pip", "install", "-e", "."],
    cwd=SRC,
)
run([sys.executable, "-m", "pip", "install", "pytest==8.2.0"])

ensure_torch_with_cuda()

run([sys.executable, "-c", "import platform; print('Python:', platform.python_version())"])
run([sys.executable, "-c", "import torch; print('Torch:', torch.__version__)" ])
run([sys.executable, "-c", "import torch; print('CUDA available:', torch.cuda.is_available())"])
run([sys.executable, "-c", "import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"])

# Strong GPU runtime validation: allocate and compute on CUDA.
run(
    [
        sys.executable,
        "-c",
        (
            "import torch; "
            "assert torch.cuda.is_available(), 'CUDA is required for GPU CI but is not available.'; "
            "a=torch.randn(64,64, device='cuda'); "
            "b=torch.randn(64,64, device='cuda'); "
            "c=a@b; "
            "assert c.is_cuda and torch.isfinite(c).all().item(); "
            "print('GPU runtime tensor check: OK')"
        ),
    ]
)

print("[gpu-ci] Running full pytest suite (CPU + GPU-marked tests).", flush=True)
run(["pytest", "-q", "--maxfail=1"], cwd=SRC)
print("[gpu-ci] ===== Stackformer GPU test execution succeeded =====", flush=True)
