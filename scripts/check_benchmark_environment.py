from __future__ import annotations

import importlib
import importlib.metadata as metadata
import subprocess
import sys
from dataclasses import dataclass


SUPPORTED_NUMPY = "1.26.4"
SUPPORTED_FAISS_CPU = "1.8.0.post1"


@dataclass
class CheckResult:
    ok: bool
    label: str
    detail: str


def main() -> int:
    results = run_checks()
    for result in results:
        status = "PASS" if result.ok else "FAIL"
        print(f"[{status}] {result.label}: {result.detail}")
    return 0 if all(result.ok for result in results) else 1


def run_checks() -> list[CheckResult]:
    results: list[CheckResult] = [
        _check_python(),
        _check_numpy(),
        _check_faiss_import(),
        _check_torch(),
    ]
    if any(result.label == "FAISS import" and result.ok for result in results):
        results.append(_check_faiss_index_operation())
    else:
        results.append(CheckResult(False, "FAISS index operation", "skipped because FAISS import failed"))
    return results


def _check_python() -> CheckResult:
    version = ".".join(str(part) for part in sys.version_info[:3])
    if sys.version_info < (3, 11):
        return CheckResult(False, "Python executable", f"{sys.executable} (Python {version}; require >=3.11)")
    return CheckResult(True, "Python executable", f"{sys.executable} (Python {version})")


def _check_numpy() -> CheckResult:
    try:
        numpy = importlib.import_module("numpy")
    except Exception as exc:
        return CheckResult(False, "NumPy", f"import failed: {exc}")
    version = str(getattr(numpy, "__version__", "unknown"))
    if version != SUPPORTED_NUMPY:
        return CheckResult(
            False,
            "NumPy",
            f"{version} is unsupported; expected {SUPPORTED_NUMPY} for the pinned FAISS environment",
        )
    return CheckResult(True, "NumPy", version)


def _check_faiss_import() -> CheckResult:
    try:
        faiss = importlib.import_module("faiss")
    except Exception as exc:
        return CheckResult(False, "FAISS import", f"failed: {exc}")
    version = str(getattr(faiss, "__version__", "unknown"))
    package_version = _package_version("faiss-cpu")
    if package_version != SUPPORTED_FAISS_CPU:
        return CheckResult(
            False,
            "FAISS import",
            f"imported faiss {version}, but faiss-cpu package is {package_version}; expected {SUPPORTED_FAISS_CPU}",
        )
    return CheckResult(True, "FAISS import", f"faiss {version}; faiss-cpu {package_version}")


def _check_faiss_index_operation() -> CheckResult:
    try:
        faiss = importlib.import_module("faiss")
        np = importlib.import_module("numpy")
        vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
        query = np.array([[1.0, 0.0]], dtype="float32")
        index = faiss.IndexFlatIP(2)
        index.add(vectors)
        scores, indices = index.search(query, 2)
        if scores.shape != (1, 2) or indices.shape != (1, 2):
            return CheckResult(False, "FAISS index operation", f"unexpected shapes scores={scores.shape} indices={indices.shape}")
        if int(indices[0][0]) != 0:
            return CheckResult(False, "FAISS index operation", f"unexpected top result index={int(indices[0][0])}")
    except Exception as exc:
        return CheckResult(False, "FAISS index operation", f"failed: {exc}")
    return CheckResult(True, "FAISS index operation", "IndexFlatIP add/search returned expected rank")


def _check_torch() -> CheckResult:
    code = (
        "import torch; "
        "print(f\"{getattr(torch, '__version__', 'unknown')}; "
        "cuda_available={bool(torch.cuda.is_available())}; "
        "cuda_version={getattr(torch.version, 'cuda', None)}\")"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )
    except Exception as exc:
        return CheckResult(False, "Torch", f"subprocess check failed: {exc}")
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        return CheckResult(False, "Torch", f"import failed: {detail}")
    return CheckResult(
        True,
        "Torch",
        result.stdout.strip(),
    )


def _package_version(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "not installed"


if __name__ == "__main__":
    raise SystemExit(main())
