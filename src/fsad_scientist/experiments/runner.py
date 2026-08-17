from __future__ import annotations

import asyncio
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from time import monotonic

from fsad_scientist.datasets.models import DatasetViewManifest
from fsad_scientist.domain.models import ExperimentRun, utc_now
from fsad_scientist.experiments.models import CommandSpec, ExecutionRecord, SupportSetManifest
from fsad_scientist.experiments.results import ResultNormalizer

SAFE_AMBIENT_ENVIRONMENT = {
    "ALLUSERSPROFILE",
    "APPDATA",
    "CUDA_PATH",
    "CUDA_PATH_V12_8",
    "HF_HOME",
    "HOMEDRIVE",
    "HOMEPATH",
    "LOCALAPPDATA",
    "NUMBER_OF_PROCESSORS",
    "PATH",
    "PATHEXT",
    "PROGRAMDATA",
    "PROGRAMFILES",
    "PROGRAMFILES(X86)",
    "SYSTEMDRIVE",
    "SYSTEMROOT",
    "TEMP",
    "TMP",
    "USERPROFILE",
    "WINDIR",
}
SAFE_COMMAND_ENVIRONMENT = {
    "CUDA_VISIBLE_DEVICES",
    "FSAD_OUTPUT_ROOT",
    "HF_HOME",
    "PYTHONPATH",
    "TOKENIZERS_PARALLELISM",
    "TRANSFORMERS_CACHE",
}


class ExperimentRunner:
    """Execute argv-only detector commands in isolated, auditable run directories."""

    def __init__(
        self,
        *,
        project_root: Path,
        artifact_root: Path,
        normalizer: ResultNormalizer | None = None,
    ) -> None:
        self.project_root = project_root.expanduser().resolve()
        self.artifact_root = artifact_root.expanduser().resolve()
        self.normalizer = normalizer or ResultNormalizer()

    async def execute(
        self,
        run: ExperimentRun,
        command: CommandSpec,
        *,
        output_dir: Path,
        dataset_view: DatasetViewManifest,
        support_manifest: SupportSetManifest,
        timeout_seconds: float = 3600.0,
    ) -> ExecutionRecord:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        output_dir = output_dir.expanduser().resolve()
        _ensure_within(output_dir, self.artifact_root)
        output_dir.mkdir(parents=True, exist_ok=True)
        cwd = command.cwd.expanduser().resolve()
        if not cwd.exists() and cwd == output_dir:
            cwd.mkdir(parents=True, exist_ok=True)
        if not cwd.is_dir():
            raise FileNotFoundError(f"command working directory does not exist: {cwd}")
        if not (_is_within(cwd, self.project_root) or _is_within(cwd, self.artifact_root)):
            raise ValueError(f"command working directory is outside project roots: {cwd}")
        if command.method.casefold() != run.detector.casefold():
            raise ValueError("command method does not match the experiment run")

        executable = self._resolve_executable(command.executable)
        environment = self._build_environment(command.environment)
        argv = [executable, *command.args]
        stdout_path = output_dir / "stdout.log"
        stderr_path = output_dir / "stderr.log"
        record_path = output_dir / "execution.json"
        if record_path.exists() or stdout_path.exists() or stderr_path.exists():
            raise FileExistsError(
                f"run output already contains execution records; create a new run id: {output_dir}"
            )
        started = monotonic()
        record = ExecutionRecord(
            run_id=run.id,
            method=command.method,
            status="running",
            command=argv,
            cwd=str(cwd),
            output_dir=str(output_dir),
            environment_overrides=command.environment,
            dataset_view_digest=dataset_view.digest,
            support_manifest_digest=support_manifest.digest,
            code_revision=_git_revision(self.project_root),
            environment_digest=_environment_digest(),
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
        )
        _write_record(record_path, record)

        process: asyncio.subprocess.Process | None = None
        try:
            with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
                process = await asyncio.create_subprocess_exec(
                    *argv,
                    cwd=str(cwd),
                    env=environment,
                    stdout=stdout,
                    stderr=stderr,
                )
                try:
                    exit_code = await asyncio.wait_for(process.wait(), timeout=timeout_seconds)
                except TimeoutError:
                    process.kill()
                    await process.wait()
                    record.status = "timed_out"
                    record.error = f"Execution exceeded {timeout_seconds:.1f} seconds"
                    record.exit_code = process.returncode
                else:
                    record.exit_code = exit_code
                    if exit_code != 0:
                        record.status = "failed"
                        record.error = f"Detector process exited with code {exit_code}"
                    else:
                        record.normalized_result = self.normalizer.parse(
                            command.method,
                            output_dir,
                            category=run.category,
                        )
                        record.status = "succeeded"
        except asyncio.CancelledError:
            if process is not None and process.returncode is None:
                process.kill()
                await process.wait()
            record.status = "failed"
            record.error = "Execution task was cancelled"
            raise
        except Exception as exc:
            if process is not None and process.returncode is None:
                process.kill()
                await process.wait()
            record.status = "failed"
            record.error = f"{type(exc).__name__}: {exc}"
        finally:
            record.finished_at = utc_now()
            record.duration_seconds = round(monotonic() - started, 3)
            record.discovered_artifacts = sorted(
                str(path.resolve())
                for path in output_dir.rglob("*")
                if path.is_file() and path != record_path
            )
            _write_record(record_path, record)
        return record

    @staticmethod
    def _resolve_executable(executable: str) -> str:
        if Path(executable).name.casefold() not in {"python", "python.exe"}:
            raise ValueError("Only the configured Python runtime may execute detector adapters")
        return sys.executable

    @staticmethod
    def _build_environment(overrides: dict[str, str]) -> dict[str, str]:
        unsupported = sorted(set(overrides) - SAFE_COMMAND_ENVIRONMENT)
        if unsupported:
            raise ValueError(f"unsupported command environment keys: {', '.join(unsupported)}")
        environment = {
            key: value
            for key, value in os.environ.items()
            if key.upper() in SAFE_AMBIENT_ENVIRONMENT
        }
        environment.update(overrides)
        environment["PYTHONUNBUFFERED"] = "1"
        return environment


def _write_record(path: Path, record: ExecutionRecord) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(record.model_dump_json(indent=2), encoding="utf-8")
    temporary.replace(path)


def _environment_digest() -> str:
    package_names = [
        "faiss-cpu",
        "numpy",
        "pillow",
        "scikit-learn",
        "torch",
        "torchvision",
        "transformers",
    ]
    versions: dict[str, str] = {}
    for name in package_names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    payload = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": versions,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _git_revision(root: Path) -> str | None:
    try:
        revision = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(root), "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout.strip()
        )
        return f"{revision}{'-dirty' if dirty else ''}"
    except (OSError, subprocess.SubprocessError):
        return None


def _ensure_within(path: Path, parent: Path) -> None:
    if not _is_within(path, parent):
        raise ValueError(f"path escapes allowed root: {path}")


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False
