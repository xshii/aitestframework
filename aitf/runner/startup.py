"""Startup/restart engine for FPGA/EMU environments.

Executes a declarative step pipeline defined in targets.yaml::

    fpga-01:
      startup:
        deps: ["fpga-firmware", "cann-toolkit"]
        steps:
          - name: 烧录 bitstream
            type: script
            script: scripts/flash.py
            args: ["--bitfile", "{dep:fpga-firmware}/top.bit"]
            port: jtag
            timeout: 120
          - name: 上传固件
            type: upload
            local: "{dep:fpga-firmware}/fw.bin"
            remote: /tmp/fw.bin
            port: ctrl
          - name: 启动服务
            type: script
            script: scripts/start.sh
            port: ctrl
            timeout: 30
          - name: 验证就绪
            type: check
            command: "curl -s http://localhost:8080/health"
            expect: "ok"
            retry: 3
            retry_delay: 5

Placeholders:
  {dep:name}       → resolved to dep install directory
  {target}         → target name
  {port:ctrl:host} → host of the named port
  {port:ctrl:port} → port number of the named port
"""

from __future__ import annotations

import logging
import re
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

SCRIPT_SUFFIXES = {".sh", ".bash", ".py"}
_INTERPRETERS = {".py": "python", ".sh": "bash", ".bash": "bash"}


class StepType:
    SCRIPT = "script"
    UPLOAD = "upload"
    CHECK = "check"


@dataclass
class StartupStep:
    """A single step in the startup pipeline."""
    name: str
    type: str                       # script | upload | check
    script: str = ""                # path to .sh/.py script
    args: list[str] = field(default_factory=list)
    command: str = ""               # for check type: shell command
    expect: str = ""                # for check type: expected stdout content
    local: str = ""                 # for upload type: local file path
    remote: str = ""                # for upload type: remote destination
    port: str = ""                  # which port to use (ctrl/jtag/...)
    timeout: int = 60
    retry: int = 1
    retry_delay: int = 5


@dataclass
class StepResult:
    """Result of executing a single step."""
    name: str
    success: bool
    output: str = ""
    error: str = ""
    duration_s: float = 0.0
    attempts: int = 1

    def to_dict(self) -> dict:
        d = asdict(self)
        # Truncate long output for API response
        for key in ("output", "error"):
            if d[key] and len(d[key]) > 1000:
                d[key] = d[key][-1000:]
        d["duration_s"] = round(d["duration_s"], 2)
        return d


@dataclass
class StartupConfig:
    """Parsed startup configuration for a target."""
    deps: list[str] = field(default_factory=list)
    steps: list[StartupStep] = field(default_factory=list)


def _dataclass_from_dict(cls, data: dict):
    """Construct a dataclass from a dict, using only known fields."""
    fields = cls.__dataclass_fields__
    return cls(**{k: data[k] for k in fields if k in data})


def parse_startup_config(raw: dict) -> StartupConfig | None:
    """Parse the 'startup' section of a target config."""
    startup = raw.get("startup")
    if not startup:
        return None
    steps = [_dataclass_from_dict(StartupStep, s) for s in startup.get("steps", [])]
    return StartupConfig(deps=startup.get("deps", []), steps=steps)


class StartupEngine:
    """Execute a startup/restart pipeline for a target.

    Steps run sequentially. On first failure, remaining steps are skipped.
    """

    def __init__(
        self,
        target_name: str,
        target_cfg: dict,
        project_root: Path,
        deps_mgr=None,
    ):
        self._target = target_name
        self._target_cfg = target_cfg
        self._root = project_root
        self._deps_mgr = deps_mgr
        self._dep_paths: dict[str, str] = {}

    def run(self, config: StartupConfig) -> list[StepResult]:
        """Execute the full startup pipeline. Returns list of StepResults."""
        results: list[StepResult] = []

        # 1. Install deps if configured
        if config.deps:
            dep_result = self._install_deps(config.deps)
            results.append(dep_result)
            if not dep_result.success:
                return results

        # 2. Execute steps in order
        for step in config.steps:
            sr = self._execute_step(step)
            results.append(sr)
            if not sr.success:
                logger.warning("Startup %s: step '%s' failed, stopping",
                               self._target, step.name)
                break

        return results

    def _install_deps(self, dep_names: list[str]) -> StepResult:
        """Install required deps and cache their paths."""
        t0 = time.time()
        if not self._deps_mgr:
            return StepResult(
                name="安装依赖",
                success=False,
                error="DepsManager not available",
                duration_s=time.time() - t0,
            )

        errors = []
        for name in dep_names:
            try:
                self._deps_mgr.install(name)
                path = self._deps_mgr.get_install_dir(name)
                if path:
                    self._dep_paths[name] = str(path)
            except Exception as exc:
                errors.append(f"{name}: {exc}")

        return StepResult(
            name="安装依赖",
            success=not errors,
            output=f"已安装: {', '.join(dep_names)}",
            error="\n".join(errors) if errors else "",
            duration_s=time.time() - t0,
        )

    def _resolve(self, text: str) -> str:
        """Replace {dep:name}, {target}, {port:name:field} placeholders."""
        def _replace(m: re.Match) -> str:
            full = m.group(1)
            if full == "target":
                return self._target
            if full.startswith("dep:"):
                dep_name = full[4:]
                return self._dep_paths.get(dep_name, f"<dep:{dep_name} not found>")
            if full.startswith("port:"):
                parts = full.split(":")
                if len(parts) == 3:
                    _, pname, pfield = parts
                    ports = self._target_cfg.get("ports", {})
                    pcfg = ports.get(pname, {})
                    return str(pcfg.get(pfield, ""))
            return m.group(0)

        return re.sub(r'\{([^}]+)\}', _replace, text)

    def _resolve_list(self, items: list[str]) -> list[str]:
        return [self._resolve(s) for s in items]

    def _execute_step(self, step: StartupStep) -> StepResult:
        """Execute a single step with retry support."""
        sr = StepResult(name=step.name, success=False, error="no attempts")
        for attempt in range(max(1, step.retry)):
            if attempt > 0:
                time.sleep(step.retry_delay)

            t0 = time.time()
            try:
                if step.type == StepType.SCRIPT:
                    sr = self._run_script(step)
                elif step.type == StepType.UPLOAD:
                    sr = self._run_upload(step)
                elif step.type == StepType.CHECK:
                    sr = self._run_check(step)
                else:
                    sr = StepResult(name=step.name, success=False,
                                    error=f"unknown step type: {step.type}")
            except Exception as exc:
                sr = StepResult(name=step.name, success=False,
                                error=str(exc))

            sr.duration_s = time.time() - t0
            sr.attempts = attempt + 1

            if sr.success:
                return sr

            logger.warning("Step '%s' attempt %d/%d failed: %s",
                           step.name, attempt + 1, step.retry, sr.error)

        return sr  # last attempt result

    def _run_script(self, step: StartupStep) -> StepResult:
        """Execute a script (.sh/.py)."""
        script_path = self._root / self._resolve(step.script)
        if not script_path.is_file():
            return StepResult(name=step.name, success=False,
                              error=f"script not found: {script_path}")

        args = self._resolve_list(step.args)

        interp = _INTERPRETERS.get(script_path.suffix)
        cmd = [interp, str(script_path)] + args if interp else [str(script_path)] + args

        r = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=step.timeout, cwd=str(self._root),
        )
        return StepResult(
            name=step.name,
            success=r.returncode == 0,
            output=r.stdout,
            error=r.stderr if r.returncode != 0 else "",
        )

    def _run_upload(self, step: StartupStep) -> StepResult:
        """Upload a file to the target (via SSH/SFTP)."""
        local_path = self._root / self._resolve(step.local)
        if not local_path.is_file():
            return StepResult(name=step.name, success=False,
                              error=f"local file not found: {local_path}")

        remote_path = self._resolve(step.remote)

        # Get SSH info from the target's port config
        port_name = step.port or "ctrl"
        ports = self._target_cfg.get("ports", {})
        pcfg = ports.get(port_name, {})
        host = pcfg.get("host", "")
        ssh_port = pcfg.get("port", 22)
        user = pcfg.get("user", "")

        if not host:
            return StepResult(name=step.name, success=False,
                              error=f"no host configured for port '{port_name}'")

        try:
            import paramiko
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            connect_kw: dict = {"hostname": host, "port": ssh_port}
            if user:
                connect_kw["username"] = user
            auth = pcfg.get("auth", {})
            if auth.get("password"):
                connect_kw["password"] = auth["password"]
            ssh.connect(**connect_kw, timeout=step.timeout)
            sftp = ssh.open_sftp()
            sftp.put(str(local_path), remote_path)
            sftp.close()
            ssh.close()
            return StepResult(
                name=step.name, success=True,
                output=f"uploaded {local_path.name} → {remote_path}",
            )
        except Exception as exc:
            return StepResult(name=step.name, success=False, error=str(exc))

    def _run_check(self, step: StartupStep) -> StepResult:
        """Run a check command and verify output."""
        command = self._resolve(step.command)
        r = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            timeout=step.timeout, cwd=str(self._root),
        )
        output = r.stdout.strip()
        if step.expect:
            ok = step.expect in output
            return StepResult(
                name=step.name, success=ok, output=output,
                error=f"expected '{step.expect}' not found in output" if not ok else "",
            )
        return StepResult(
            name=step.name,
            success=r.returncode == 0,
            output=output,
            error=r.stderr if r.returncode != 0 else "",
        )
