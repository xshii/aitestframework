"""SyncClient — REST client for communicating with a remote aitf server."""

from __future__ import annotations

import io
import json
import logging
import urllib.error
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

_TIMEOUT = 120  # seconds


class SyncError(Exception):
    """Raised when a sync operation fails."""


class SyncClient:
    """Thin wrapper around urllib to call remote aitf server APIs."""

    def __init__(self, server_url: str) -> None:
        self.base = server_url.rstrip("/")

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _get(self, path: str, *, timeout: int = _TIMEOUT) -> bytes:
        url = f"{self.base}{path}"
        try:
            resp = urllib.request.urlopen(url, timeout=timeout)
            return resp.read()
        except urllib.error.URLError as exc:
            raise SyncError(f"GET {url} failed: {exc}") from exc

    def _get_json(self, path: str, **kw) -> dict | list:
        data = self._get(path, **kw)
        return json.loads(data)

    def _post_json(self, path: str, body: dict, *, timeout: int = _TIMEOUT) -> dict:
        url = f"{self.base}{path}"
        payload = json.dumps(body).encode()
        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            resp = urllib.request.urlopen(req, timeout=timeout)
            return json.loads(resp.read())
        except urllib.error.URLError as exc:
            raise SyncError(f"POST {url} failed: {exc}") from exc

    def _post_multipart(self, path: str, fields: dict[str, str],
                        files: dict[str, tuple[str, bytes, str]],
                        *, timeout: int = _TIMEOUT) -> dict:
        """POST multipart/form-data with fields and files."""
        url = f"{self.base}{path}"
        boundary = "----AitfSyncBoundary"
        parts: list[bytes] = []

        for name, value in fields.items():
            parts.append(
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
                f"{value}\r\n".encode()
            )

        for name, (filename, data, content_type) in files.items():
            parts.append(
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
                f"Content-Type: {content_type}\r\n\r\n".encode()
                + data + b"\r\n"
            )

        parts.append(f"--{boundary}--\r\n".encode())
        body = b"".join(parts)

        req = urllib.request.Request(
            url, data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        try:
            resp = urllib.request.urlopen(req, timeout=timeout)
            return json.loads(resp.read())
        except urllib.error.URLError as exc:
            raise SyncError(f"POST {url} failed: {exc}") from exc

    def _download_to(self, path: str, dest: Path,
                     *, timeout: int = 300) -> Path:
        """Download a file and save to *dest*."""
        data = self._get(path, timeout=timeout)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return dest

    # ------------------------------------------------------------------
    # health
    # ------------------------------------------------------------------

    def ping(self) -> bool:
        try:
            self._get_json("/api/sync/ping", timeout=5)
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Golden — download
    # ------------------------------------------------------------------

    def golden_list_models(self) -> list[str]:
        return self._get_json("/api/golden/models")

    def golden_list_versions(self, model: str) -> list[str]:
        return self._get_json(f"/api/golden/{model}/versions")

    def golden_list_operators(self, model: str, version: str) -> list[str]:
        return self._get_json(f"/api/golden/{model}/{version}/operators")

    def golden_download_operator(self, model: str, version: str,
                                 operator: str, dest: Path) -> Path:
        """Download operator golden data as zip, extract to dest."""
        import zipfile

        zip_data = self._get(
            f"/api/golden/{model}/{version}/{operator}/download",
            timeout=300,
        )
        buf = io.BytesIO(zip_data)
        dest.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(buf, "r") as zf:
            zf.extractall(dest, filter="data")
        logger.info("Downloaded golden %s/%s/%s → %s", model, version, operator, dest)
        return dest

    def golden_download_version(self, model: str, version: str,
                                dest: Path) -> Path:
        """Download all operators for a model/version."""
        import zipfile

        zip_data = self._get(
            f"/api/golden/{model}/{version}/download",
            timeout=600,
        )
        buf = io.BytesIO(zip_data)
        dest.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(buf, "r") as zf:
            zf.extractall(dest, filter="data")
        logger.info("Downloaded golden %s/%s → %s", model, version, dest)
        return dest

    # ------------------------------------------------------------------
    # Golden — upload
    # ------------------------------------------------------------------

    def golden_upload_files(self, model: str, version: str,
                            operator: str, file_paths: list[Path]) -> dict:
        """Upload golden data files to the server."""
        fields = {"model": model, "version": version, "operator": operator}
        files = {}
        for i, fp in enumerate(file_paths):
            key = "file" if i == 0 else f"file_{i}"
            files[key] = (fp.name, fp.read_bytes(), "application/octet-stream")

        # Server expects all files under field name "file"
        # Use the existing /api/golden/upload which accepts multiple files
        # But urllib multipart needs unique keys, so we build raw body
        return self._upload_golden_raw(fields, file_paths)

    def _upload_golden_raw(self, fields: dict, file_paths: list[Path]) -> dict:
        """Upload multiple files with the same field name 'file'."""
        url = f"{self.base}/api/golden/upload"
        boundary = "----AitfSyncUpload"
        parts: list[bytes] = []

        for name, value in fields.items():
            parts.append(
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
                f"{value}\r\n".encode()
            )

        for fp in file_paths:
            parts.append(
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="file"; filename="{fp.name}"\r\n'
                f"Content-Type: application/octet-stream\r\n\r\n".encode()
                + fp.read_bytes() + b"\r\n"
            )

        parts.append(f"--{boundary}--\r\n".encode())
        body = b"".join(parts)

        req = urllib.request.Request(
            url, data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        try:
            resp = urllib.request.urlopen(req, timeout=300)
            return json.loads(resp.read())
        except urllib.error.URLError as exc:
            raise SyncError(f"Golden upload failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Deps — download / upload
    # ------------------------------------------------------------------

    def deps_download_archive(self, name: str, version: str,
                              dest: Path) -> Path:
        """Download a dependency archive from the server."""
        return self._download_to(
            f"/api/sync/deps/{name}/{version}",
            dest / f"{name}-{version}.tar.gz",
            timeout=600,
        )

    def deps_upload_archive(self, name: str, version: str,
                            archive_path: Path) -> dict:
        """Upload a dependency archive to the server."""
        return self._post_multipart(
            "/api/sync/deps/upload",
            fields={"name": name, "version": version},
            files={"file": (archive_path.name, archive_path.read_bytes(),
                            "application/gzip")},
            timeout=600,
        )

    def bundle_download(self, name: str, dest: Path) -> Path:
        """Download a bundle archive from the server."""
        return self._download_to(
            f"/api/sync/deps/bundle/{name}",
            dest / f"{name}.tar.gz",
            timeout=600,
        )

    def bundle_upload(self, name: str, archive_path: Path) -> dict:
        """Upload a bundle archive to the server."""
        return self._post_multipart(
            "/api/sync/deps/bundle/upload",
            fields={"name": name},
            files={"file": (archive_path.name, archive_path.read_bytes(),
                            "application/gzip")},
            timeout=600,
        )

    # ------------------------------------------------------------------
    # Execution results
    # ------------------------------------------------------------------

    def upload_execution(self, data: dict) -> bool:
        """Upload execution results to the server."""
        try:
            result = self._post_json("/api/sync/results", data, timeout=30)
            return result.get("ok", False)
        except SyncError:
            return False

    def list_executions(self, limit: int = 50) -> list[dict]:
        """Fetch execution history from the server."""
        return self._get_json(f"/api/sync/results?limit={limit}")

    def get_execution_detail(self, execution_id: str) -> dict | None:
        """Fetch a single execution with case results."""
        try:
            return self._get_json(f"/api/sync/results/{execution_id}")
        except SyncError:
            return None

    # ------------------------------------------------------------------
    # Test plans
    # ------------------------------------------------------------------

    def list_plans(self) -> list[dict]:
        """List test plans on the server."""
        return self._get_json("/api/sync/plans")

    def download_plan(self, filename: str) -> dict:
        """Download a test plan YAML as dict."""
        return self._get_json(f"/api/sync/plans/{filename}")

    def upload_plan(self, filename: str, plan_data: dict) -> bool:
        """Upload a test plan to the server."""
        try:
            result = self._post_json(
                "/api/sync/plans/upload",
                {"filename": filename, "plan": plan_data},
            )
            return result.get("ok", False)
        except SyncError:
            return False

    # ------------------------------------------------------------------
    # Test plans with golden check
    # ------------------------------------------------------------------

    def golden_version_fingerprint(self, model: str, version: str) -> dict:
        """Get operator-level fingerprints for a golden version on the remote server."""
        return self._get_json(f"/api/golden/{model}/{version}/fingerprint")

    def upload_plan_with_golden_check(self, filename: str, plan_data: dict,
                                       local_fingerprints: dict) -> dict:
        """Upload testplan and check golden availability on remote.

        Args:
            filename: testplan filename
            plan_data: testplan YAML content as dict
            local_fingerprints: {"{model}/{version}": {"operators": {op: fp}}}

        Returns dict with:
            - ok: bool
            - golden_status: [{model, version, status, missing_ops, diff_ops}]
              status: "match" | "missing" | "partial" | "diff"
        """
        return self._post_json("/api/sync/plans/upload_with_check", {
            "filename": filename,
            "plan": plan_data,
            "golden_fingerprints": local_fingerprints,
        })

    def golden_upload_zip(self, model: str, version: str,
                          operator: str | None,
                          zip_data: bytes, zip_name: str = "golden.zip") -> dict:
        """Upload a golden zip archive to the remote server.

        Uses upload_version endpoint if operator is None,
        otherwise upload_operator.
        """
        fields = {"model": model, "version": version}
        if operator:
            fields["operator"] = operator
            endpoint = "/api/sync/golden/upload_operator"
        else:
            endpoint = "/api/sync/golden/upload_version"

        return self._post_multipart(
            endpoint,
            fields=fields,
            files={"file": (zip_name, zip_data, "application/zip")},
            timeout=300,
        )
