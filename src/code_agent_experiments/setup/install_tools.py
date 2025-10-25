"""Bootstrap script to install required CLI binaries for the project."""
from __future__ import annotations

import argparse
import collections.abc as collections_abc
from dataclasses import dataclass
import os
import platform
import shutil
import sys
import tarfile
import tempfile
from http import HTTPStatus
import http.client
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import ParseResult, urlparse

__all__ = [
    "ToolSpec",
    "detect_platform",
    "extract_binary",
    "install_tool",
    "install_tools",
    "is_binary_available",
    "place_binary",
]


@dataclass(frozen=True)
class ToolSpec:
    """Metadata describing how to install a third-party CLI binary."""

    name: str
    binary: str
    repo: str
    tag: str
    assets: dict[str, str]

    def download_url(self, platform_key: str) -> str:
        """Return the release download URL for ``platform_key``."""
        asset = self.assets.get(platform_key)
        if asset is None:
            supported = ", ".join(sorted(self.assets)) or "<none>"
            message = (
                f"Platform '{platform_key}' is not supported for {self.name}. "
                f"Supported platforms: {supported}."
            )
            raise RuntimeError(message)
        return f"https://github.com/{self.repo}/releases/download/{self.tag}/{asset}"


SYSTEM_ALIASES: dict[str, str] = {
    "linux": "linux",
    "linux2": "linux",
    "darwin": "macos",
    "windows": "windows",
    "cygwin": "windows",
}

ARCH_ALIASES: dict[str, str] = {
    "x86_64": "x86_64",
    "amd64": "x86_64",
    "x64": "x86_64",
    "aarch64": "aarch64",
    "arm64": "aarch64",
}


SUPPORTED_PLATFORMS: dict[tuple[str, str], str] = {
    ("linux", "x86_64"): "linux-x86_64",
    ("linux", "aarch64"): "linux-aarch64",
    ("macos", "x86_64"): "macos-x86_64",
    ("macos", "aarch64"): "macos-arm64",
}


TOOL_SPECS: dict[str, ToolSpec] = {
    "ripgrep": ToolSpec(
        name="ripgrep",
        binary="rg",
        repo="BurntSushi/ripgrep",
        tag="15.1.0",
        assets={
            "linux-x86_64": "ripgrep-15.1.0-x86_64-unknown-linux-musl.tar.gz",
            "linux-aarch64": "ripgrep-15.1.0-aarch64-unknown-linux-gnu.tar.gz",
            "macos-x86_64": "ripgrep-15.1.0-x86_64-apple-darwin.tar.gz",
            "macos-arm64": "ripgrep-15.1.0-aarch64-apple-darwin.tar.gz",
        },
    ),
    "fd": ToolSpec(
        name="fd",
        binary="fd",
        repo="sharkdp/fd",
        tag="v10.3.0",
        assets={
            "linux-x86_64": "fd-v10.3.0-x86_64-unknown-linux-gnu.tar.gz",
            "linux-aarch64": "fd-v10.3.0-aarch64-unknown-linux-gnu.tar.gz",
            "macos-x86_64": "fd-v10.3.0-x86_64-apple-darwin.tar.gz",
            "macos-arm64": "fd-v10.3.0-aarch64-apple-darwin.tar.gz",
        },
    ),
    "uv": ToolSpec(
        name="uv",
        binary="uv",
        repo="astral-sh/uv",
        tag="0.9.5",
        assets={
            "linux-x86_64": "uv-x86_64-unknown-linux-gnu.tar.gz",
            "linux-aarch64": "uv-aarch64-unknown-linux-gnu.tar.gz",
            "macos-x86_64": "uv-x86_64-apple-darwin.tar.gz",
            "macos-arm64": "uv-aarch64-apple-darwin.tar.gz",
        },
    ),
}

REDIRECT_STATUSES: set[HTTPStatus] = {
    HTTPStatus.MOVED_PERMANENTLY,
    HTTPStatus.FOUND,
    HTTPStatus.SEE_OTHER,
    HTTPStatus.TEMPORARY_REDIRECT,
    HTTPStatus.PERMANENT_REDIRECT,
}

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
else:  # pragma: no cover - runtime aliases for postponed annotations
    Iterable = collections_abc.Iterable
    Iterator = collections_abc.Iterator


def detect_platform() -> str:
    """Return a normalized platform identifier for the current system."""
    raw_system = platform.system().lower()
    system = SYSTEM_ALIASES.get(raw_system)
    if system is None:
        message = f"Unsupported operating system '{platform.system()}'"
        raise RuntimeError(message)

    raw_arch = platform.machine().lower()
    arch = ARCH_ALIASES.get(raw_arch)
    if arch is None:
        message = f"Unsupported architecture '{platform.machine()}'"
        raise RuntimeError(message)

    platform_key = SUPPORTED_PLATFORMS.get((system, arch))
    if platform_key is None:
        message = f"Unsupported platform combination: {system} / {arch}"
        raise RuntimeError(message)
    return platform_key


def install_tools(
    *,
    bin_dir: Path | None = None,
    tools: Iterable[str] | None = None,
    force: bool = False,
) -> list[str]:
    """Install the requested CLI tools into ``bin_dir``."""
    platform_key = detect_platform()
    resolved_bin_dir = (bin_dir or Path.home() / ".local" / "bin").expanduser()
    resolved_bin_dir.mkdir(parents=True, exist_ok=True)
    targets = list(tools or TOOL_SPECS.keys())
    messages: list[str] = []
    available = ", ".join(sorted(TOOL_SPECS))
    for tool_name in targets:
        spec = TOOL_SPECS.get(tool_name)
        if spec is None:
            message = f"Unknown tool '{tool_name}'. Available: {available}"
            raise KeyError(message)
        changed = install_tool(spec, platform_key, resolved_bin_dir, force=force)
        status = "installed" if changed else "already present"
        messages.append(f"{spec.name}: {status} -> {resolved_bin_dir / spec.binary}")
    if str(resolved_bin_dir) not in os.environ.get("PATH", ""):
        messages.append(f"[hint] Add '{resolved_bin_dir}' to your PATH to use the installed binaries.")
    return messages


def install_tool(spec: ToolSpec, platform_key: str, bin_dir: Path, *, force: bool = False) -> bool:
    """Install ``spec`` for ``platform_key`` into ``bin_dir``."""
    target_path = bin_dir / spec.binary
    if not force and is_binary_available(spec.binary, target_path):
        return False

    download_url = spec.download_url(platform_key)
    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        archive_path = temp_dir / Path(download_url).name
        _download_file(download_url, archive_path)
        extracted = extract_binary(archive_path, spec.binary, temp_dir)
        place_binary(extracted, target_path)
    return True


def is_binary_available(binary: str, target_path: Path) -> bool:
    """Return ``True`` if ``binary`` is accessible on PATH or at ``target_path``."""
    existing = shutil.which(binary)
    return existing is not None or target_path.exists()


def _download_file(url: str, destination: Path) -> None:
    """Download ``url`` into ``destination``."""
    parsed = urlparse(url)
    if parsed.scheme != "https":
        message = f"Refusing to download non-HTTPS resource: {url}"
        raise ValueError(message)
    if not parsed.netloc:
        message = f"Malformed download URL: {url}"
        raise ValueError(message)

    current = parsed
    for _ in range(5):
        connection, response = _open_https(current)
        try:
            redirect_target = _handle_redirect(url, response)
            if redirect_target is not None:
                current = redirect_target
                continue
            with destination.open("wb") as file_obj:
                for chunk in _iter_response_chunks(response):
                    file_obj.write(chunk)
            return
        finally:
            connection.close()
    message = f"Exceeded redirect limit while downloading {url}"
    raise RuntimeError(message)


def _open_https(parsed: ParseResult) -> tuple[http.client.HTTPSConnection, http.client.HTTPResponse]:
    """Open an HTTPS connection for ``parsed`` and return the response."""
    connection = http.client.HTTPSConnection(parsed.netloc, timeout=30)
    connection.request(
        "GET",
        _build_request_path(parsed),
        headers={"User-Agent": "code-agent-experiments-installer"},
    )
    response = connection.getresponse()
    return connection, response


def _build_request_path(parsed: ParseResult) -> str:
    """Construct the request target for ``parsed``."""
    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"
    return path


def _iter_response_chunks(response: http.client.HTTPResponse, chunk_size: int = 65536) -> Iterator[bytes]:
    """Yield response body chunks until the stream is exhausted."""
    while True:
        chunk = response.read(chunk_size)
        if not chunk:
            break
        yield chunk


def _handle_redirect(source_url: str, response: http.client.HTTPResponse) -> ParseResult | None:
    """Return the redirect target if one is specified, otherwise ``None``."""
    if response.status in REDIRECT_STATUSES:
        location = response.headers.get("Location")
        if not location:
            message = f"Redirect without Location header from {source_url}"
            raise RuntimeError(message)
        parsed = urlparse(location)
        if parsed.scheme != "https":
            message = f"Refusing to follow non-HTTPS redirect: {location}"
            raise ValueError(message)
        return parsed
    if response.status != HTTPStatus.OK:
        message = f"Failed to download {source_url}: HTTP {response.status}"
        raise RuntimeError(message)
    return None


def extract_binary(archive_path: Path, binary: str, work_dir: Path) -> Path:
    """Extract ``binary`` from ``archive_path`` into ``work_dir``."""
    if archive_path.suffixes[-2:] != [".tar", ".gz"] and archive_path.suffix != ".tar.gz":
        message = f"Unsupported archive format: {archive_path.name}"
        raise RuntimeError(message)
    with tarfile.open(archive_path, mode="r:gz") as archive:
        member = next(
            (item for item in archive.getmembers() if item.isfile() and Path(item.name).name == binary),
            None,
        )
        if member is None:
            message = f"Binary '{binary}' not found inside {archive_path.name}"
            raise RuntimeError(message)
        archive.extract(member, path=work_dir)
        return work_dir / member.name


def place_binary(extracted_path: Path, target_path: Path) -> None:
    """Move ``extracted_path`` into ``target_path`` and ensure executability."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists():
        target_path.unlink()
    shutil.move(str(extracted_path), target_path)
    target_path.chmod(0o755)


def _parse_args(argv: Iterable[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Install CLI tools used by the project.")
    parser.add_argument(
        "--bin-dir",
        type=Path,
        default=Path.home() / ".local" / "bin",
        help="Destination directory for the binaries (defaults to ~/.local/bin).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reinstall binaries even if they are already available on PATH.",
    )
    parser.add_argument(
        "tools",
        nargs="*",
        choices=sorted(TOOL_SPECS),
        help="Subset of tools to install (default: all).",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    """Console entry point for the installer script."""
    args = _parse_args(argv)
    messages = install_tools(bin_dir=args.bin_dir, tools=args.tools, force=args.force)
    for line in messages:
        sys.stdout.write(f"{line}\n")


if __name__ == "__main__":
    main()
