"""Unit tests for the CLI tool installer."""
from __future__ import annotations

import collections.abc as collections_abc
import tarfile
from typing import TYPE_CHECKING
from unittest import mock

import pytest

from code_agent_experiments.setup.install_tools import (
    TOOL_SPECS,
    detect_platform,
    extract_binary,
    install_tool,
    is_binary_available,
    place_binary,
)


if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path
else:  # pragma: no cover - runtime alias for postponed annotations
    Iterator = collections_abc.Iterator

@pytest.mark.parametrize(
    ("system", "machine", "expected"),
    [
        ("Linux", "x86_64", "linux-x86_64"),
        ("Linux", "aarch64", "linux-aarch64"),
        ("Darwin", "x86_64", "macos-x86_64"),
        ("Darwin", "arm64", "macos-arm64"),
    ],
)
def test_detect_platform_supported(system: str, machine: str, expected: str) -> None:
    """Return normalized keys for supported platforms."""
    with mock.patch(
        "platform.system",
        return_value=system,
    ), mock.patch(
        "platform.machine",
        return_value=machine,
    ):
        assert detect_platform() == expected


def test_detect_platform_unsupported_system() -> None:
    """Raise an error when the operating system is not supported."""
    with mock.patch(
        "platform.system",
        return_value="Plan9",
    ), mock.patch(
        "platform.machine",
        return_value="x86_64",
    ), pytest.raises(RuntimeError):
        detect_platform()


def test_install_tool_skips_when_available(tmp_path: Path) -> None:
    """Skip installation when the binary is already present."""
    spec = TOOL_SPECS["ripgrep"]
    target = tmp_path / spec.binary
    target.write_text("echo")
    with mock.patch(
        "code_agent_experiments.setup.install_tools.shutil.which",
        return_value=str(target),
    ):
        changed = install_tool(spec, "linux-x86_64", tmp_path)
    assert not changed


def test_install_tool_downloads_and_places_binary(tmp_path: Path) -> None:
    """Download the archive and place the extracted binary."""
    spec = TOOL_SPECS["ripgrep"]

    def fake_download(url: str, destination: Path) -> None:
        assert url.endswith(spec.assets["linux-x86_64"])
        with tarfile.open(destination, "w:gz") as tar:
            binary_path = tmp_path / "rg"
            binary_path.write_text("echo")
            tar.add(binary_path, arcname="ripgrep/rg")

    with mock.patch(
        "code_agent_experiments.setup.install_tools._download_file",
        side_effect=fake_download,
    ) as download, mock.patch(
        "code_agent_experiments.setup.install_tools.place_binary",
    ) as place:
        changed = install_tool(spec, "linux-x86_64", tmp_path, force=True)
    assert changed
    download.assert_called_once()
    place.assert_called_once()


def test_extract_binary_from_tar(tmp_path: Path) -> None:
    """Extract a binary member from a tarball."""
    archive_path = tmp_path / "fd.tar.gz"
    binary_dir = tmp_path / "fd-v"
    binary_dir.mkdir()
    binary_file = binary_dir / "fd"
    binary_file.write_bytes(b"#!/bin/sh\\n")
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(binary_file, arcname="fd-v/bin/fd")

    extracted = extract_binary(archive_path, "fd", tmp_path)
    assert extracted.exists()
    assert extracted.read_bytes().startswith(b"#!/bin/sh")


def test_place_binary(tmp_path: Path) -> None:
    """Move the extracted binary into the target directory."""
    source = tmp_path / "source"
    source.write_text("#!/bin/sh")
    target = tmp_path / "bin" / "tool"
    place_binary(source, target)
    assert target.exists()
    assert target.stat().st_mode & 0o777 == 0o755


@pytest.mark.parametrize("available", [True, False])
def test_is_binary_available_checks_path(tmp_path: Path, available: bool) -> None:
    """Consider either PATH lookups or a pre-existing target file."""
    target = tmp_path / "uv"
    if available:
        target.write_text("#!/bin/sh")
    with mock.patch(
        "code_agent_experiments.setup.install_tools.shutil.which",
        return_value=None,
    ):
        assert is_binary_available("uv", target) is available


@pytest.fixture(autouse=True)
def cleanup_temp_files(tmp_path: Path) -> Iterator[None]:
    """Remove temporary files created as part of archive generation."""
    # ``tarfile`` leaves behind the source file we add to the archive. Remove it to avoid surprises.
    yield
    for child in tmp_path.iterdir():
        if child.is_file() and child.suffix != ".gz":
            child.unlink(missing_ok=True)
