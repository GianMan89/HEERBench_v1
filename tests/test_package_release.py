"""Release archives exclude local outputs and credentials, without paid calls."""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import zipfile

import pytest


_spec = importlib.util.spec_from_file_location(
    "heerbench_package_release", Path(__file__).resolve().parents[1] / "scripts/package_release.py")
packager = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(packager)


def write(root, relative, text):
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target


def test_release_excludes_credentials_and_replaces_keys_without_reading_them(tmp_path, monkeypatch):
    root = tmp_path / "project"
    marker = "NON_PUBLIC_SENTINEL_91e0a3d7"
    included = ["README.md", ".gitignore", "src/heerbench/a.py", "configs/pilot.yaml",
                "docs/guide.md", "tests/test_fixture.py", "scripts/example.py",
                "data/reference/questions.jsonl", "examples/offline_demo/report.md",
                "LICENSES/DATA.md", "secrets/README.md"]
    for name in included:
        write(root, name, "publishable\n")
    excluded = [
        ".env", ".env.local", "custom.key", "docs/nested/credential.KEY",
        "configs/certificate.pem", "docs/.env.production", "docs/.ENV",
        "runs/result.json", "reports/report.md", "studies/a/manifest.json",
        ".git/config", ".venv/lib/a.py", "src/heerbench/__pycache__/a.pyc",
        "tests/.pytest_cache/result", "docs/reports/private.md", "docs/secrets/custom.txt",
        "scripts/old_archive.zip", "scratch.txt", "private_notes/confidential.md",
    ] + list(packager.KEY_PLACEHOLDERS) + ["secrets/custom_route.txt"]
    for name in excluded:
        write(root, name, marker)
    read_regular = packager._read_regular_file
    opened = []

    def audited_read(path):
        relative = path.relative_to(root).as_posix()
        opened.append(relative)
        assert relative not in excluded
        return read_regular(path)

    monkeypatch.setattr(packager, "_read_regular_file", audited_read)
    result = packager.package_release(root, tmp_path / "release.zip")
    with zipfile.ZipFile(result["archive"]) as archive:
        names = set(archive.namelist())
        assert names == {f"project/{name}" for name in included + list(packager.KEY_PLACEHOLDERS)
                         + ["RELEASE_MANIFEST.sha256"]}
        assert all(marker.encode() not in archive.read(name) for name in names)
        for name in packager.KEY_PLACEHOLDERS:
            assert archive.read(f"project/{name}") == b""
    assert set(opened) == set(included)
    # Packaging must not overwrite a user's local key.
    assert (root / "secrets/openai.key").read_text() == marker


def test_release_manifest_and_archive_are_deterministic(tmp_path):
    root = tmp_path / "project"
    write(root, "src/heerbench/__init__.py", "VERSION = '0.2'\n")
    write(root, "RELEASE_MANIFEST.sha256", "stale manifest\n")
    first = packager.package_release(root, tmp_path / "first.zip")
    second = packager.package_release(root, tmp_path / "second.zip")
    assert first["sha256"] == second["sha256"]
    with zipfile.ZipFile(first["archive"]) as archive:
        lines = archive.read("project/RELEASE_MANIFEST.sha256").decode().splitlines()
        assert len(lines) == len(archive.namelist()) - 1
        for line in lines:
            expected, filename = line.split("  ", 1)
            assert filename != "RELEASE_MANIFEST.sha256"
            assert hashlib.sha256(archive.read(f"project/{filename}")).hexdigest() == expected


def test_release_omits_symlinks_to_external_content(tmp_path):
    root = tmp_path / "project"
    write(root, "README.md", "public")
    outside = write(tmp_path, "outside/credential.txt", "external-secret")
    (root / "docs").mkdir()
    (root / "docs/linked.txt").symlink_to(outside)
    (root / "docs/linked_dir").symlink_to(outside.parent, target_is_directory=True)
    (root / "secrets").symlink_to(outside.parent, target_is_directory=True)
    result = packager.package_release(root, tmp_path / "release.zip")
    with zipfile.ZipFile(result["archive"]) as archive:
        assert all(b"external-secret" not in archive.read(name) for name in archive.namelist())
        assert not any("linked" in name for name in archive.namelist())


def test_release_preserves_existing_archives_unless_explicitly_overwritten(tmp_path):
    root = tmp_path / "project"
    write(root, "README.md", "public")
    output = write(tmp_path, "release.zip", "existing")
    with pytest.raises(FileExistsError):
        packager.package_release(root, output)
    assert output.read_text() == "existing"
    packager.package_release(root, output, overwrite=True)
    assert zipfile.is_zipfile(output)


@pytest.mark.parametrize("prefix", ["", ".", "..", "../escape", "/absolute", "a/b", "a\\b"])
def test_release_rejects_invalid_archive_prefixes(tmp_path, prefix):
    with pytest.raises(ValueError, match="prefix"):
        packager.package_release(tmp_path, tmp_path / "release.zip", prefix=prefix)


def test_release_refuses_to_replace_a_publishable_source_file(tmp_path):
    output = write(tmp_path, "README.md", "keep this")
    with pytest.raises(ValueError, match="source"):
        packager.package_release(tmp_path, output, overwrite=True)
    assert output.read_text() == "keep this"
