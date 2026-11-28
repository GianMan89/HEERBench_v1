"""Build a deterministic, credential-free release archive without consulting Git.

Usage from the repository root::

    python scripts/package_release.py --output ../HEERBench_experiments_v0.2.zip

Only the documented source/data directories and root publication files are
eligible. Local runs, reports, environments, caches, symlinks, and credentials
are excluded. Four EMPTY key-file placeholders are generated directly in the
archive; their local counterparts are never opened. The archive's SHA-256
manifest is generated anew, excluding the manifest itself.

Do not place credentials in source or documentation. This helper protects the
specified key-file locations and naming conventions, not secrets pasted into
otherwise publishable files. Use --overwrite only to replace an existing ZIP.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path, PurePosixPath
import stat
import tempfile
import zipfile


SOURCE_DIRECTORIES = frozenset({
    "src", "tests", "docs", "configs", "scripts", "examples", "prompts",
    "data", "LICENSES",
})
ROOT_FILES = frozenset({
    "README.md", "VALIDATION.md", "CHANGELOG.md", "LICENSE", "NOTICE.md", "CITATION.cff",
    "pyproject.toml", "constraints.txt", "requirements.txt", ".gitignore",
})
EXCLUDED_DIRECTORY_NAMES = frozenset({
    "runs", "reports", "reviews", "studies", "secrets", ".git", ".venv",
    "venv", "env", ".env", "__pycache__", ".pytest_cache", ".mypy_cache",
    ".ruff_cache", ".tox", ".nox", "node_modules", "build", "dist",
    ".ipynb_checkpoints", ".cache", "htmlcov",
})
EXCLUDED_SUFFIXES = frozenset({
    ".key", ".pem", ".p12", ".pfx", ".pyc", ".pyo", ".zip", ".whl",
    ".tar", ".gz", ".bz2", ".xz", ".log", ".tmp", ".swp",
})
KEY_PLACEHOLDERS = (
    "secrets/openai.key", "secrets/anthropic.key", "secrets/google.key",
    "secrets/openrouter.key",
)
FIXED_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


def _eligible_file(path: Path) -> bool:
    """Apply case-insensitive secret/output exclusions at every depth."""
    name = path.name.lower()
    if name == ".env" or name.startswith(".env."):
        return False
    if name.startswith(".") or name.endswith("~"):
        return False
    return path.suffix.lower() not in EXCLUDED_SUFFIXES


def _source_files(root: Path):
    for name in sorted(ROOT_FILES):
        path = root / name
        if path.is_file() and not path.is_symlink():
            yield path
    guide = root / "secrets" / "README.md"
    # Follow neither a linked secrets directory nor a linked setup guide.
    if not guide.parent.is_symlink() and guide.is_file() and not guide.is_symlink():
        yield guide
    for name in sorted(SOURCE_DIRECTORIES):
        base = root / name
        if not base.is_dir() or base.is_symlink():
            continue
        for directory, directories, files in os.walk(base, followlinks=False):
            parent = Path(directory)
            directories[:] = sorted(
                child for child in directories
                if child.lower() not in EXCLUDED_DIRECTORY_NAMES
                and not child.startswith(".")
                and not child.lower().endswith(".egg-info")
                and not (parent / child).is_symlink()
            )
            for filename in sorted(files):
                path = parent / filename
                if _eligible_file(path) and path.is_file() and not path.is_symlink():
                    yield path


def _read_regular_file(path: Path) -> bytes:
    # O_NOFOLLOW also rejects a file replaced by a symlink between enumeration
    # and opening on platforms that support it. Local key files are filtered
    # before this function is called.
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError(f"Release input is not a regular file: {path}")
        return stream.read()


def _member(archive: zipfile.ZipFile, name: str, content: bytes) -> None:
    info = zipfile.ZipInfo(name, date_time=FIXED_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    archive.writestr(info, content, compress_type=zipfile.ZIP_DEFLATED,
                     compresslevel=9)


def package_release(root: str | Path, output: str | Path, *,
                    prefix: str | None = None, overwrite: bool = False) -> dict:
    """Create the safe ZIP and return non-secret release metadata.

    The default prefix is the source directory's name. Existing archives are
    preserved unless overwrite=True. This function never changes local keys.
    """
    root = Path(root).expanduser().resolve()
    output = Path(output).expanduser().absolute()
    if not root.is_dir():
        raise ValueError("Release root must be an existing directory")
    prefix = root.name if prefix is None else prefix
    if (not prefix or prefix in {".", ".."} or "/" in prefix or "\\" in prefix
            or "\x00" in prefix or PurePosixPath(prefix).is_absolute()):
        raise ValueError("Archive prefix must be one ordinary directory name")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output already exists; use --overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    entries: dict[str, bytes] = {}
    for path in _source_files(root):
        # The output may be inside the repository, even if given a non-ZIP
        # extension. Never bundle it or overwrite an eligible source file.
        if path.absolute() == output:
            raise ValueError("Release output cannot replace a publishable source file")
        entries[path.relative_to(root).as_posix()] = _read_regular_file(path)
    for name in KEY_PLACEHOLDERS:
        entries[name] = b""
    manifest = "".join(
        f"{hashlib.sha256(data).hexdigest()}  {name}\n"
        for name, data in sorted(entries.items())
    ).encode("utf-8")
    entries["RELEASE_MANIFEST.sha256"] = manifest

    descriptor, temporary = tempfile.mkstemp(prefix=".heerbench-release-", suffix=".tmp",
                                             dir=output.parent)
    os.close(descriptor)
    try:
        with zipfile.ZipFile(temporary, "w") as archive:
            for name, content in sorted(entries.items()):
                _member(archive, f"{prefix}/{name}", content)
        if overwrite:
            os.replace(temporary, output)
        else:
            # Exclusive creation prevents an intervening file from being
            # silently overwritten after the existence check above.
            with open(temporary, "rb") as source, output.open("xb") as target:
                import shutil
                shutil.copyfileobj(source, target)
        return {"archive": str(output), "prefix": prefix, "files": len(entries),
                "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "empty_key_placeholders": list(KEY_PLACEHOLDERS)}
    finally:
        Path(temporary).unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefix", help="Top-level ZIP directory, defaulting to the repository name")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = package_release(args.root, args.output, prefix=args.prefix, overwrite=args.overwrite)
    print(f"Created {result['archive']} ({result['files']} files)")
    print(f"SHA-256 {result['sha256']}")
    print("Included four empty credential placeholders; local key contents were not read.")


if __name__ == "__main__":
    main()
