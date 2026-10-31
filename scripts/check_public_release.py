"""Check the Git index for private artifacts and common identifier/secret patterns.

This is a source release gate, not a comprehensive secret scanner. Findings show
only paths, line numbers, and categories so credentials are never echoed to logs.
Use --worktree to inspect untracked source before initializing a repository.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRECTORIES = {
    ".git", "node_modules", ".cache", "__pycache__", ".pytest_cache", ".ruff_cache", "dist", "build",
}
PRIVATE_DIRECTORIES = {".aws", ".agents", ".codex", ".claude", "tests", "fixtures", "artifacts", "outputs"}
PRIVATE_SUFFIXES = {
    ".jpg", ".jpeg", ".png", ".webp", ".heic", ".gif", ".bmp", ".tif", ".tiff", ".mp4", ".mov", ".avi",
    ".pdf", ".docx", ".pptx", ".csv", ".tsv", ".jsonl", ".xlsx", ".xls", ".parquet", ".arrow",
    ".pt", ".pth", ".onnx", ".safetensors", ".pkl", ".pickle", ".joblib", ".db", ".log", ".pid",
    ".pem", ".key", ".p12", ".pfx", ".zip", ".7z", ".tar", ".gz", ".pyc",
}
PATTERNS = {
    "cloud access key": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "GitHub token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{40,})\b"),
    "API secret": re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{24,}\b"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "embedded media": re.compile(r"data:image/[a-zA-Z0-9.+-]+;base64,[A-Za-z0-9+/]{64,}"),
    "credential in URL": re.compile(r"https?://[^\s/:@]+:[^\s/@]+@"),
    "absolute workstation path": re.compile(r"(?i)\b[A-Z]:[\\/]"),
    "vehicle identifier literal": re.compile(r"\b[A-Z]{2,3}[- ]?\d{3,4}\b"),
    "mobile phone literal": re.compile(r"(?<!\d)(?:\+886[- ]?9\d{2}|09\d{2})[- ]?\d{3}[- ]?\d{3}(?!\d)"),
    "email address": re.compile(r"[A-Za-z0-9_.+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
}
TECHNICAL_IDENTIFIERS = {"SHA-256", "SHA256", "ISO-8601", "ISO8601"}
# The only reviewed image: a de-identified UI preview with the real photo removed.
# Changes require visual review and an intentional digest update.
REVIEWED_ASSETS = {
    "docs/assets/guardian-demo-redacted.png":
        "a02747cb55834424d6a74dfc59a3091febc6bb0076fd6785707fe5ec4ee706da",
}


def git(*arguments: str) -> bytes:
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={ROOT.as_posix()}", "-C", str(ROOT), *arguments],
        stderr=subprocess.PIPE,
    )


def worktree_files(directory: Path):
    for entry in sorted(directory.iterdir()):
        if entry.name in SKIP_DIRECTORIES or entry.name.startswith(".venv"):
            continue
        if entry.is_symlink():
            yield entry
        elif entry.is_dir():
            yield from worktree_files(entry)
        else:
            yield entry


def inspect(relative: Path, content: bytes) -> list[str]:
    name = relative.as_posix()
    findings = []
    if name in REVIEWED_ASSETS:
        if hashlib.sha256(content).hexdigest() == REVIEWED_ASSETS[name]:
            return []
        return [f"{name}: reviewed asset changed; repeat visual privacy review"]
    if (
        any(part in PRIVATE_DIRECTORIES for part in relative.parts)
        or relative.suffix.lower() in PRIVATE_SUFFIXES
        or ".sqlite" in relative.name.lower()
        or (relative.name.startswith(".env") and relative.name != ".env.example")
        or relative.name.startswith("test_")
    ):
        findings.append(f"{name}: private artifact or test material")
    if relative.parts[0] in {"data", "models", "logs"} and relative.name not in {"README.md", ".gitkeep"}:
        findings.append(f"{name}: runtime artifact")
    try:
        source = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        return [*findings, f"{name}: non-text file"]
    if "\x00" in source:
        findings.append(f"{name}: binary content")
    if relative.suffix.lower() == ".md" and re.search(r"[\u3400-\u9fff]", source):
        findings.append(f"{name}: Markdown must be in English")
    for number, line in enumerate(source.splitlines(), 1):
        for category, pattern in PATTERNS.items():
            for match in pattern.finditer(line):
                value = match.group()
                if category == "vehicle identifier literal" and (
                    value in TECHNICAL_IDENTIFIERS or re.fullmatch(r"ES20\d{2}", value)
                ):
                    continue
                if category == "email address" and value.rsplit("@", 1)[1] in {"example.com", "example.org", "example.net"}:
                    continue
                findings.append(f"{name}:{number}: {category}")
                break
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worktree", action="store_true", help="Scan source on disk, including untracked files")
    args = parser.parse_args()
    findings = []
    count = 0
    if args.worktree:
        entries = [(path.relative_to(ROOT), path) for path in worktree_files(ROOT)]
        for relative, path in entries:
            if path.is_symlink():
                findings.append(f"{relative.as_posix()}: symbolic link")
                continue
            count += 1
            findings.extend(inspect(relative, path.read_bytes()))
    else:
        try:
            entries = git("ls-files", "--stage", "-z").decode("utf-8").split("\x00")
        except subprocess.CalledProcessError:
            print("Cannot read the Git index. Use --worktree before initializing version control.")
            return 1
        for entry in filter(None, entries):
            metadata, name = entry.split("\t", 1)
            mode, object_id, stage = metadata.split()
            if mode not in {"100644", "100755"} or stage != "0":
                findings.append(f"{name}: unsupported index entry")
                continue
            count += 1
            findings.extend(inspect(Path(name), git("cat-file", "blob", object_id)))
    if not count:
        findings.append("No source files were inspected")
    for finding in sorted(set(findings)):
        print(finding)
    print(f"Inspected {count} files; {len(set(findings))} findings.")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
