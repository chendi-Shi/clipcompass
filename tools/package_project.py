"""Package source and executed evidence, excluding raw data and base model weights."""
import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def run(destination):
    root = Path(__file__).resolve().parents[1]
    folders = ["clipcompass", "tools", "tests", "docs", "configs", "artifacts/research"]
    files = [root / p for p in ["README.md", "pyproject.toml", "requirements-research.txt", "requirements-content.txt", ".gitignore", "artifacts/research-environment.txt", "artifacts/research-machine.json", "artifacts/research-verification.json"]]
    for folder in folders:
        files += [p for p in (root / folder).rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"]
    with ZipFile(destination, "x", compression=ZIP_DEFLATED, compresslevel=5) as archive:
        for path in sorted(set(files)):
            if path.exists():
                archive.write(path, "clipcompass/" + path.relative_to(root).as_posix())
    print(f"Packaged {len(files)} files; no raw data, virtual environment or base model weights")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="../ClipCompass-resume-project.zip")
    run(p.parse_args().output)
