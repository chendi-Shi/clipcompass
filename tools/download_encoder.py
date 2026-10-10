"""Download pinned official encoder files with resumable concurrent ranges and SHA checks."""
import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import Request, urlopen

from clipcompass.data import write_json


def download(output, model="intfloat/multilingual-e5-small", revision=None):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    revision_path = f"/revision/{revision}" if revision else ""
    with urlopen(f"https://huggingface.co/api/models/{model}{revision_path}?blobs=true", timeout=60) as response:
        metadata = json.load(response)
    revision = metadata["sha"]
    chosen = {"config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "model.safetensors"}
    files = [f for f in metadata["siblings"] if f["rfilename"] in chosen]
    for entry in files:
        name = entry["rfilename"]
        url = f"https://huggingface.co/{model}/resolve/{revision}/{name}"
        target = out / name
        if name != "model.safetensors":
            if not target.exists():
                with urlopen(url, timeout=60) as response:
                    target.write_bytes(response.read())
            # Verify small Git blobs too (Git SHA1 includes the blob header).
            raw = target.read_bytes()
            valid = hashlib.sha256(raw).hexdigest() == entry["lfs"]["sha256"] if "lfs" in entry else hashlib.sha1(f"blob {len(raw)}\0".encode()+raw).hexdigest() == entry["blobId"]
            if not valid:
                raise ValueError(f"Git blob mismatch: {name}")
            continue
        size = entry["size"]
        chunk_size = 4*1024*1024
        parts = out / ".parts"
        parts.mkdir(exist_ok=True)
        def fetch(start):
            end = min(size, start+chunk_size)-1
            part = parts / f"{start}-{end}"
            if part.exists() and part.stat().st_size == end-start+1:
                return part
            for attempt in range(6):
                try:
                    request = Request(url+f"?range={start}-{end}", headers={"Range": f"bytes={start}-{end}"})
                    with urlopen(request, timeout=90) as response:
                        if response.status != 206:
                            raise ValueError("Range not honored")
                        raw = response.read()
                    if len(raw) != end-start+1:
                        raise ValueError("Incomplete weight range")
                    part.write_bytes(raw)
                    return part
                except Exception:
                    if attempt == 5:
                        raise
                    time.sleep(2**attempt)
        starts = list(range(0, size, chunk_size))
        with ThreadPoolExecutor(max_workers=12) as pool:
            completed = 0
            for future in as_completed([pool.submit(fetch, start) for start in starts]):
                future.result(); completed += 1
                if completed % 10 == 0:
                    print(f"Encoder weights {completed}/{len(starts)} ranges", flush=True)
        digest = hashlib.sha256()
        with target.open("wb") as stream:
            for start in starts:
                raw = (parts / f"{start}-{min(size, start+chunk_size)-1}").read_bytes()
                digest.update(raw); stream.write(raw)
        if digest.hexdigest() != entry["lfs"]["sha256"]:
            raise ValueError("Encoder weight SHA256 mismatch")
        print("Verified encoder SHA256", digest.hexdigest(), flush=True)
    write_json(out / "encoder_download_manifest.json", {"model": model, "revision": revision, "files": files,
                                                        "verification": "official Git blob SHA1 + LFS SHA256"})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="data/models/multilingual-e5-small")
    p.add_argument("--model", default="intfloat/multilingual-e5-small")
    p.add_argument("--revision", help="Model commit from the recorded experiment manifest")
    a = p.parse_args()
    download(a.output, a.model, a.revision)
