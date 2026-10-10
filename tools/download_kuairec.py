"""Download official release, verify published checksum, extract only research inputs."""
import argparse
import hashlib
import io
import json
import shutil
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen

from clipcompass.data import write_json


class RemoteZip(io.RawIOBase):
    """Seekable HTTP range reader. Avoid downloading unrelated big_matrix logs."""
    def __init__(self, url, size, cache):
        self.url, self.size, self.position = url, size, 0
        self.cache = Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)

    def seekable(self):
        return True

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        return self.position

    def tell(self):
        return self.position

    def read(self, n=-1):
        n = min(self.size-self.position, n if n >= 0 else self.size-self.position)
        if n <= 0:
            return b""
        def fetch(bounds):
            start, end = bounds
            cache_file = self.cache / f"{start}-{end}.bin"
            if cache_file.exists() and cache_file.stat().st_size == end-start+1:
                return cache_file.read_bytes()
            url = self.url + f"?range_start={start}&range_end={end}"
            request = Request(url, headers={"Range": f"bytes={start}-{end}"})
            for attempt in range(5):
                try:
                    with urlopen(request, timeout=90) as response:
                        if response.status != 206:
                            raise RuntimeError("Server must support byte ranges")
                        result = response.read()
                    if len(result) != end-start+1:
                        raise RuntimeError("Incomplete range")
                    break
                except Exception:
                    if attempt == 4:
                        raise
                    time.sleep(2**attempt)
            cache_file.write_bytes(result)
            return result
        chunks = [(start, min(start+1024*1024, self.position+n)-1) for start in range(self.position, self.position+n, 1024*1024)]
        with ThreadPoolExecutor(max_workers=16) as pool:
            result = b"".join(pool.map(fetch, chunks))
        if len(result) != n:
            raise RuntimeError("Incomplete range response")
        self.position += len(result)
        return result


def download(output, range_only=False):
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    with urlopen("https://zenodo.org/api/records/18164998", timeout=60) as response:
        record = json.load(response)
    files = {f["key"]: f for f in record["files"]}
    names = ["kuairec_caption_category.csv"] if range_only else ["KuaiRec.zip", "kuairec_caption_category.csv"]
    for name in names:
        entry = files[name]
        target = out / name
        if not target.exists() or target.stat().st_size != entry["size"]:
            tmp = out / (name + ".part")
            with urlopen(entry["links"]["self"], timeout=90) as response, tmp.open("wb") as stream:
                copied, last = 0, time.monotonic()
                while True:
                    block = response.read(1024*1024)
                    if not block:
                        break
                    stream.write(block); copied += len(block)
                    if time.monotonic()-last > 10:
                        print(f"{name}: {copied/entry['size']:.1%}", flush=True)
                        last = time.monotonic()
            tmp.replace(target)
        digest = hashlib.md5()
        with target.open("rb") as stream:
            for block in iter(lambda: stream.read(1024*1024), b""):
                digest.update(block)
        if "md5:" + digest.hexdigest() != entry["checksum"]:
            raise ValueError(f"Checksum mismatch: {name}")
        print(f"Verified {name}: {entry['checksum']}", flush=True)
    source_zip = RemoteZip(files["KuaiRec.zip"]["links"]["self"], files["KuaiRec.zip"]["size"], out / ".ranges") if range_only else out / "KuaiRec.zip"
    extracted = []
    with zipfile.ZipFile(source_zip) as archive:
        print("Archive members:", [(m.filename, m.compress_size) for m in archive.infolist()], flush=True)
        for member in archive.namelist():
            basename = Path(member).name
            if basename in {"small_matrix.csv", "item_categories.csv"}:
                target = out / basename  # flatten known filenames; no traversal paths
                info = archive.getinfo(member)
                partial = out / (basename + ".part")
                with archive.open(member) as source, partial.open("wb") as destination:
                    copied, last = 0, time.monotonic()
                    while True:
                        block = source.read(16*1024*1024)
                        if not block:
                            break
                        destination.write(block); copied += len(block)
                        if time.monotonic()-last > 10:
                            print(f"Extracting {basename}: {copied/info.file_size:.1%}", flush=True)
                            last = time.monotonic()
                partial.replace(target)
                extracted.append({"name": basename, "bytes": info.file_size, "crc32": f"{info.CRC:08x}"})
    write_json(out / "download_manifest.json", {"record_id": 18164998, "url": "https://zenodo.org/records/18164998",
               "files": [files[n] for n in ["KuaiRec.zip", "kuairec_caption_category.csv"]],
               "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "license": "CC-BY-SA-4.0",
               "range_only": range_only, "zip_verification": "per-entry ZIP CRC32 only" if range_only else "full published MD5",
               "extracted": extracted})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="data/raw/kuairec")
    p.add_argument("--range-only", action="store_true")
    a = p.parse_args()
    download(a.output, a.range_only)
