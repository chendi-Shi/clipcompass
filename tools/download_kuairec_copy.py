"""Fast byte-identical public mirror; validate against official ZIP metadata."""
import hashlib
import json
import time
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import Request, urlopen

from clipcompass.data import write_json
from tools.download_kuairec import RemoteZip


def run(output="data/raw/kuairec"):
    out = Path(output)
    with urlopen("https://zenodo.org/api/records/18164998", timeout=60) as r:
        official = json.load(r)
    archive = next(f for f in official["files"] if f["key"] == "KuaiRec.zip")
    with zipfile.ZipFile(RemoteZip(archive["links"]["self"], archive["size"], out / ".ranges")) as z:
        info = next(i for i in z.infolist() if i.filename.endswith("/small_matrix.csv"))
    repo = "hiiamkik/miniOneRec-kuairec"
    with urlopen(f"https://huggingface.co/api/models/{repo}?blobs=true", timeout=60) as r:
        metadata = json.load(r)
    file = next(f for f in metadata["siblings"] if f["rfilename"] == "small_matrix.csv")
    if file["size"] != info.file_size:
        raise ValueError("Mirror size differs from official CSV")
    url = f"https://huggingface.co/{repo}/resolve/{metadata['sha']}/small_matrix.csv"
    parts = out / ".csvparts"
    parts.mkdir(exist_ok=True)
    size, chunk = file["size"], 4*1024*1024
    def fetch(start):
        end = min(start+chunk, size)-1
        path = parts / f"{start}-{end}"
        if path.exists() and path.stat().st_size == end-start+1:
            return
        for attempt in range(6):
            try:
                with urlopen(Request(url+f"?part={start}", headers={"Range": f"bytes={start}-{end}"}), timeout=90) as response:
                    if response.status != 206:
                        raise ValueError("Range not honored")
                    raw = response.read()
                if len(raw) != end-start+1:
                    raise ValueError("Incomplete range")
                path.write_bytes(raw)
                return
            except Exception:
                if attempt == 5:
                    raise
                time.sleep(2**attempt)
    starts = list(range(0, size, chunk))
    with ThreadPoolExecutor(max_workers=12) as pool:
        completed = 0
        for future in as_completed([pool.submit(fetch, start) for start in starts]):
            future.result(); completed += 1
            if completed % 10 == 0:
                print(f"Real CSV {completed}/{len(starts)} ranges", flush=True)
    digest, crc = hashlib.sha256(), 0
    target = out / "small_matrix.csv"
    with target.open("wb") as stream:
        for start in starts:
            raw = (parts / f"{start}-{min(size, start+chunk)-1}").read_bytes()
            digest.update(raw); crc = zlib.crc32(raw, crc); stream.write(raw)
    if digest.hexdigest() != file["lfs"]["sha256"] or crc != info.CRC:
        raise ValueError("CSV checksum mismatch against public mirror or official release")
    write_json(out / "download_manifest.json", {"official_record": 18164998, "official_archive_checksum": archive["checksum"],
               "download_url": url, "copy_repo": repo, "copy_revision": metadata["sha"], "size": size,
               "sha256": digest.hexdigest(), "official_zip_entry_crc32": f"{info.CRC:08x}",
               "verification": "mirror LFS SHA256 + official ZIP entry size/CRC32; full official ZIP MD5 not evaluated",
               "license": "CC-BY-SA-4.0", "caption_md5": "31bc38cdccdf75a71df137779035f8cb"})
    print(f"Verified real CSV: {size} bytes; CRC32={crc:08x}; SHA256={digest.hexdigest()}", flush=True)


if __name__ == "__main__":
    run()
