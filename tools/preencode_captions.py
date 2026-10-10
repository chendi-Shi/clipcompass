import argparse
import hashlib

import numpy as np
import pandas as pd

from clipcompass.data import write_json
from clipcompass.research_data import embed_texts, load_captions

p = argparse.ArgumentParser()
p.add_argument("--captions", default="data/raw/kuairec/kuairec_caption_category.csv")
p.add_argument("--model", default="data/models/multilingual-e5-small")
p.add_argument("--output", default="data/raw/kuairec/preencoded.npz")
a = p.parse_args()
frame, audit = load_captions(a.captions)
texts = [(str(r["caption"])+" "+str(r["manual_cover_text"])).strip() or "无文本内容" for r in frame.to_dict("records")]
features, manifest = embed_texts(texts, a.model)
manifest["caption_parse_audit"] = audit
np.savez_compressed(a.output, video_ids=frame.video_id.to_numpy(), features=features,
                    text_hashes=np.array([hashlib.sha256(text.encode()).hexdigest() for text in texts]))
write_json(a.output+".json", manifest)
print("Cached real pretrained caption vectors:", len(texts), features.shape, flush=True)
