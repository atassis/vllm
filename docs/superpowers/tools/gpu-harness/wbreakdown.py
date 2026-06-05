import json, struct, glob, os, re
from collections import defaultdict
D="/models/Qwen3.5-27B-AWQ"
def header(f):
    with open(f,'rb') as fh:
        n=struct.unpack('<Q',fh.read(8))[0]
        return json.loads(fh.read(n))
agg=defaultdict(lambda:[0,0])  # category -> [bytes, count]
layer_types=json.load(open(f"{D}/config.json")).get("text_config",{}).get("layer_types",[])
def cat(name):
    if name=="__metadata__": return None
    if "mtp" in name or "draft" in name: return "MTP/draft"
    if "embed_tokens" in name: return "embed_tokens"
    if "lm_head" in name: return "lm_head"
    m=re.search(r"layers\.(\d+)\.", name)
    if m:
        i=int(m.group(1))
        if "mtp" in name: return "MTP/draft"
        lt=layer_types[i] if i<len(layer_types) else "?"
        return f"target.{lt}"
    if "norm" in name: return "final_norm/other"
    return "other"
for f in sorted(glob.glob(f"{D}/*.safetensors")):
    h=header(f)
    for name,meta in h.items():
        if name=="__metadata__": continue
        st,en=meta["data_offsets"]; sz=en-st
        c=cat(name)
        if c: agg[c][0]+=sz; agg[c][1]+=1
total=sum(v[0] for v in agg.values())
print(f"{'component':<26}{'GiB':>8}{'%':>7}{'tensors':>9}")
for k,(b,n) in sorted(agg.items(), key=lambda x:-x[1][0]):
    print(f"{k:<26}{b/2**30:>8.3f}{100*b/total:>6.1f}%{n:>9}")
print(f"{'TOTAL':<26}{total/2**30:>8.3f}")
# per-layer-type averages
gdn=agg.get("target.linear_attention",[0,0]); attn=agg.get("target.full_attention",[0,0])
nz=lambda x: x if x else 1
print(f"\nper-GDN-layer avg:  {gdn[0]/nz(48)/2**20:.1f} MiB  (48 layers)")
print(f"per-attn-layer avg: {attn[0]/nz(16)/2**20:.1f} MiB  (16 layers)")
