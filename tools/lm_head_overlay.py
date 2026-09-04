#!/usr/bin/env python3
"""Add the Hub pack's EXL3 lm_head to a local GLM-5.3-Flash EXL3 pack.

The serving pack keeps ``lm_head.weight`` in BF16 (1.27 GB, read once per
target verify and once per MTP draft step: ~27 ms of a ~200 ms decode step on a
DGX Spark).  turboderp/GLM-5.3-Flash-exl3 quantizes the head with EXL3
(``head_bits`` 5 on the 2.05bpw branch: 0.40 GB).  This tool fetches
``lm_head.{trellis,suh,svh,mul1}`` from that branch by HTTP range reads, writes
them into one safetensors file next to a symlink copy of the source pack, and
rewrites ``model.safetensors.index.json`` and ``config.json`` so the vllm_exl3
plugin's ``Exl3LMHeadMethod`` picks the head up
(``quantization_config.non_routed_exl3.layers["language_model.lm_head"]``).

    lm_head_overlay.py --branch 2.05bpw --src <pack> --out <pack-head5>

The output pack is a directory of symlinks plus the new file; the source pack is
not modified.
"""
import argparse
import json
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dense_overlay as D  # noqa: E402

HEAD_PARTS = ("trellis", "suh", "svh")
MARKERS = ("mul1", "mcg")
FORK_KEY = "language_model.lm_head"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--branch", default="2.05bpw")
    ap.add_argument("--src", required=True, help="local EXL3 pack (e.g. the dense overlay pack)")
    ap.add_argument("--out", required=True, help="output pack directory (symlinks + head file)")
    ap.add_argument("--repo", default="turboderp/GLM-5.3-Flash-exl3")
    args = ap.parse_args()
    base = "https://huggingface.co/%s/resolve/%s/" % (args.repo, args.branch)

    D.log("remote index %s" % base)
    idx = json.loads(D.http_bytes(base + "model.safetensors.index.json"))
    wmap = idx["weight_map"]
    names = [n for n in wmap if n.startswith("lm_head.")]
    if not any(n == "lm_head.trellis" for n in names):
        sys.exit("branch %s has no EXL3 lm_head (names: %s)" % (args.branch, names))
    files = sorted({wmap[n] for n in names})
    metas = {}
    for fn in files:
        hlen = struct.unpack("<Q", D.http_bytes(base + fn, (0, 7)))[0]
        header = json.loads(D.http_bytes(base + fn, (8, 8 + hlen - 1)).decode("utf-8"))
        for n in names:
            if wmap[n] == fn:
                metas[n] = (fn, hlen, header[n])
    marker = next((m for m in MARKERS if "lm_head." + m in metas), None)
    if marker is None:
        sys.exit("lm_head has no codebook marker tensor")
    tr = metas["lm_head.trellis"][2]
    in_f, out_f, k = tr["shape"][0] * 16, tr["shape"][1] * 16, tr["shape"][2] // 16
    D.log("lm_head: in=%d out=%d K=%d codebook=%s (%.2f GB)" % (in_f, out_f, k, marker, D.nbytes(tr) / 1e9))

    local_idx = json.load(open(os.path.join(args.src, "model.safetensors.index.json")))
    cfg = json.load(open(os.path.join(args.src, "config.json")))
    tc = cfg.get("text_config", cfg)
    if int(tc["hidden_size"]) != in_f or int(tc["vocab_size"]) != out_f:
        sys.exit("shape mismatch: pack hidden/vocab %s/%s vs head %d/%d" % (tc["hidden_size"], tc["vocab_size"], in_f, out_f))

    overlay_name = "lm_head-exl3-%dbpw.safetensors" % k
    D.link_pack(args.src, args.out, overlay_name)

    # one safetensors file: header + concatenated tensor bytes, 8-byte aligned header
    order = ["lm_head." + p for p in HEAD_PARTS] + ["lm_head." + marker]
    header, off = {}, 0
    for n in order:
        meta = metas[n][2]
        size = D.nbytes(meta)
        header[n] = {"dtype": meta["dtype"], "shape": meta["shape"], "data_offsets": [off, off + size]}
        off += size
    header["__metadata__"] = {"format": "pt", "source": "%s@%s" % (args.repo, args.branch)}
    hb = json.dumps(header, separators=(",", ":")).encode("utf-8")
    hb += b" " * ((8 - len(hb) % 8) % 8)
    out_path = os.path.join(args.out, overlay_name)
    with open(out_path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for n in order:
            fn, hlen, meta = metas[n]
            rng = D.tensor_range(hlen, meta)
            D.log("  %s <- %s bytes %d-%d" % (n, fn, rng[0], rng[1]))
            D.http_stream_to(base + fn, rng, f, D.nbytes(meta))

    # index: drop the BF16 head, add the EXL3 tensors
    new_map = {n: fn for n, fn in local_idx["weight_map"].items() if n != "lm_head.weight"}
    for n in order:
        new_map[n] = overlay_name
    new_idx = dict(local_idx)
    new_idx["weight_map"] = dict(sorted(new_map.items()))
    json.dump(new_idx, open(os.path.join(args.out, "model.safetensors.index.json"), "w"), indent=2)

    q = cfg.setdefault("quantization_config", {})
    nr = q.setdefault("non_routed_exl3", {"codebook": marker, "layers": {}})
    nr.setdefault("layers", {})[FORK_KEY] = {"bits": k}
    q["head_bits"] = k
    json.dump(cfg, open(os.path.join(args.out, "config.json"), "w"), indent=2)

    # verify the written file parses
    with open(out_path, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(hlen))
        size = os.path.getsize(out_path)
    want = 8 + hlen + max(v["data_offsets"][1] for kk, v in h.items() if kk != "__metadata__")
    if size != want:
        sys.exit("written file size %d != expected %d" % (size, want))
    D.log("LM_HEAD_OVERLAY_OK file=%s bytes=%d K=%d out=%s" % (out_path, size, k, args.out))


if __name__ == "__main__":
    main()
