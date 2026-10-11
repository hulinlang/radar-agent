#!/usr/bin/env python3
"""Dump GGUF tensor-type distribution (counts + bytes) for one file."""
import struct
import sys
from collections import defaultdict

TYPES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
         8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K",
         13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS",
         18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S",
         22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32",
         27: "I64", 28: "F64", 29: "IQ1_M"}

# bytes per block / values per block for types we care about
BLOCK = {0: (4, 1), 1: (2, 1), 12: (144, 256), 14: (210, 256), 8: (34, 32)}


def read_string(f):
    (n,) = struct.unpack("<Q", f.read(8))
    return f.read(n).decode("utf-8")


def skip_value(f, vtype):
    fmt = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f",
           7: "?", 10: "Q", 11: "q", 12: "d"}
    if vtype in fmt:
        f.read(struct.calcsize("<" + fmt[vtype]))
    elif vtype == 8:  # string
        read_string(f)
    elif vtype == 9:  # array
        (etype,) = struct.unpack("<I", f.read(4))
        (n,) = struct.unpack("<Q", f.read(8))
        if etype == 8:
            for _ in range(n):
                read_string(f)
        else:
            f.read(struct.calcsize("<" + fmt[etype]) * n)
    else:
        raise ValueError(f"unknown kv type {vtype}")


def main(path):
    with open(path, "rb") as f:
        magic, version, n_tensors, n_kv = struct.unpack("<IIQQ", f.read(24))
        assert magic == 0x46554747, "not GGUF"
        for _ in range(n_kv):
            read_string(f)
            (vtype,) = struct.unpack("<I", f.read(4))
            skip_value(f, vtype)
        counts = defaultdict(lambda: [0, 0, 0])  # type -> [n_tensors, elems, bytes]
        per_name = {}
        for _ in range(n_tensors):
            name = read_string(f)
            (nd,) = struct.unpack("<I", f.read(4))
            dims = struct.unpack(f"<{nd}Q", f.read(8 * nd))
            (ttype,) = struct.unpack("<I", f.read(4))
            f.read(8)  # offset
            elems = 1
            for d in dims:
                elems *= d
            bpb, vpb = BLOCK.get(ttype, (0, 1))
            nbytes = elems // vpb * bpb if bpb else 0
            counts[ttype][0] += 1
            counts[ttype][1] += elems
            counts[ttype][2] += nbytes
            per_name[name] = (ttype, dims, nbytes)
    total = sum(v[2] for v in counts.values())
    print(f"tensors={n_tensors} gguf_v{version}")
    for ttype, (n, elems, nbytes) in sorted(counts.items(), key=lambda kv: -kv[1][2]):
        print(f"  {TYPES.get(ttype, ttype):8s} n={n:4d} elems={elems:12d} "
              f"bytes={nbytes/1e6:9.1f} MB ({100*nbytes/total:5.1f}%)")
    print(f"  TOTAL data bytes = {total/1e6:.1f} MB")
    # per-weight-name breakdown for the big ones
    print("\nrepresentative tensors:")
    for name in ["token_embd.weight", "blk.0.attn_q.weight", "blk.0.attn_k.weight",
                 "blk.0.attn_v.weight", "blk.0.attn_output.weight",
                 "blk.0.ffn_gate.weight", "blk.0.ffn_up.weight", "blk.0.ffn_down.weight",
                 "blk.0.attn_norm.weight", "output_norm.weight"]:
        if name in per_name:
            t, dims, nb = per_name[name]
            print(f"  {name:32s} {TYPES.get(t, t):6s} dims={dims} {nb/1e6:.2f} MB")


if __name__ == "__main__":
    main(sys.argv[1])
