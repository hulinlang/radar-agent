#!/usr/bin/env python3
"""B11 layer-1 reference: int8 activations (Q8_1-aligned) x Q4_K/Q6_K weights.

This is the *independent* NumPy implementation that docs/34 section 3.2.1 pins
down. Two implementations of the same written-down semantics (this one and the
CUDA kernel) must agree BIT-EXACTLY element by element -- that is the only layer
that can be exact when the numerical route itself changes (layer 2 is
teacher-forcing against llama.cpp, layer 3 is the quality cost).

What is compared, and why each part is exact:

  * activations   -> d/sum_q/qs are compared as exact values (integers + the
                     half-rounded scale). The quantizer mirrors llama.cpp CUDA's
                     quantize_q8_1 (ggml/src/ggml-cuda/quantize.cu:54): quantize
                     with the fp32 scale amax/127, but *store* d as fp16 and use
                     that fp16 value in the dot.
  * int32 dots    -> dot1/dot2 are exact integers, so no ordering freedom.
  * fp32 combine  -> every op is a separate round-to-nearest multiply / add
                     (the CUDA side uses __fmul_rn/__fadd_rn to forbid FMA
                     contraction), so the *order* is part of the spec and both
                     sides must use the same parenthesisation.

Usage:
  python3 ref_q4_int8.py --gguf M.gguf --tensor blk.0.attn_q.weight \
      --outdir /tmp/b11 --ncol 16 [--nrow 0] [--seed N]

Writes, per tensor:
  <outdir>/<tag>.meta.txt   in/out/nb/ncol/type + FNV-1a hashes of the payloads
  <outdir>/<tag>.x.f32      [ncol][in] activations (fp32, little endian)
  <outdir>/<tag>.q8.bin     [ncol][nb] blocks of {f32 d; i32 sum_q; i8 qs[32]}
  <outdir>/<tag>.y.f32      [ncol][nrow] reference outputs (fp32)
"""

import argparse
import os
import struct
import sys

import numpy as np

QK_K = 256
GGML_Q4_K = 12
GGML_Q6_K = 14
BLOCK_BYTES = {GGML_Q4_K: 144, GGML_Q6_K: 210}

FNV_OFFSET = 0xcbf29ce484222325
FNV_PRIME = 0x100000001b3


# ---------------------------------------------------------------- GGUF reader

def _read_string(f):
    (n,) = struct.unpack("<Q", f.read(8))
    return f.read(n).decode("utf-8")


def _skip_value(f, vtype):
    fmt = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f",
           7: "?", 10: "Q", 11: "q", 12: "d"}
    if vtype in fmt:
        f.read(struct.calcsize("<" + fmt[vtype]))
    elif vtype == 8:
        _read_string(f)
    elif vtype == 9:
        (etype,) = struct.unpack("<I", f.read(4))
        (n,) = struct.unpack("<Q", f.read(8))
        if etype == 8:
            for _ in range(n):
                _read_string(f)
        else:
            f.read(struct.calcsize("<" + fmt[etype]) * n)
    else:
        raise ValueError("unknown gguf kv type %d" % vtype)


def gguf_tensor(path, name):
    """Return (dims, ggml_type, raw_bytes) for one tensor."""
    with open(path, "rb") as f:
        magic, _version, n_tensors, n_kv = struct.unpack("<IIQQ", f.read(24))
        if magic != 0x46554747:
            raise ValueError("not a GGUF file: %s" % path)
        alignment = 32
        for _ in range(n_kv):
            key = _read_string(f)
            (vtype,) = struct.unpack("<I", f.read(4))
            if key == "general.alignment":
                (alignment,) = struct.unpack("<I", f.read(4))
            else:
                _skip_value(f, vtype)
        want = None
        for _ in range(n_tensors):
            tname = _read_string(f)
            (nd,) = struct.unpack("<I", f.read(4))
            dims = struct.unpack("<%dQ" % nd, f.read(8 * nd))
            (ttype,) = struct.unpack("<I", f.read(4))
            (off,) = struct.unpack("<Q", f.read(8))
            if tname == name:
                want = (dims, ttype, off)
        if want is None:
            raise KeyError("tensor %s not in %s" % (name, path))
        dims, ttype, off = want
        n = 1
        for d in dims:
            n *= int(d)
        bb = BLOCK_BYTES.get(ttype)
        if bb is None:
            raise ValueError("tensor %s has unsupported type %d" % (name, ttype))
        n_bytes = n // QK_K * bb
        pos = f.tell()
        data_start = (pos + alignment - 1) // alignment * alignment
        f.seek(data_start + off)
        raw = f.read(n_bytes)
    if len(raw) != n_bytes:
        raise IOError("short read for %s" % name)
    return dims, ttype, raw


# ------------------------------------------------------- activation quantizer

def quantize_q8_1(x):
    """x: [ncol][k] float32 -> (d [ncol][nb] f32, sum_q [ncol][nb] i32, qs [ncol][nb][32] i8).

    Mirrors llama.cpp CUDA quantize_q8_1: the fp32 d32 = amax/127 quantizes, the
    *stored and used* d is fp16(d32).
    """
    x = np.ascontiguousarray(x, dtype=np.float32)
    ncol, k = x.shape
    assert k % 32 == 0, "k must be a multiple of 32"
    nb = k // 32
    xr = x.reshape(ncol, nb, 32)

    amax = np.abs(xr).max(axis=2).astype(np.float32)
    d32 = (amax / np.float32(127.0)).astype(np.float32)
    # round-to-nearest-even fp16 round trip == __float2half_rn + __half2float
    d = d32.astype(np.float16).astype(np.float32)

    safe = np.where(d32 == 0.0, np.float32(1.0), d32).astype(np.float32)
    v = (xr / safe[:, :, None]).astype(np.float32)
    # roundf(): half away from zero (np.round is half-to-even, so do not use it)
    q = (np.floor(np.abs(v) + np.float32(0.5)) * np.sign(v)).astype(np.float32)
    q = np.where(d32[:, :, None] == 0.0, np.float32(0.0), q)
    if np.max(np.abs(q)) > 127.0:
        raise AssertionError("quantized value out of int8 range: %g" % np.max(np.abs(q)))
    qs = q.astype(np.int8)
    sum_q = qs.astype(np.int32).sum(axis=2)
    return d.reshape(ncol, nb), sum_q.reshape(ncol, nb), qs


def q8_pack(d, sum_q, qs):
    """Pack into the block-major layout the C++/CUDA side reads:
    int8 qs[k/32][ncol][32] (a block's columns contiguous) followed by
    Q8_1Ds ds[k/32][ncol]."""
    ncol, nb = d.shape
    qflat = np.ascontiguousarray(qs.transpose(1, 0, 2)).reshape(-1).view(np.uint8)
    ds = np.zeros((nb, ncol, 8), dtype=np.uint8)
    ds[:, :, 0:4] = np.ascontiguousarray(d.astype(np.float32).T).view(np.uint8).reshape(nb, ncol, 4)
    ds[:, :, 4:8] = np.ascontiguousarray(sum_q.astype(np.int32).T).view(np.uint8).reshape(nb, ncol, 4)
    return np.concatenate([qflat, ds.reshape(-1)])


# ------------------------------------------------------------- weight layouts

def f16_field(sb, off):
    """sb: [..., block_bytes] uint8 -> [...] float32 of the fp16 at `off`."""
    two = np.ascontiguousarray(sb[..., off:off + 2])
    return two.view(np.float16).reshape(sb.shape[:-1]).astype(np.float32)


def q4k_scales(sc12):
    """sc12: [...,12] uint8 -> (sc [...,8] i32, m [...,8] i32) via get_scale_min_k4."""
    q = sc12.astype(np.int32)
    sc = np.zeros(q.shape[:-1] + (8,), dtype=np.int32)
    m = np.zeros(q.shape[:-1] + (8,), dtype=np.int32)
    for j in range(8):
        if j < 4:
            sc[..., j] = q[..., j] & 63
            m[..., j] = q[..., j + 4] & 63
        else:
            sc[..., j] = (q[..., j + 4] & 0xF) | ((q[..., j - 4] >> 6) << 4)
            m[..., j] = (q[..., j + 4] >> 4) | ((q[..., j] >> 6) << 4)
    return sc, m


def q4k_unpack(raw, out, nb):
    """raw -> (d4, dmin, sc, m, nib) with nib [out][nb][8][32] uint8 in 0..15."""
    sb = np.frombuffer(raw, dtype=np.uint8).reshape(out, nb, 144)
    d4 = f16_field(sb, 0)          # [out][nb]
    dmin = f16_field(sb, 2)
    sc, m = q4k_scales(sb[:, :, 4:16])
    qs = sb[:, :, 16:144].reshape(out, nb, 4, 32)
    nib = np.empty((out, nb, 8, 32), dtype=np.uint8)
    nib[:, :, 0::2, :] = qs & 0x0F          # sub-block 2g = low nibbles
    nib[:, :, 1::2, :] = (qs >> 4) & 0x0F   # sub-block 2g+1 = high nibbles
    return d4, dmin, sc, m, nib


def q4k_dequant(raw, in_dim, out_dim):
    """Plain fp32 dequant [out][in] -- sanity report only (pre-B11 route)."""
    nb = in_dim // QK_K
    d4, dmin, sc, m, nib = q4k_unpack(raw, out_dim, nb)
    w = (d4[:, :, None, None] * sc[:, :, :, None] * nib.astype(np.float32)
         - dmin[:, :, None, None] * m[:, :, :, None])
    return w.reshape(out_dim, in_dim)


def q6k_unpack(raw, out, nb):
    """raw -> (d6, sc [out][nb][16] i32, q6 [out][nb][8][2][16] i8 in [-32,31])."""
    sb = np.frombuffer(raw, dtype=np.uint8).reshape(out, nb, 210)
    d6 = f16_field(sb, 208)
    sc = np.ascontiguousarray(sb[:, :, 192:208]).view(np.int8).reshape(out, nb, 16).astype(np.int32)
    ql = sb[:, :, 0:128]
    qh = sb[:, :, 128:192]
    q6 = np.empty((out, nb, 8, 2, 16), dtype=np.int8)
    for nn in range(2):
        qlo = ql[:, :, nn * 64:nn * 64 + 32].astype(np.int32)
        qhi = ql[:, :, nn * 64 + 32:nn * 64 + 64].astype(np.int32)
        h = qh[:, :, nn * 32:nn * 32 + 32].astype(np.int32)
        groups = [
            (qlo & 0xF) | ((h & 3) << 4),
            (qhi & 0xF) | (((h >> 2) & 3) << 4),
            (qlo >> 4) | (((h >> 4) & 3) << 4),
            (qhi >> 4) | (((h >> 6) & 3) << 4),
        ]
        for qq, g in enumerate(groups):
            # scale sc[nn*8 + 2*qq] covers l<16, sc[nn*8 + 2*qq + 1] covers l>=16
            q6[:, :, nn * 4 + qq, 0, :] = (g[:, :, 0:16] - 32).astype(np.int8)
            q6[:, :, nn * 4 + qq, 1, :] = (g[:, :, 16:32] - 32).astype(np.int8)
    return d6, sc, q6


def q6k_dequant(raw, in_dim, out_dim):
    nb = in_dim // QK_K
    d6, sc, q6 = q6k_unpack(raw, out_dim, nb)
    # group g = nn*4+qq; scale index nn*8 + 2*qq + half -> reshape sc to [nn][qq][half]
    sc5 = sc.reshape(out_dim, nb, 2, 4, 2)
    q65 = q6.astype(np.float32).reshape(out_dim, nb, 2, 4, 2, 16)
    w = d6[:, :, None, None, None, None] * (sc5[:, :, :, :, :, None] * q65)
    return w.reshape(out_dim, in_dim)


# ------------------------------------------------------------------ the kernel

def matvec_q4k_int8(raw, in_dim, out_dim, d8, sum_q, qs):
    """y[out][ncol]; the exact parenthesisation of docs/34 section 3.2.1."""
    nb = in_dim // QK_K
    ncol = d8.shape[0]
    d4, dmin, sc, m, nib = q4k_unpack(raw, out_dim, nb)
    qa = qs.astype(np.int32).reshape(ncol, nb, 8, 32)
    # dot1 is an exact int32 sum -> integer addition order is irrelevant
    dot1 = np.einsum("objl,cbjl->ocbj", nib.astype(np.int32), qa, optimize=True)
    del nib, qa

    y = np.zeros((out_dim, ncol), dtype=np.float32)
    for b in range(nb):
        sumf_d = np.zeros((out_dim, ncol), dtype=np.float32)
        sumf_m = np.zeros((out_dim, ncol), dtype=np.float32)
        for j in range(8):
            a8 = d8[:, b * 8 + j].astype(np.float32)                     # [ncol]
            t1 = (dot1[:, :, b, j] * sc[:, b, j][:, None]).astype(np.float32)
            t2 = (sum_q[:, b * 8 + j][None, :] * m[:, b, j][:, None]).astype(np.float32)
            sumf_d = (sumf_d + (a8[None, :] * t1).astype(np.float32)).astype(np.float32)
            sumf_m = (sumf_m + (a8[None, :] * t2).astype(np.float32)).astype(np.float32)
        part = ((d4[:, b][:, None] * sumf_d).astype(np.float32)
                - (dmin[:, b][:, None] * sumf_m).astype(np.float32)).astype(np.float32)
        y = (y + part).astype(np.float32)
    return y


def matvec_q6k_int8(raw, in_dim, out_dim, d8, sum_q, qs):
    del sum_q  # the Q6_K weight has no offset term: dot2 is unused
    nb = in_dim // QK_K
    ncol = d8.shape[0]
    d6, sc, q6 = q6k_unpack(raw, out_dim, nb)
    qa = qs.astype(np.int32).reshape(ncol, nb, 8, 2, 16)
    dot = np.einsum("objhl,cbjhl->ocbjh", q6.astype(np.int32), qa, optimize=True)
    del q6, qa

    y = np.zeros((out_dim, ncol), dtype=np.float32)
    for b in range(nb):
        sd = np.zeros((out_dim, ncol), dtype=np.float32)
        for nn in range(2):
            for qq in range(4):
                g = nn * 4 + qq
                a8 = d8[:, b * 8 + g].astype(np.float32)
                for half in range(2):
                    sg = sc[:, b, nn * 8 + 2 * qq + half][:, None]
                    t = (dot[:, :, b, g, half] * sg).astype(np.float32)
                    sd = (sd + (a8[None, :] * t).astype(np.float32)).astype(np.float32)
        part = (d6[:, b][:, None] * sd).astype(np.float32)
        y = (y + part).astype(np.float32)
    return y


# -------------------------------------------------------------- misc plumbing

def fnv1a(path):
    h = FNV_OFFSET
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 16)
            if not chunk:
                break
            for byte in chunk:
                h ^= byte
                h = (h * FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    return h


def make_x(ncol, k, seed=0x5EED):
    """Deterministic activations with a wide dynamic range (so amax varies).

    Written to disk and read back by C++, so the two sides cannot drift.
    """
    rng = np.random.default_rng(seed)
    scale = np.exp(rng.uniform(-4.0, 4.0, size=(ncol, 1))).astype(np.float32)
    idx = np.arange(k, dtype=np.float64)
    base = np.sin(idx * 0.017) + 0.5 * np.cos(idx * 0.043)
    base = (base * 0.7 + 0.3 * rng.standard_normal(k)).astype(np.float32)
    return (base[None, :] * scale).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--tensor", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--ncol", type=int, default=16)
    ap.add_argument("--nrow", type=int, default=0, help="0 = all rows")
    ap.add_argument("--seed", type=int, default=0x5EED)
    args = ap.parse_args()

    dims, ttype, raw = gguf_tensor(args.gguf, args.tensor)
    in_dim, out_dim = int(dims[0]), int(dims[1])
    nrow = out_dim if args.nrow <= 0 else min(args.nrow, out_dim)
    assert in_dim % QK_K == 0 and in_dim % 32 == 0

    x = make_x(args.ncol, in_dim, args.seed)
    d8, sum_q, qs = quantize_q8_1(x)
    nb = in_dim // QK_K

    if ttype == GGML_Q4_K:
        y = matvec_q4k_int8(raw, in_dim, out_dim, d8, sum_q, qs)
        w = q4k_dequant(raw, in_dim, out_dim)
        tag = os.path.basename(args.tensor).replace(".", "_") + "_q4k"
    elif ttype == GGML_Q6_K:
        y = matvec_q6k_int8(raw, in_dim, out_dim, d8, sum_q, qs)
        w = q6k_dequant(raw, in_dim, out_dim)
        tag = os.path.basename(args.tensor).replace(".", "_") + "_q6k"
    else:
        raise ValueError("unsupported type %d" % ttype)

    os.makedirs(args.outdir, exist_ok=True)
    p_x = os.path.join(args.outdir, tag + ".x.f32")
    p_q = os.path.join(args.outdir, tag + ".q8.bin")
    p_y = os.path.join(args.outdir, tag + ".y.f32")
    p_m = os.path.join(args.outdir, tag + ".meta.txt")

    x.tofile(p_x)
    q8_pack(d8, sum_q, qs).tofile(p_q)
    y[:nrow].tofile(p_y)

    # two error numbers, both reported (not asserted). Denominator is the
    # tensor-wide max |y| (per-element denominators are meaningless here: many
    # outputs are near zero, so relative error per element blows up).
    yf = (w @ x.T).astype(np.float32)          # [out][ncol]
    xq = (np.repeat(d8, 32, axis=1) * qs.reshape(x.shape[0], -1)).astype(np.float32)
    yq = (w @ xq.T).astype(np.float32)
    scale = float(np.max(np.abs(yf[:nrow])))
    rel_act = float(np.max(np.abs(y[:nrow] - yf[:nrow]))) / scale
    rel_ord = float(np.max(np.abs(y[:nrow] - yq[:nrow]))) / scale

    with open(p_m, "w", encoding="utf-8") as f:
        f.write("tensor %s\n" % args.tensor)
        f.write("ggml_type %d\n" % ttype)
        f.write("in %d\nout %d\nnb %d\nncol %d\nnrow %d\nseed %d\n"
                % (in_dim, out_dim, nb, args.ncol, nrow, args.seed))
        f.write("x_fnv %016x\n" % fnv1a(p_x))
        f.write("q8_fnv %016x\n" % fnv1a(p_q))
        f.write("y_fnv %016x\n" % fnv1a(p_y))
        f.write("int8_vs_fp32act_max_rel %.6e\n" % rel_act)
        f.write("int8_vs_dequantact_max_rel %.6e\n" % rel_ord)
        f.write("y_scale %.6e\n" % scale)

    print("REF %-30s type=%d in=%d out=%d nb=%d ncol=%d nrow=%d"
          % (args.tensor, ttype, in_dim, out_dim, nb, args.ncol, nrow))
    print("    x_fnv=%016x q8_fnv=%016x y_fnv=%016x"
          % (fnv1a(p_x), fnv1a(p_q), fnv1a(p_y)))
    print("    int8 vs fp32-activation route : max_rel=%.3e" % rel_act)
    print("    int8 vs dequantized-act route : max_rel=%.3e" % rel_ord)


if __name__ == "__main__":
    sys.exit(main())
