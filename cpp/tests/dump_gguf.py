#!/usr/bin/env python3
"""Independent GGUF parser emitting the canonical dump format shared with
src/gguf_dump.cpp (see that file for the format spec). Stdlib only; used as
the golden reference for the B1 byte-exact diff test. Strings are handled as
raw bytes end to end so both sides emit identical byte streams."""

import mmap
import struct
import sys

TYPE_TABLE = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22), 7: ("Q5_1", 32, 24), 8: ("Q8_0", 32, 34), 9: ("Q8_1", 32, 40),
    10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110), 12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210), 15: ("Q8_K", 256, 292),
}
TAG = {0: "u8", 1: "i8", 2: "u16", 3: "i16", 4: "u32", 5: "i32", 6: "f32",
       7: "bool", 8: "str", 9: "array", 10: "u64", 11: "i64", 12: "f64"}
FMT = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f",
       7: "B", 10: "Q", 11: "q", 12: "d"}


class Cursor:
    def __init__(self, buf):
        self.buf = buf
        self.p = 0

    def read(self, fmt, what):
        n = struct.calcsize("<" + fmt)
        if self.p + n > len(self.buf):
            raise ValueError(f"unexpected end of file while reading {what}")
        v = struct.unpack_from("<" + fmt, self.buf, self.p)[0]
        self.p += n
        return v

    def read_string(self, what):
        n = self.read("Q", what)
        if n > len(self.buf) - self.p:
            raise ValueError(f"string length out of bounds while reading {what}")
        s = self.buf[self.p:self.p + n]
        self.p += n
        return bytes(s)


def read_value(cur, vtype, what):
    if vtype == 8:
        return ("str", cur.read_string(what))
    if vtype == 9:
        et = cur.read("I", what)
        if et > 12 or et == 9:
            raise ValueError("bad array element type")
        n = cur.read("Q", what)
        if n > len(cur.buf) - cur.p:
            raise ValueError("array length out of bounds")
        return ("array", [read_value(cur, et, what) for _ in range(n)], et)
    if vtype == 7:
        return ("bool", cur.read("B", what) != 0)
    return (TAG[vtype], cur.read(FMT[vtype], what))


def render_scalar(v):
    tag, val = v[0], v[1]
    if tag == "str":
        return b'"' + val + b'"'
    if tag == "bool":
        return b"true" if val else b"false"
    if tag in ("f32", "f64"):
        return ("%.17g" % val).encode()
    return str(val).encode()


def main():
    path = sys.argv[1]
    with open(path, "rb") as fh:
        buf = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        out = sys.stdout.buffer
        try:
            cur = Cursor(buf)
            if buf[0:4] != b"GGUF":
                raise ValueError("bad magic (not a GGUF file)")
            cur.p = 4
            version = cur.read("I", "version")
            if version not in (2, 3):
                raise ValueError(f"unsupported version {version}")
            tensor_count = cur.read("Q", "tensor_count")
            kv_count = cur.read("Q", "metadata_kv_count")

            kv_begin = cur.p
            kvs = []
            kv_map = {}
            for _ in range(kv_count):
                key = cur.read_string("metadata key")
                vtype = cur.read("I", "metadata value type")
                if vtype > 12:
                    raise ValueError("bad metadata value type")
                v = read_value(cur, vtype, "metadata value")
                kvs.append((key, v))
                kv_map[key] = v
            kv_bytes = cur.p - kv_begin

            infos_begin = cur.p
            tensors = []
            for _ in range(tensor_count):
                name = cur.read_string("tensor name")
                n_dims = cur.read("I", "tensor n_dims")
                if n_dims > 8:
                    raise ValueError("tensor n_dims out of range")
                dims = [cur.read("Q", "tensor dims") for _ in range(n_dims)]
                gtype = cur.read("I", "tensor type")
                offset = cur.read("Q", "tensor offset")
                tname, block, tsize = TYPE_TABLE[gtype]
                n_elems = 1
                for d in dims:
                    n_elems *= d
                if n_elems % block != 0:
                    raise ValueError(f"tensor {name} element count not divisible by block size")
                tensors.append((name, gtype, tname, dims, offset, n_elems // block * tsize))
            infos_bytes = cur.p - infos_begin

            alignment = 32
            if b"general.alignment" in kv_map:
                alignment = kv_map[b"general.alignment"][1]
            data_start = (cur.p + alignment - 1) // alignment * alignment
            file_size = len(buf)
            data_bytes = file_size - data_start

            max_end = 0
            for name, _, _, _, offset, nbytes in tensors:
                if offset % alignment != 0:
                    raise ValueError(f"tensor {name} offset is not aligned")
                if offset + nbytes > data_bytes:
                    raise ValueError(f"tensor {name} extends beyond end of file")
                max_end = max(max_end, offset + nbytes)
            if max_end != data_bytes:
                raise ValueError("file size accounting mismatch")

            out.write(b"GGUF version=%d tensors=%d kv=%d alignment=%d\n"
                      % (version, len(tensors), len(kvs), alignment))
            for key, v in kvs:
                out.write(b"KV " + key + b" = ")
                if v[0] == "array":
                    elems, et = v[1], v[2]
                    out.write(b"[" + b",".join(render_scalar(e) for e in elems) + b"]")
                    out.write(b" (%s[], n=%d)\n" % (TAG[et].encode(), len(elems)))
                else:
                    out.write(render_scalar(v))
                    out.write(b" (%s)\n" % v[0].encode())

            for i, (name, gtype, tname, dims, offset, nbytes) in enumerate(tensors):
                out.write(b"TENSOR %d %s %s dims=[%s] offset=%d bytes=%d\n"
                          % (i, name, tname.encode(),
                             b",".join(str(d).encode() for d in dims), offset, nbytes))

            tensors_sum = sum(t[5] for t in tensors)
            out.write(b"ACCOUNT header=24 kv=%d infos=%d padding=%d data=%d "
                      b"tensors_sum=%d pad_inside=%d total=%d ok\n"
                      % (kv_bytes, infos_bytes, data_start - 24 - kv_bytes - infos_bytes,
                         data_bytes, tensors_sum, data_bytes - tensors_sum, file_size))

            for want in (b"output_norm.weight", b"token_embd.weight"):
                t = next((t for t in tensors if t[0] == want), None)
                if t is None or t[1] not in (0, 1):
                    continue
                name, gtype, _, _, offset, _ = t
                base = data_start + offset
                fmt = "<8f" if gtype == 0 else "<8e"
                vals = struct.unpack_from(fmt, buf, base)
                out.write(b"HEAD " + name + b" ["
                          + b",".join(("%.17g" % v).encode() for v in vals) + b"]\n")
        except ValueError as e:
            sys.stderr.write(f"dump_gguf.py: error: {e}\n")
            sys.exit(1)


if __name__ == "__main__":
    main()
