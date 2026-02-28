#!/usr/bin/env python3
"""Swap endianness of a binary file (32-bit word granularity)."""
import argparse, sys, pathlib

def swap(data, word_size=4):
    out = bytearray()
    for i in range(0, len(data), word_size):
        w = data[i:i+word_size]
        out.extend(w[::-1] if len(w) == word_size else w)
    return out

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", help="Input binary file")
    ap.add_argument("-o", "--output", help="Output file (default: overwrite input)")
    ap.add_argument("-w", "--word-size", type=int, default=4, help="Word size in bytes (default: 4)")
    args = ap.parse_args()

    data = pathlib.Path(args.input).read_bytes()
    out = swap(data, args.word_size)
    dst = args.output or args.input
    pathlib.Path(dst).write_bytes(out)
    print(f"Swapped {len(data)} bytes (word={args.word_size}) -> {dst}")

if __name__ == "__main__":
    main()
