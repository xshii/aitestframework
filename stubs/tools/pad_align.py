#!/usr/bin/env python3
"""Pad a binary file with zeros to N-byte alignment."""
import argparse, pathlib

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", help="Input binary file")
    ap.add_argument("-a", "--align", type=int, default=8, help="Alignment in bytes (default: 8)")
    ap.add_argument("-o", "--output", help="Output file (default: overwrite input)")
    args = ap.parse_args()

    data = bytearray(pathlib.Path(args.input).read_bytes())
    remainder = len(data) % args.align
    if remainder:
        pad = args.align - remainder
        data.extend(b'\x00' * pad)
        print(f"Padded {pad} bytes (total {len(data)})")
    else:
        print(f"Already aligned ({len(data)} bytes)")

    dst = args.output or args.input
    pathlib.Path(dst).write_bytes(data)

if __name__ == "__main__":
    main()
