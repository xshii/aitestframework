#!/usr/bin/env python3
"""Print basic info about a binary file."""
import sys, pathlib

path = pathlib.Path(sys.argv[1])
data = path.read_bytes()
print(f"File:   {path}")
print(f"Size:   {len(data)} bytes")
print(f"First8: {data[:8].hex(' ')}")
print(f"Last8:  {data[-8:].hex(' ')}")
