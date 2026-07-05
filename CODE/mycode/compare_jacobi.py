#!/usr/bin/env python3
"""Compare two JacobiSetComputation *_jacobi.txt files.

The writer currently emits deterministic order, but the comparison is order
independent because future GPU compaction may produce a different edge order.
"""

from __future__ import annotations

import sys
from pathlib import Path


def read_jacobi_edges(path: Path) -> set[tuple[int, int]]:
    with path.open("r", encoding="utf-8") as f:
        header = f.readline().strip()
        if header != "JacobiSet":
            raise ValueError(f"{path}: invalid header {header!r}")

        try:
            expected = int(f.readline().strip())
        except ValueError as exc:
            raise ValueError(f"{path}: invalid edge count") from exc

        edges: set[tuple[int, int]] = set()
        for line_no, line in enumerate(f, start=3):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 8:
                raise ValueError(f"{path}:{line_no}: expected 8 fields, got {len(parts)}")

            a = int(parts[0])
            b = int(parts[4])
            if a > b:
                a, b = b, a
            edges.add((a, b))

    if len(edges) != expected:
        raise ValueError(f"{path}: header says {expected} edges, parsed {len(edges)} unique edges")

    return edges


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(f"Usage: {argv[0]} <reference_jacobi.txt> <candidate_jacobi.txt>", file=sys.stderr)
        return 2

    ref_path = Path(argv[1])
    cand_path = Path(argv[2])

    ref = read_jacobi_edges(ref_path)
    cand = read_jacobi_edges(cand_path)

    only_ref = sorted(ref - cand)
    only_cand = sorted(cand - ref)

    print(f"reference_edges: {len(ref)}")
    print(f"candidate_edges: {len(cand)}")

    if only_ref or only_cand:
        print(f"missing_edges: {len(only_ref)}")
        print(f"extra_edges: {len(only_cand)}")
        if only_ref:
            print("first_missing:", only_ref[:10])
        if only_cand:
            print("first_extra:", only_cand[:10])
        return 1

    print("match: yes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
