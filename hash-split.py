#!/usr/bin/env python3
"""Stable train/test split of a line-based file by a content hash.

Usage: hash-split.py <infile> <nth>

Each DATA line is hashed (md5 of its content) and bucketed by hash % nth:
bucket 0 -> <infile>-test, everything else -> <infile>-train.  So nth=3 sends
~1/3 of the lines to test, nth=10 sends ~1/10, and so on.

Because the bucket is a hash of the line's CONTENT (not its position), the split
is deterministic and order-independent: the same line always lands on the same
side, in every run and across files -- so an identical record can never end up
in both train and test, and regenerating a corpus never reshuffles the split.

Comment lines (starting with '#') are copied to BOTH outputs so provenance
headers survive on each side; blank lines are dropped.
"""
import sys
import hashlib


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: hash-split.py <infile> <nth>   (nth=3 => ~1/3 to test)")
    infile = sys.argv[1]
    try:
        nth = int(sys.argv[2])
    except ValueError:
        sys.exit(f"nth must be a positive integer, got {sys.argv[2]!r}")
    if nth < 1:
        sys.exit(f"nth must be >= 1, got {nth}")

    train_path = infile + "-train"
    test_path = infile + "-test"
    n_train = n_test = n_comment = 0

    with open(infile) as f, open(train_path, "w") as tr, open(test_path, "w") as te:
        for line in f:
            s = line.rstrip("\n")
            if s == "":
                continue
            if s.startswith("#"):
                tr.write(line)
                te.write(line)
                n_comment += 1
                continue
            h = int.from_bytes(hashlib.md5(s.encode()).digest()[:4], "little")
            if h % nth == 0:
                te.write(line)
                n_test += 1
            else:
                tr.write(line)
                n_train += 1

    total = n_train + n_test
    frac = (100.0 * n_test / total) if total else 0.0
    msg = (f"{infile}: {total} data lines -> {n_train} train, {n_test} test "
           f"({frac:.1f}% test; nth={nth} ~ {100.0 / nth:.1f}%)")
    if n_comment:
        msg += f"; {n_comment} comment line(s) copied to both"
    sys.stderr.write(msg + f"\n  {train_path}\n  {test_path}\n")


if __name__ == "__main__":
    main()
