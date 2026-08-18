"""Derive the plain-text abstract bioRxiv wants from the one in the paper.

bioRxiv's abstract box rejects the typography a manuscript uses -- arrows, multiplication
signs, en and em dashes -- and the pasted text has to match the abstract in the submitted PDF.
Keeping the two in sync by hand does not work: the paper's abstract was corrected while
`submission/abstract.txt` kept a claim the paper no longer made. Deriving one from the other
makes that failure impossible.

    python3 manuscript/make_submission_abstract.py --check    # CI-style, changes nothing
    python3 manuscript/make_submission_abstract.py
"""

import argparse
import os
import re
import sys

SRC = "docs/PAPER.md"
OUT = "manuscript/submission/abstract.txt"

# Only characters bioRxiv will not take. Anything else non-ASCII should stop the build rather
# than be guessed at.
SUBS = [
    ("—", "-"),      # em dash
    ("–", "-"),      # en dash
    ("×", "x"),      # multiplication sign
    ("→", "to"),     # rightwards arrow
    ("≥", ">="),
    ("≤", "<="),
    ("≈", "~"),
    ("’", "'"),      # curly apostrophe
    ("“", '"'),
    ("”", '"'),
]


def extract(md):
    m = (re.search(r"\*\*Abstract\.?\*\*(.*?)\n##", md, re.S)
         or re.search(r"##\s*Abstract\s*\n(.*?)\n##", md, re.S))
    if not m:
        raise SystemExit(f"no abstract found in {SRC}")
    return m.group(1).strip()


def plain(text):
    # Unwrap first, strip markup second. The paper hard-wraps at ~95 columns, so `**multi-pass
    # workflow**` can straddle a line break, and `.` does not match a newline -- stripping bold
    # before unwrapping leaves those asterisks in the submitted text. Paragraph breaks survive
    # the unwrap: a single \s+ pass over the whole abstract would run two paragraphs together.
    text = "\n\n".join(re.sub(r"\s+", " ", para).strip()
                       for para in re.split(r"\n\s*\n", text) if para.strip())
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)     # bold
    text = re.sub(r"(?<!\w)\*(.+?)\*(?!\w)", r"\1", text)   # italic
    text = text.replace("`", "")
    for a, b in SUBS:
        text = text.replace(a, b)
    left = sorted({c for c in text if ord(c) > 127})
    if left:
        raise SystemExit("unmapped non-ASCII, add it to SUBS: "
                         + ", ".join(f"{c!r} (U+{ord(c):04X})" for c in left))
    return text + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="fail if the file on disk is stale, without rewriting it")
    a = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    want = plain(extract(open(os.path.join(root, SRC), encoding="utf-8").read()))
    dst = os.path.join(root, OUT)
    have = open(dst, encoding="utf-8").read() if os.path.exists(dst) else None

    if a.check:
        if have != want:
            print(f"STALE: {OUT} does not match the abstract in {SRC}", file=sys.stderr)
            raise SystemExit(1)
        print(f"{OUT} is current ({len(want.split())} words)")
        return

    os.makedirs(os.path.dirname(dst), exist_ok=True)
    open(dst, "w", encoding="utf-8").write(want)
    print(f"wrote {OUT} ({len(want.split())} words, "
          f"{'unchanged' if have == want else 'updated'})")
    print("ABSTRACT_DONE", flush=True)


if __name__ == "__main__":
    main()
