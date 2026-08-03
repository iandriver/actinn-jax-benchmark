"""Verify that every method and dataset in the paper carries a resolvable citation.

The reference list used to be a hand-maintained block in build_manuscript.py, with no link
to the tables it was supposed to support: scPRINT was benchmarked for months with no
citation at all, five of six datasets had none, and most listed entries were never cited
anywhere. None of that is visible by reading, so it is checked instead.

Fails (exit 1) when:
  * a method row in PAPER.md section 2.2 has no entry under `methods:` in references.yaml
  * a dataset row in section 2.3 has no entry under `datasets:`
  * a citation key names an entry that does not exist
  * a cited entry is marked `status: unresolved` (we do not know the source)
  * an entry exists but nothing cites it (dead weight in the reference list)
  * a cited entry has neither `doi:` nor `url:` -- every reference must be one click from
    the source, both in the list and from the inline [Key], so a reader can check it

    python tools/check_references.py            # report + exit status
    python tools/check_references.py --list     # print the resolved bibliography
"""

import argparse
import os
import re
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.path.join(ROOT, "docs", "PAPER.md")
REFS = os.path.join(ROOT, "docs", "references.yaml")


def table_rows(text, heading):
    """First markdown table under `heading`: return each row's first cell, cleaned."""
    start = text.index(heading)
    nxt = text.find("\n### ", start + 1)
    block = text[start: nxt if nxt > 0 else len(text)]
    rows = []
    for line in block.splitlines():
        line = line.strip()
        if not line.startswith("|") or set(line) <= set("|-: "):
            continue
        cell = line.strip("|").split("|")[0].strip()
        cell = re.sub(r"[*`]", "", cell).strip()
        if cell and cell.lower() not in ("method", "dataset"):
            rows.append(cell)
    return rows


def as_list(v):
    return v if isinstance(v, list) else [v]


def check_links(entries):
    """Resolve each entry's click target. A DOI that 404s is a citation nobody can check.

    doi.org is asked without following the redirect: several publishers answer an automated
    request at the *destination* with 403, which says nothing about whether the DOI is
    registered. A 30x from doi.org does.
    """
    import concurrent.futures as cf
    import subprocess

    def one(item):
        key, e = item
        url = (f"https://doi.org/{e['doi']}" if e.get("doi") else e.get("url"))
        if not url:
            return key, None, "NO LINK", ""
        r = subprocess.run(["curl", "-sS", "-o", os.devnull, "-w", "%{http_code} %{redirect_url}",
                            "-I", "--max-time", "30", "-A", "Mozilla/5.0", url],
                           capture_output=True, text=True)
        code, _, target = r.stdout.partition(" ")
        return key, url, code, target

    todo = [(k, e) for k, e in sorted(entries.items()) if not k.startswith("UNRESOLVED")]
    bad = 0
    with cf.ThreadPoolExecutor(8) as ex:
        for key, url, code, target in ex.map(one, todo):
            ok = code.startswith(("2", "3"))
            bad += not ok
            print(f"{'ok ' if ok else 'FAIL'} {code:<7} {key:<24} {target[:64] or url or ''}")
    print(f"\n{len(todo)} entries, {bad} unresolvable link(s)")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="print the bibliography and exit")
    ap.add_argument("--links", action="store_true",
                    help="resolve every entry's doi/url over the network and exit")
    a = ap.parse_args()

    refs = yaml.safe_load(open(REFS))
    paper = open(PAPER).read()
    entries = refs["entries"]
    problems, warnings = [], []

    cited = set()
    for kind, heading in (("methods", "### 2.2 Benchmarked methods"),
                          ("datasets", "### 2.3 Datasets")):
        declared = refs.get(kind, {})
        for name in table_rows(paper, heading):
            if name not in declared:
                problems.append(f"{kind[:-1]} '{name}' in {heading.strip('# ')} has no "
                                f"citation in references.yaml")
                continue
            for key in as_list(declared[name]["cite"]):
                cited.add(key)
                if key not in entries:
                    problems.append(f"{kind[:-1]} '{name}' cites '{key}', which is not an "
                                    f"entry")
                elif entries[key].get("status") == "unresolved":
                    problems.append(f"{kind[:-1]} '{name}' has an UNRESOLVED source: "
                                    f"{entries[key].get('note', '').strip().splitlines()[0]}")

    # inline [Key] citations in the prose
    # keys are "Author YYYY" or "xkcd 927" -- do not require an initial capital
    # {1,40}, not {2,40}: a two-letter surname ("Fu 2024") has only one character between
    # the initial and the space, and the stricter form silently skipped those citations.
    for key in set(re.findall(r"\[([A-Za-z][^\]\[]{1,40}? \d{3,4})\]", paper)):
        if key in entries:
            cited.add(key)
        elif not re.search(rf"^\[{re.escape(key)}\]:", paper, re.M):
            warnings.append(f"inline citation '[{key}]' has no entry and no link definition")

    for key in entries:
        if key not in cited and not key.startswith("UNRESOLVED"):
            warnings.append(f"entry '{key}' is never cited")
        elif key in cited and not (entries[key].get("doi") or entries[key].get("url")):
            problems.append(f"entry '{key}' has no doi: or url:, so it cannot be linked")

    if a.links:
        return check_links(entries)

    if a.list:
        for i, (key, e) in enumerate(sorted(entries.items()), 1):
            if key.startswith("UNRESOLVED"):
                continue
            link = (f" https://doi.org/{e['doi']}" if e.get("doi")
                    else f" {e['url']}" if e.get("url") else "  <NO LINK>")
            print(f"{i}. {e['text']}{link}")
        return 0

    for w in warnings:
        print(f"warn:  {w}")
    for p in problems:
        print(f"FAIL:  {p}")
    n_m = len(refs.get("methods", {}))
    n_d = len(refs.get("datasets", {}))
    print(f"\n{n_m} methods, {n_d} datasets, {len(entries) - 1} bibliography entries; "
          f"{len(problems)} failure(s), {len(warnings)} warning(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
