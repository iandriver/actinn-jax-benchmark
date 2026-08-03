#!/usr/bin/env python3
"""Transform docs/PAPER.md into a bioRxiv-style manuscript.md for pandoc->PDF.

- strips the repo-doc chrome (H1, status blockquote, 'in-repo' note)
- pulls the Abstract into YAML front matter (pandoc abstract environment)
- strips internal *.md links to plain text; keeps external http links
- rewrites figure paths to absolute; upgrades key figure captions
- appends a formatted References section

Run this script first, then build the three deliverables from `manuscript/`:

    python manuscript/build_manuscript.py
    cd manuscript
    pandoc -f markdown-implicit_figures manuscript.md -o actinn-jax_preprint.pdf \
        --pdf-engine=tectonic --number-sections --resource-path=.:../docs:../docs/figures
    pandoc -f markdown-implicit_figures manuscript_portable.md -o actinn-jax_preprint.rtf \
        --standalone --template=rtf-with-abstract.rtf.template \
        --resource-path=.:../docs:../docs/figures
    pandoc -f markdown-implicit_figures manuscript_portable.md -o actinn-jax_preprint.docx \
        --resource-path=.:../docs:../docs/figures

``-f markdown-implicit_figures`` is load-bearing everywhere: captions are written into
PAPER.md as numbered "**Figure N.**" / "**Table N.**" paragraphs so they survive into all
three formats *and* stay visible in the repo doc. Left on, pandoc would additionally promote
each image to a LaTeX float and number it itself, so the PDF alone would carry a second,
conflicting set of numbers.

The RTF flags are load-bearing and easy to lose:
  * ``--standalone`` -- without it pandoc emits a *fragment*: no {\\rtf1 header,
    no title block. The file opens but starts abruptly at the author name.
  * ``--template`` -- pandoc's stock RTF template has no ``$abstract$`` variable,
    so the abstract is silently dropped. The local template adds it.
PDF (via LaTeX) and DOCX imply standalone and need neither flag.
After building, sanity-check the RTF: it should start with ``{\\rtf1`` and contain
both the title and the first line of the abstract.
"""
import re, pathlib

ROOT = pathlib.Path("/Users/iandriver/Downloads/actinn-jax-benchmark")
FIGDIR = ROOT / "docs" / "figures"
src = (ROOT / "docs" / "PAPER.md").read_text()

# ---- author block (confirmed 2026-08-01; not a placeholder) ----
TITLE = ("Annotating single-cell data on a laptop: a 13-method benchmark and practical "
         "low-memory workflows, with actinn-jax")
# "Independent Researcher" is the standard bioRxiv affiliation for unaffiliated authors,
# and is the intended value here -- this work is not submitted under an institution.
AUTHORS = "Ian Driver$^{\\ast}$"
AFFIL = "Independent Researcher, Detroit, MI, USA"
CORR = "$^{\\ast}$ Correspondence: driver.ian@gmail.com"

lines = src.splitlines()

# 1. drop H1 title, the italic 'in-repo' note, and the Status blockquote
out, i = [], 0
while i < len(lines):
    l = lines[i]
    if l.startswith("# ") and i == 0:        # H1 title line
        i += 1; continue
    if l.startswith("*In-repo benchmark report"):   # italic note (until blank)
        while i < len(lines) and lines[i].strip():
            i += 1
        continue
    if l.startswith("> **Status:**") or l.startswith("> "):  # status blockquote
        i += 1; continue
    out.append(l); i += 1
body = "\n".join(out)

# unicode handled font-independently via \newunicodechar in the LaTeX preamble
# (below), so keep the literal characters in the text — this avoids pandoc's
# "$ before a digit isn't math" rule that would mangle e.g. 8->86 or x195.

# 2. extract Abstract -> yaml
m = re.search(r"## Abstract\n(.*?)\n## 1\. Introduction", body, re.S)
abstract = re.sub(r"\s+", " ", m.group(1)).strip()
body = body[m.end()-len("## 1. Introduction"):]        # start body at Introduction

# 3. handle internal .md/.png links (NOT image embeds, hence (?<!!)). If the
# link text is itself a filename (a self-referential pointer), drop it;
# otherwise keep the readable text.
def linkrepl(m):
    text = m.group(1).strip()
    return "" if (text.endswith(".md") or text.endswith(".png")) else text
body = re.sub(r"(?<!!)\[([^\]]+)\]\([^)]*\.(?:md|png)(?:#[^)]*)?\)", linkrepl, body)
# tidy the parentheses/phrases left behind by dropped doc pointers
body = re.sub(r"\s*See\s*\.", "", body)                       # "See ."
body = re.sub(r"\(full detail:\s*", "(", body)                # "(full detail: X)" -> "(X)"
body = re.sub(r"\(\s*,\s*", "(", body)                        # "(, and ..." -> "(and ..."
body = re.sub(r",\s*\)", ")", body)                           # "(X, )" -> "(X)"
body = re.sub(r"\s*\(\s*\)", "", body)                        # empty "()"
body = re.sub(r"\(\s*and the", "(see the", body)              # "(and the smooth..." -> "(see the..."
body = re.sub(r"^\[[^\]]+\]:\s*https?://\S+\s*$", "", body, flags=re.M)  # remove link defs

# 4. figures -> absolute path. Captions are numbered "**Figure N.**" paragraphs in PAPER.md
# itself: they have to be visible in the repo doc on GitHub (alt text is not), and only one
# source of numbering can exist if prose references are to stay correct. Every figure and
# table therefore carries its own caption in the source, and the three pandoc calls run with
# `-f markdown-implicit_figures` so LaTeX does not add a *second*, differently-numbered
# "Figure N:" of its own on top of ours -- which it did, calling Figure 3 "Figure 1".
body = re.sub(r"!\[([^\]]*)\]\(figures/([^)]+)\)",
              lambda m: f"![{m.group(1)}]({FIGDIR / m.group(2)})", body)

# 4b. headings: promote one level (## main section -> #) and strip the manual
# "N." / "N.M" numbers so pandoc --number-sections produces clean 1 / 1.1 numbering.
def headings(text):
    out = []
    for l in text.splitlines():
        h = re.match(r"^(#{2,})\s+(.*)$", l)
        if h:
            level = max(1, len(h.group(1)) - 1)
            title = re.sub(r"^\d+(\.\d+)?\.?\s+", "", h.group(2))
            tag = " {-}" if title.strip() in ("Data & code availability",) else ""
            out.append("#" * level + " " + title + tag)
        else:
            out.append(l)
    return "\n".join(out)
body = headings(body)

# 5. references section -- generated from docs/references.yaml, which tools/check_references.py
# validates against the methods and datasets tables. Hand-editing a list here is how scPRINT
# ended up benchmarked with no citation and five of six datasets with none.
def _references():
    import yaml
    with open(ROOT / "docs" / "references.yaml") as fh:
        refs = yaml.safe_load(fh)
    out = ["", "# References {-}", ""]
    n = 0
    for key, e in sorted(refs["entries"].items()):
        if e.get("status") == "unresolved":
            continue
        n += 1
        doi = f" doi:{e['doi']}." if e.get("doi") else ""
        out.append(f"{n}. {e['text']}{doi}")
    return "\n".join(out) + "\n"


REFS = _references()

# 6. DOIs are digits and punctuation, so no hyphenation pattern applies to them and TeX has
# nowhere to break: an unbreakable `doi:10.64898/2026.06.30.735539` simply overruns the right
# margin, which is what pushed three lines of the reference list and the Zenodo DOI in Data &
# code availability off the text block. Inserting explicit break points after `/` and `.` lets
# them wrap with no hyphen invented inside the identifier. LaTeX output only -- the RTF/DOCX
# variant must stay free of raw TeX, and Word wraps them on its own.
DOI = re.compile(r"10\.\d{4,5}/[^\s)\]}]+")
# Skip link targets and reference definitions: an \allowbreak inside a URL breaks the link.
PROTECTED = re.compile(r"(\]\([^)]*\)|^\[[^\]]+\]:.*$)", re.M)


def breakable_dois(text):
    parts = PROTECTED.split(text)
    return "".join(part if i % 2 else DOI.sub(
        lambda m: re.sub(r"([/.])", r"\1\\allowbreak{}", m.group(0)), part)
        for i, part in enumerate(parts))

FRONT = f"""---
title: |
  {TITLE}
author:
  - {AUTHORS}
date: ""
abstract: |
  {abstract}
geometry: margin=1in
fontsize: 11pt
linkcolor: RoyalBlue
urlcolor: RoyalBlue
header-includes:
  # long unbreakable code tokens (paths, dotted names) otherwise run off the page;
  # emergencystretch lets TeX loosen a problem paragraph and move them to the next
  # line, without hyphenating inside a filename and inventing a name that does not exist
  - \\setlength{{\\emergencystretch}}{{3em}}
  - \\usepackage{{authblk}}
  - \\renewcommand\\Authands{{, }}
  - \\usepackage{{newunicodechar}}
  - \\newunicodechar{{μ}}{{\\ensuremath{{\\mu}}}}
  - \\newunicodechar{{σ}}{{\\ensuremath{{\\sigma}}}}
  - \\newunicodechar{{≈}}{{\\ensuremath{{\\approx}}}}
  - \\newunicodechar{{×}}{{\\ensuremath{{\\times}}}}
  - \\newunicodechar{{→}}{{\\ensuremath{{\\rightarrow}}}}
  - \\newunicodechar{{≥}}{{\\ensuremath{{\\ge}}}}
  - \\newunicodechar{{≤}}{{\\ensuremath{{\\le}}}}
  - \\newunicodechar{{±}}{{\\ensuremath{{\\pm}}}}
  - \\newunicodechar{{≠}}{{\\ensuremath{{\\neq}}}}
  - \\newunicodechar{{≫}}{{\\ensuremath{{\\gg}}}}
  - \\newunicodechar{{²}}{{\\textsuperscript{{2}}}}
---

\\begin{{center}}\\small {AFFIL} \\\\ {CORR}\\end{{center}}
"""

(ROOT / "manuscript" / "manuscript.md").write_text(
    FRONT + "\n" + breakable_dois(body) + "\n" + breakable_dois(REFS))
print("wrote manuscript/manuscript.md (PDF via LaTeX)")

# ---- portable variant for RTF / DOCX (Pages-editable): no raw LaTeX; the
# unicode chars are kept literal (Word/Pages render them natively). ----
# The author name comes from the `author:` field via the title block, so the body
# carries only the affiliation/correspondence line -- repeating the name here would
# print it twice in the RTF/DOCX.
FRONT_PORTABLE = f"""---
title: "{TITLE}"
author: "Ian Driver"
date: ""
abstract: |
  {abstract}
---

*{AFFIL}*  ·  *Correspondence: driver.ian@gmail.com*
"""
(ROOT / "manuscript" / "manuscript_portable.md").write_text(FRONT_PORTABLE + "\n" + body + "\n" + REFS)
print("wrote manuscript/manuscript_portable.md (RTF/DOCX)")
