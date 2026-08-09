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
import re, pathlib, sys

ROOT = pathlib.Path("/Users/iandriver/Downloads/actinn-jax-benchmark")
FIGDIR = ROOT / "docs" / "figures"
# Two manuscripts share this pipeline: the full report and a condensed, journal-length
# version. `python build_manuscript.py brief` builds the second.
VARIANT = (sys.argv[1] if len(sys.argv) > 1 else "full").lower()
SOURCE = {"full": "PAPER.md", "brief": "PAPER_BRIEF.md",
          "supp": "SUPPLEMENTARY.md"}[VARIANT]
STEM = {"full": "manuscript", "brief": "manuscript_brief",
        "supp": "manuscript_supp"}[VARIANT]
src = (ROOT / "docs" / SOURCE).read_text()

# ---- author block (confirmed 2026-08-01; not a placeholder) ----
TITLE = ("Annotating single-cell data on a laptop: a 13-method benchmark and practical "
         "low-memory workflows, with actinn-jax")
if VARIANT == "brief":
    TITLE = ("Accuracy is not the binding constraint in single-cell annotation: "
             "a 13-method benchmark of cost, scaling and workflow")
if VARIANT == "supp":
    TITLE = "Supplementary material"
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
if VARIANT == "supp":
    abstract = ""
else:
    nxt = "## 1. Introduction" if VARIANT == "full" else "## Key Points"
    m = re.search(r"## Abstract\n(.*?)\n" + re.escape(nxt), body, re.S)
    abstract = re.sub(r"\s+", " ", m.group(1)).strip()
    body = body[m.end()-len(nxt):]             # body starts at the next heading

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
# Hand-written link definitions are collected rather than discarded: they are what makes an
# inline [Key] clickable, and dropping them was why not one citation in the PDF was a link.
# references.yaml wins for any key it defines (step 5); anything else -- the availability
# section's bare-DOI pointers -- is carried through unchanged.
LINKDEF = re.compile(r"^\[([^\]]+)\]:\s*(https?://\S+)\s*$", re.M)
inline_links = {m.group(1): m.group(2) for m in LINKDEF.finditer(body)}
body = LINKDEF.sub("", body)

# 3b. repo-relative links to data and code are live hyperlinks in the PDF but resolve
# against nothing outside the repo, so they printed as dead links. Point them at GitHub --
# these are the result files and adapters behind the numbers, so a reader should be able to
# open them. Paths are relative to docs/, hence the posixpath.normpath.
REPO = "https://github.com/iandriver/actinn-jax-benchmark/blob/main"


def repo_link(m):
    import posixpath
    return f"]({REPO}/{posixpath.normpath(posixpath.join('docs', m.group(1)))})"


body = re.sub(r"(?<!!)\]\((?!https?://)([^)]+\.(?:csv|py|ya?ml|txt|json|sh))\)",
              repo_link, body)

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
            # the brief's gene-budget figures live only in the full report

            tag = " {-}" if title.strip() in ("Data & code availability",) else ""
            out.append("#" * level + " " + title + tag)
        else:
            out.append(l)
    return "\n".join(out)
body = headings(body)

# 5. references section -- generated from docs/references.yaml, which tools/check_references.py
# validates against the methods and datasets tables. Hand-editing a list here is how scPRINT
# ended up benchmarked with no citation and five of six datasets with none.
def locator(e):
    """(url, display) for the click-through target of one entry, or (None, None).

    Every entry needs one so a reader can check the citation without retyping it;
    tools/check_references.py fails the build if an entry has neither doi nor url."""
    if e.get("doi"):
        return f"https://doi.org/{e['doi']}", f"doi:{e['doi']}"
    if e.get("url"):
        return e["url"], re.sub(r"^https?://(www\.)?", "", e["url"]).rstrip("/")
    return None, None


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
        url, display = locator(e)
        # The entry's own DOI/URL is the click target: linking the whole entry would print
        # the reference list entirely in link colour.
        out.append(f"{n}. {e['text']}" + (f" [{display}]({url})." if url else ""))
        if url:
            # ... and the same target makes every inline [Key] in the body clickable.
            inline_links[key] = url
    defs = "\n".join(f"[{k}]: {u}" for k, u in sorted(inline_links.items()))
    globals()["CITE_URLS"] = dict(inline_links)
    return "\n".join(out) + "\n\n" + defs + "\n"


CITE_URLS: dict[str, str] = {}          # filled by _references(); empty is valid
REFS = "" if VARIANT == "supp" else _references()

# 5b. Citations rendered inconsistently: a shortcut link like [Kalfon 2025] resolves and
# markdown eats its brackets ("Foundation models Kalfon 2025 raise accuracy"), while an
# unresolvable multi-key one like [Abdelaal 2019, Fu 2024] stays literal and keeps them. So
# half the citations in the text had brackets and half did not. Rewriting each resolvable
# key as an inline link whose *text* is the escaped bracketed key gives every citation the
# same [Author Year] shape while staying clickable.
CITEKEY = re.compile(r"\d{4}$")          # a citation key ends in a year; `repo` does not


def bracket_citations(text):
    # Only citation-shaped keys. `inline_links` also holds plumbing keys such as [repo],
    # and rewriting that one shattered the reference-style link [benchmark repository][repo]
    # into literal brackets plus a bare URL -- a 157pt overfull line.
    cites = {k: u for k, u in CITE_URLS.items() if CITEKEY.search(k)}
    for key, url in sorted(cites.items(), key=lambda kv: -len(kv[0])):
        text = text.replace(f"[{key}]", f"[\\[{key}\\]]({url})")
    # a multi-key citation is plain text; escape it so LaTeX does not read the brackets
    text = re.sub(r"(?<!\\)\[([A-Z][^\]]*?\d{4}(?:, [^\]]*?\d{4})+)\]",
                  lambda m: "[" + m.group(1) + "]", text)
    return text


body = bracket_citations(body)

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

(ROOT / "manuscript" / f"{STEM}.md").write_text(
    FRONT + "\n" + breakable_dois(body) + "\n" + breakable_dois(REFS))
print(f"wrote manuscript/{STEM}.md (PDF via LaTeX)")

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
(ROOT / "manuscript" / f"{STEM}_portable.md").write_text(FRONT_PORTABLE + "\n" + body + "\n" + REFS)
print(f"wrote manuscript/{STEM}_portable.md (RTF/DOCX)")
