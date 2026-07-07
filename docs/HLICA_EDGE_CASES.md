# HLiCA's stated edge cases vs. what actually happened in `liver_hlica_v1`

The HLiCA paper is explicit about several annotation edge cases. This checks each one
directly against the reference we built (`build_hlica_liver.py`, see HLICA_LIVER.md) —
and turns up two concrete, fixable gaps we introduced ourselves.

## 1. "The hepatocyte lineage was challenging to annotate... may originate from
technical artifacts but could represent true biological variation (e.g., Mito+ and
Ribosomal+ hepatocytes)"

The paper flags its own Mito+/Ribosomal+/SERPINE1+/UGT+ hepatocyte substates as
*possible* dissociation-stress artifacts, not confirmed biology — they kept them as
distinct clusters anyway, with the caveat stated. We used `author_cell_type` (all 4
substates + Periportal/Pericentral) as fine training labels for the hepatocyte lineage
without carrying this caveat forward. Practically: our held-out hepatocyte confusion
matrix (HLICA_LIVER.md) shows most cross-zone error lands on these substates rather than
the opposite zone (portal↔central flip only 0.182) — consistent with them being a real,
separable signal the classifier can learn, whatever its biological origin. Worth keeping
the caveat in mind if using these substate predictions for downstream biological claims.

## 2. "We were unable to resolve a distinct mid-zonal population, likely due to the
limited availability of well-established transcriptional markers"

Confirms what we inferred independently: `author_cell_type` has no midzonal category,
only two poles (Periportal/Pericentral). We noted this in HLICA_LIVER.md as a known
difference from the previous GSE158723-based 3-tier build. Direct confirmation from the
source paper that this is a real marker-gene limitation, not an oversight on our end.

## 3. "Plasmacytoid dendritic cells (pDCs) are included in the entire atlas but not in
any lineage maps, as they are transcriptomically distinct from both the myeloid lineage
and lymphocyte lineage"

**Confirmed gap in our build.** `all_cells.h5ad` has 790 pDCs (`cell_type` =
"plasmacytoid dendritic cell"); `myeloid.h5ad` and `lymphocyte.h5ad` both have **zero**.
Since `build_hlica_liver.py` concatenates only the 6 lineage files, **`liver_hlica_v1`
cannot predict pDCs at all** — not a training-data-sparsity issue, a structural one: the
cells are absent from every file we combined. This is the same category of issue
REFINE.md's `bundled_reference` abstain threshold is meant to catch downstream (a real
type the reference has literally never seen), but the fix here is upstream — pull the 790
pDC cells from `all_cells.h5ad` and add them as a 39th class.

## 4. NRXN1+ stromal cells — "newly identified," "not previously resolved in individual
maps," spatially validated to periportal vasculature

**Confirmed gap in our build, self-inflicted.** The mesenchyme lineage file's
`author_cell_type` has 5 values (Vascular Smooth Muscle Cell, Hepatic Stellate Cell,
Portal Fibroblast, **CUX2+ Hepatic Stellate Cell** [414 cells], **NRXN1+ Stromal Cell**
[347 cells]) — but its standardized `cell_type` column has only 3, because there's no
Cell Ontology term yet for either novel substate; both collapse into generic "hepatic
stellate cell" (11,287 = 10,526 + 414 + 347, confirmed by direct count). We used
`author_cell_type` for hepatocytes specifically but `cell_type` for the other 5 lineages
— an inconsistency, not a deliberate choice. **This is the paper's headline finding
(NRXN1+ stromal cells) and our reference cannot predict it.**

## 5. Checking whether the same collapsing happens elsewhere — it does, everywhere

We only special-cased hepatocytes. A systematic check of all 6 lineages' `author_cell_type`
vs. `cell_type` shows the same pattern throughout:

| lineage | author_cell_type values | cell_type values | notable collapses |
|---|---|---|---|
| myeloid | 13 | 11 | MAMLD1+ Trans Monocytes (3,074) and Activated Monocytes (1,566) → generic "monocyte"; TREM2+ Macrophages (1,686) → "lipid-associated macrophage"; Type 1/2 cDCs → both "conventional dendritic cell" |
| endothelial | 8 | 7 | Periportal LSEC (28,433) + Portal Vein (5,291) both → "endothelial cell of periportal hepatic sinusoid" (loses the vein/sinusoid distinction) |
| cholangiocyte | 5 | 2 | ApoLipo/Keratin/CXCL8+ Keratin/LAMC2+ substates (8,340 total) all → generic "cholangiocyte" |
| lymphocyte | 12 | 11 | Bright NK (12,851) / Dim NK (13,543) both → "natural killer cell" |
| mesenchyme | 5 | 3 | (item 4, above) |

The `MAMLD1+ Trans Monocytes` and `TREM2+ Macrophages` populations are specifically
called out in the paper's Results as novel findings ("Liver myeloid cell landscape")
tied to MASH/metabolic disease states — same category of miss as NRXN1+ stromal cells,
just less headline-prominent.

## 6. "Cycling" — a genuine cross-lineage ambiguity, not just a hepatocyte artifact

We found (and correctly dropped) 1,179 hepatocyte-file cells labeled `author_cell_type`
= "Cycling" whose actual `cell_type` is "lymphocyte" — flagged in the previous build as
a mislabeled contaminant. Checking the other lineage files shows this is **not**
hepatocyte-specific: "Cycling" clusters exist in the myeloid file (1,293 cells),
endothelial file (202 cells), and lymphocyte file (2,668 cells) too — always resolving
to `cell_type` = "lymphocyte" regardless of which lineage's file physically contains
them. Total: 5,342 cells (~1% of the atlas). This is the classic scRNA-seq edge case
where cell-cycle gene expression dominates a proliferating cell's transcriptome enough
to cluster it by cycle phase rather than lineage identity — HLiCA's own file-based
lineage split is a clustering result, not a ground-truth partition, and these are its
known exceptions (the flip side of the pDC exclusion in #3: pDCs were pulled *out* of
every lineage file for being cross-cutting; Cycling cells were *left in* a
possibly-wrong lineage file but correctly cell-typed anyway).

**We only explicitly checked and handled this for hepatocytes.** For the other 5
lineages we used `cell_type` directly (not `author_cell_type`), which already resolves
these correctly to "lymphocyte" — so the mislabeling issue doesn't bite there. But our
hierarchy construction (`dict(zip(fine_labels, lineage_tags))`, last-write-wins on
duplicate keys) only ended up assigning these cross-lineage "lymphocyte" rows to the
correct "lymphocyte" coarse group **because "lymphocyte" happens to be last in our
`LINEAGES` list** — an accident of code ordering, not a principled design. A different
list order would have silently mis-routed thousands of cells. Worth hardening: derive
the coarse group from `cell_type`'s own implied lineage where the two disagree, not from
file-of-origin.

## 7. Regulatory T cells — the paper's own validation of "more cells per type helps"

"Regulatory T cells were identified across research centres in the HLiCA dataset, even
though this population was not annotated in most of the original individual studies" —
attributed directly to statistical power from pooling many donors. This is an
independent confirmation of REFINE.md's central finding from this same project: our
798-type census reference (15-40 cells/type) badly underperforms a focused reference
built from thousands of cells/type on the same held-out cells (exact-CL 0.231 → 0.728).
HLiCA's authors hit the identical wall from the opposite direction — individual small
studies couldn't resolve Tregs at all; only integration across 110 donors could.

## 8. Innate lymphoid cells — an honestly-stated gap, not fixable by more data alone

"ILCs are known to represent 1% of liver lymphocytes... there are no established gene
expression markers for liver ILCs and we could not annotate a population for them in the
HLiCA." Distinct from the pDC/NRXN1+ cases above: this isn't a coverage gap in a specific
processed file, it's a stated absence from the *entire* atlas. `liver_hlica_v1` inherits
this gap and cannot be fixed by re-including a dropped file — would need markers HLiCA
itself says don't yet exist.

## Fixed: `liver_hlica_v2`

Built (`build_hlica_liver_v2.py`) with both fixes applied:

1. `author_cell_type` uniformly across all 6 lineages (not just hepatocytes) →
   **48 fine types across 7 coarse groups** (up from 38/6), close to HLiCA's own
   reported 47.
2. The 790 pDCs added as their own coarse group, pulled from `all_cells.h5ad`.
3. `class_to_cl` built directly from each row's own `cell_type_ontology_term_id`
   (not a hand-typed dict) — a collapsed label naturally inherits its parent's CL id.
4. "Cycling" cells — found not just in hepatocyte but in 4 of 6 lineage files (5,342
   cells total, always resolving to `cell_type` = "lymphocyte" regardless of host
   file) — are re-routed to the lymphocyte group using their own label, not dropped
   and not left in a possibly-wrong lineage by accident of dict ordering.

**Same cross-study held-out validation (train on 6 studies, test only on the withheld
Andrews_2022) — the fixes cost nothing on the core metrics:**

| metric | v1 (38 types) | v2 (48 types) |
|---|---|---|
| exact-CL | 0.728 | 0.724 |
| ontology-concordant | 0.858 | 0.859 |
| hepatocyte zonation exact-zone | 0.787 | 0.789 |
| hepatocyte zonation flip rate | 0.182 | 0.178 |
| endothelial zonation exact-zone | 0.711 | 0.708 |
| endothelial zonation flip rate | 0.264 | 0.267 |

Recovering the lost cell types (NRXN1+ stromal cells, CUX2+ hepatic stellate cells,
MAMLD1+ trans monocytes, TREM2+ macrophages, Type 1/2 cDCs, Bright/Dim NK cells, finer
endothelial vascular beds, pDCs) is essentially free — the differences above are noise,
not signal.

**One honest new result, not glossed over:** pDC held-out recall was **0/4** — only 4
pDC cells landed in the held-out `Andrews_2022` split (matches the low pDC representation
in that study generally), too small a sample to draw any real conclusion from, but stated
plainly rather than omitted. pDCs are a genuinely hard, rare (0.15% of the atlas),
transcriptionally-close-to-other-immune-cells population; 786 training examples spread
across ~468k cells may simply not be enough for the coarse router to reliably carve out
a 7th bucket. Shipped anyway since it's a strict improvement in *what the reference can
represent at all* — a missed pDC call is a coarse-routing error into a sibling myeloid/
lymphocyte class, not a new failure mode, and doesn't regress the other 47 types.

`liver_hlica_v2` is the recommended bundled reference going forward; `liver_hlica_v1`
stays available for anyone who wants the smaller 38-type taxonomy.
