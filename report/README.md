# Project report

`report.tex` is the LaTeX source of the project report. It covers the problem,
the solution, the theoretical analysis, the experimental results, the lessons
learned and ideas for improvement.

| File | Content |
|---|---|
| `report.tex` | the report (standard packages only) |
| `make_figures.py` | runs every experiment; writes `figures/*.pdf` and `generated/*.tex` |
| `capture_screens.py` | screenshots of the BEAR / DBEAR windows (`figures/screen_*.png`) |
| `figures/` | figures included by the report |
| `generated/results.tex` | every number quoted in the text, as LaTeX macros |
| `generated/tab_*.tex` | rows of the result tables |

## Building the PDF

With a TeX distribution (TeX Live, MiKTeX), run from this folder:

```bash
pdflatex report.tex
```

Run it twice, so the cross-references and the table of contents resolve.
`latexmk -pdf report.tex` does both runs in one go.

### On Overleaf

`report.tex` reads its numbers and tables from `generated/` and its images
from `figures/`. To avoid uploading folders, make a flat, single-folder copy.
From the project root:

```bash
python report/flatten_for_overleaf.py
```

This writes `report/overleaf/` and `report_overleaf.zip` (project root):

- one `report.tex` with every number and table written directly into it;
- the 13 images next to it;
- no sub-folders, no `\input` of other files.

On Overleaf, choose *New Project → Upload Project* and select
`report_overleaf.zip`. Alternatively, create a blank project and upload all
14 files of `report/overleaf/` at once. Then press *Recompile*; the default
pdfLaTeX compiler is right.

## Regenerating the results

From the project root:

```bash
python report/make_figures.py
```

This takes about 20 seconds and is fully seeded, so it reproduces the same
numbers. Rebuild the PDF afterwards. The text never contains hand-copied
results: it uses the macros in `generated/results.tex`, so the report always
matches the code.
