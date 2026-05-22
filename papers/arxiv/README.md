# HeurChain v4 findings - arXiv submission bundle

Self-contained LaTeX source for the HeurChain v2-v4c findings paper, ready to
compile and submit to arXiv (under `cs.CL` once endorsement clears).

## Files

| File | Purpose |
| --- | --- |
| `paper.tex` | Main LaTeX source (NeurIPS 2024 preprint style, single column). |
| `references.bib` | BibTeX entries for all cited works (7 external + 2 self-cites). |
| `neurips_2024.sty` | Official NeurIPS 2024 style file (downloaded from `media.neurips.cc`). |
| `fig/cascade.py` | matplotlib script that regenerates the cascade figure. |
| `fig/cascade.png` | Cascade figure, 300 dpi raster (web preview). |
| `fig/cascade.pdf` | Cascade figure, vector (embedded by `paper.tex`). |
| `Makefile` | Build, clean, and bundle targets. |
| `README.md` | This file. |

## Quick start (local build)

```bash
# One-shot full build (runs pdflatex + bibtex + pdflatex + pdflatex).
make

# Equivalent to:
pdflatex paper
bibtex   paper
pdflatex paper
pdflatex paper

# Clean intermediates AND paper.pdf.
make clean

# Produce the final arXiv tarball.
make arxiv-bundle
# -> heurchain_v4_arxiv.tar.gz
```

## Dependencies

A standard TeX Live (or MiKTeX) install with the following packages:

- `texlive-latex-base`
- `texlive-latex-recommended`   (`graphicx`, `hyperref`, `xcolor`, `url`)
- `texlive-latex-extra`         (`booktabs`, `cleveref`, `microtype`)
- `texlive-fonts-recommended`
- `texlive-bibtex-extra` and `biber`-free `bibtex` (we use plain `bibtex`)
- `texlive-science`             (`amsmath`, `amssymb`)
- `texlive-publishers`          (provides `neurips_2024.sty` dependencies if any)

On Ubuntu / Debian:

```bash
sudo apt-get install texlive-full   # easiest; or pick the subset above
```

On macOS (MacTeX):

```bash
brew install --cask mactex-no-gui
```

On Overleaf: just upload `paper.tex`, `references.bib`, `neurips_2024.sty`, and
the `fig/` directory; Overleaf auto-detects everything.

## Regenerating the figure

```bash
cd fig
python cascade.py          # writes cascade.png AND cascade.pdf
```

Requires `matplotlib >= 3.4`. No other Python dependencies.

## Customization spots

Edit `paper.tex`:

- **Title** -- line near `\title{...}`.
- **Author block** -- `\author{...}` (currently Peter J. Medina, HeurChain).
- **Bibliography style** -- `\bibliographystyle{plainnat}` is natbib-compatible
  with `\citep` / `\citet`. Switch to `unsrtnat` for citation-order numbering.
- **NeurIPS preprint flag** -- `\usepackage[preprint]{neurips_2024}` keeps the
  paper readable as a preprint (no anonymous-submission box). Drop the
  `[preprint]` option to switch back to anonymous review style.

## arXiv submission notes

- arXiv requires a `cs.CL` endorsement for first-time CL submitters; this bundle
  is ready to upload once that clears.
- `make arxiv-bundle` produces a tarball containing only the files arXiv needs:
  `paper.tex`, `references.bib`, the pre-built `paper.bbl`, `neurips_2024.sty`,
  and `fig/cascade.pdf`. (arXiv compiles with `pdflatex` only, so the `.bbl`
  must be included so they don't have to run `bibtex`.)
- Make sure to run `make` at least once before `make arxiv-bundle` so the
  `.bbl` file exists.
