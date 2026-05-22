.PHONY: all clean arxiv-bundle figure

all: paper.pdf

paper.pdf: paper.tex references.bib fig/cascade.pdf neurips_2024.sty
	pdflatex -interaction=nonstopmode paper
	bibtex paper
	pdflatex -interaction=nonstopmode paper
	pdflatex -interaction=nonstopmode paper

fig/cascade.pdf: fig/cascade.py
	cd fig && python cascade.py

figure: fig/cascade.pdf

clean:
	rm -f paper.aux paper.log paper.out paper.bbl paper.blg paper.pdf paper.toc

arxiv-bundle: paper.pdf
	rm -rf arxiv_upload heurchain_v4_arxiv.tar.gz
	mkdir -p arxiv_upload/fig
	cp paper.tex references.bib paper.bbl neurips_2024.sty arxiv_upload/
	cp fig/cascade.pdf arxiv_upload/fig/
	tar czf heurchain_v4_arxiv.tar.gz -C arxiv_upload .
	@echo "Bundle ready: heurchain_v4_arxiv.tar.gz"
	@echo "Contents:"
	@tar tzf heurchain_v4_arxiv.tar.gz
