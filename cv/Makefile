# Primary résumé only (the one the site uses):
#   make                    # cv/tex/main.tex → cv/pdf/main.pdf
#   make pdf/academic.pdf   # academic CV variant (preview; not the site download)
#
# Other .tex files in tex/ are for Cursor preview; do not need a PDF unless you
# later want one on the site — then: make pdf/resume-uniA.pdf
#
# Needs latexmk + pdflatex (MacTeX or BasicTeX).

LATEXMK ?= latexmk
LATEXMK_FLAGS ?= -pdf -interaction=nonstopmode -halt-on-error

.PHONY: all clean

all: pdf/main.pdf

pdf/main.pdf: tex/main.tex
	mkdir -p pdf
	cd tex && $(LATEXMK) $(LATEXMK_FLAGS) main.tex
	mv -f tex/main.pdf pdf/main.pdf
	cd tex && $(LATEXMK) -c

pdf/%.pdf: tex/%.tex
	mkdir -p pdf
	cd tex && $(LATEXMK) $(LATEXMK_FLAGS) $*.tex
	mv -f tex/$*.pdf pdf/$*.pdf
	cd tex && $(LATEXMK) -c

clean:
	cd tex && $(LATEXMK) -c
	rm -f tex/*.fls tex/*.fdb_latexmk tex/*.synctex.gz
	rm -f pdf/*.aux pdf/*.log pdf/*.out pdf/*.fls pdf/*.fdb_latexmk pdf/*.synctex.gz
	rm -rf .latex-build
