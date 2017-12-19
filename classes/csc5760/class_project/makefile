
PANDOC=/usr/bin/pandoc
PANDOC_OPTIONS=--standalone 
PANDOC_PDF_OPTIONS=-V geometry:margin=1in --pdf-engine=xelatex
PANDOC_PDF_SLIDS_OPTIONS=--pdf-engine=xelatex


#pandoc -f markdown+smart

# Targets and dependencies
.PHONY: clean paper slides notes

all : clean paper slides notes

clean :
	- $(RM) $(EXPORTED_DOCS)

paper :
	$(PANDOC) $(PANDOC_OPTIONS) $(PANDOC_PDF_OPTIONS) -o 5760-Paper_T00215814.pdf 5760-Paper_T00215814.md

slides:
	$(PANDOC) $(PANDOC_OPTIONS) $(PANDOC_PDF_SLIDS_OPTIONS) -t beamer 5760-Slides_T00215814.md -o 5760-Slides_T00215814.pdf --slide-level 2
# -V theme:Madrid

notes:
	$(PANDOC) $(PANDOC_OPTIONS) $(PANDOC_PDF_OPTIONS) -t beamer 5760-Slides_T00215814.md -o 5760-Notes_T00215814.pdf --slide-level 2 -V handout
	pdfnup 5760-Notes_T00215814.pdf --nup 1x2 --no-landscape --keepinfo --paper letterpaper --frame true --scale 0.9 --suffix "nup"
	mv 5760-Notes_T00215814-nup.pdf 5760-Notes_T00215814.pdf
