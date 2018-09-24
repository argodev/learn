PANDOC=/usr/bin/pandoc
PANDOC_OPTIONS=--standalone
PANDOC_PDF_OPTIONS=-V geometry:margin=1in

# Targets and dependencies
.PHONY: clean

all : clean abstract questions

clean :
	- $(RM) $(EXPORTED_DOCS)

abstract :
	$(PANDOC) $(PANDOC_OPTIONS) $(PANDOC_PDF_OPTIONS) -o Thesis_Abstract_Draft_T00215814.pdf draft.md

questions :
	$(PANDOC) $(PANDOC_OPTIONS) $(PANDOC_PDF_OPTIONS) -o Draft_Questions_T00215814.pdf questions.md
