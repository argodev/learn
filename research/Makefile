PANDOC=/usr/bin/pandoc
PANDOC_OPTIONS=--standalone --filter pandoc-citeproc --csl ieee.csl
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

exam :
	$(PANDOC) $(PANDOC_OPTIONS) $(PANDOC_PDF_OPTIONS) -o Comprehensive_Exam_T00215814.pdf comprehensive_exam_T00215814.md
