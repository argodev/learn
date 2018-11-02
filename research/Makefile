PANDOC=/usr/bin/pandoc
PANDOC_OPTIONS=--standalone --filter pandoc-citeproc --csl ieee.csl
PANDOC_PDF_OPTIONS=-V geometry:margin=1in

# Targets and dependencies
.PHONY: clean

all : clean abstract proposal questions

clean :
	- $(RM) $(EXPORTED_DOCS)

abstract :
	$(PANDOC) $(PANDOC_OPTIONS) $(PANDOC_PDF_OPTIONS) -o Thesis_Abstract_Draft_T00215814.pdf draft.md

proposal :
	$(PANDOC) $(PANDOC_OPTIONS) $(PANDOC_PDF_OPTIONS) -o Dissertation_Proposal_T00215814.pdf proposal.md

questions :
	$(PANDOC) $(PANDOC_OPTIONS) $(PANDOC_PDF_OPTIONS) -o Draft_Questions_T00215814.pdf questions.md

exam :
	$(PANDOC) $(PANDOC_OPTIONS) $(PANDOC_PDF_OPTIONS) -o Comprehensive_Exam_Q1_T00215814.pdf comprehensive_exam_q1_T00215814.md
	$(PANDOC) $(PANDOC_OPTIONS) $(PANDOC_PDF_OPTIONS) -o Comprehensive_Exam_Q2_T00215814.pdf comprehensive_exam_q2_T00215814.md
	$(PANDOC) $(PANDOC_OPTIONS) $(PANDOC_PDF_OPTIONS) -o Comprehensive_Exam_Q3_T00215814.pdf comprehensive_exam_q3_T00215814.md
